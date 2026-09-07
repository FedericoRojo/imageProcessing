"""Pipeline de extracción de tablas de facturas argentinas.

    imagen original
       ↓  detectar_tabla()   LLM de visión → bounding box de la tabla de items
       ↓  recortar_zoom()    recorte + ampliación
    imagen de la tabla
       ↓  ocr_tabla()        PaddleOCR → cajas de texto con coordenadas
       ↓  agrupar_filas()    agrupado por coordenada y
    filas / DataFrame

Uso típico:

    from pipeline import preparar_tabla, ocr_tabla, agrupar_filas, filas_a_dataframe

    img_tabla = preparar_tabla("factura2.jpeg")
    res       = ocr_tabla(img_tabla)
    filas     = agrupar_filas(res)
    df        = filas_a_dataframe(filas)
"""

from __future__ import annotations

import base64
import difflib
import functools
import json
import mimetypes
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

MODELO_VISION = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
BASE_URL = "https://integrate.api.nvidia.com/v1"

MARGEN_DEFAULT = 0.02   # % del alto/ancho que se agrega alrededor del bbox
ZOOM_DEFAULT = 2        # factor de ampliación del recorte
BORDE_DEFAULT = 30      # px de borde blanco agregados DESPUÉS del zoom

# Por qué el borde blanco: el detector de texto de PaddleOCR recorta las cajas
# contra el límite de la imagen, y el preprocesador (unwarping) puede correr la
# imagen unos píxeles. Cualquiera de las dos cosas se come el primer carácter de
# la columna más a la izquierda. Con un borde blanco, ningún texto queda pegado
# al borde y el problema desaparece.


def _api_key() -> str:
    """Busca la key en variable de entorno y, si no, en los secrets de Colab."""
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("NVIDIA_API_KEY")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No encontré NVIDIA_API_KEY. En Colab cargala en los secrets "
            "(ícono de la llave); local, exportala como variable de entorno."
        ) from e


@functools.lru_cache(maxsize=1)
def get_client():
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=_api_key())


# --------------------------------------------------------------------------
# Paso 1 — imagen → LLM → bounding box de la tabla
# --------------------------------------------------------------------------

class TablaBBox(BaseModel):
    # Con default en True: si el modelo devuelve las 4 coordenadas pero se come
    # o escribe mal la bandera, igual aprovechamos el bbox.
    contiene_tabla: bool = True
    x_min: float = Field(description="0-1000, borde izquierdo de la tabla de items")
    y_min: float = Field(description="0-1000, arranca en el encabezado de columnas")
    x_max: float
    y_max: float

    @property
    def ancho(self) -> float:
        return self.x_max - self.x_min

    @property
    def alto(self) -> float:
        return self.y_max - self.y_min

    @property
    def cobertura(self) -> float:
        """Fracción de la imagen que cubre el recuadro (0 a 1)."""
        return max(0.0, self.ancho / 1000) * max(0.0, self.alto / 1000)

    def es_razonable(self, ancho_min: float = 150.0, alto_min: float = 15.0,
                     cobertura_max: float = 0.90) -> tuple[bool, str]:
        """Valida el recuadro por los dos extremos. Devuelve (ok, motivo).

        - Muy chico o invertido: el modelo erró de zona.
        - Casi toda la imagen: el modelo se rindió y devolvió el marco entero
          (típicamente copiando el ejemplo del prompt). Recortar ahí no recorta
          nada, así que es tan inútil como no detectar.

        El umbral chico va por ancho y alto y no por área, porque una tabla de
        un solo ítem es legítimamente muy baja.
        """
        if self.ancho < ancho_min or self.alto < alto_min:
            return False, f"muy chico o invertido ({self.ancho:.0f}x{self.alto:.0f})"
        if self.cobertura > cobertura_max:
            return False, (f"cubre el {self.cobertura*100:.0f}% de la imagen: "
                           "es la imagen entera, no un recorte")
        return True, "ok"


_CAMPOS = ("contiene_tabla", "x_min", "y_min", "x_max", "y_max")


def _normalizar_claves(d: dict) -> dict:
    """Tolera erratas del modelo: 'contene_tabla', 'xmin', 'X_MIN', etc.

    Los modelos de visión chicos escriben mal las claves cada tantas llamadas.
    Rechazar la respuesta entera por una letra sale más caro que mapearla.
    """
    salida = {}
    for clave, valor in d.items():
        k = str(clave).strip().lower().replace("-", "_").replace(" ", "_")
        if k in _CAMPOS:
            salida[k] = valor
            continue
        # 'xmin' -> 'x_min'
        k2 = re.sub(r"^([xy])(min|max)$", r"\1_\2", k)
        if k2 in _CAMPOS:
            salida[k2] = valor
            continue
        cerca = difflib.get_close_matches(k, _CAMPOS, n=1, cutoff=0.72)
        if cerca:
            salida[cerca[0]] = valor
    return salida


def _parsear_bbox(raw: str, verbose: bool = True) -> TablaBBox | None:
    """Texto crudo del modelo → TablaBBox, aguantando ruido alrededor."""
    texto = raw.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", texto, re.S)     # el primer objeto JSON que aparezca
    if not m:
        if verbose:
            print(f"[bbox] no hay JSON en la respuesta: {texto[:200]}")
        return None
    try:
        crudo = json.loads(m.group())
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[bbox] JSON inválido ({e}): {m.group()[:200]}")
        return None

    datos = _normalizar_claves(crudo)
    faltan = [c for c in ("x_min", "y_min", "x_max", "y_max") if c not in datos]
    if faltan:
        if verbose:
            print(f"[bbox] faltan coordenadas {faltan} en {crudo}")
        return None

    try:
        bbox = TablaBBox.model_validate(datos)
    except Exception as e:
        if verbose:
            print(f"[bbox] no valida: {e}")
        return None

    # el modelo a veces devuelve valores fuera de rango
    bbox.x_min = max(0.0, min(1000.0, bbox.x_min))
    bbox.y_min = max(0.0, min(1000.0, bbox.y_min))
    bbox.x_max = max(0.0, min(1000.0, bbox.x_max))
    bbox.y_max = max(0.0, min(1000.0, bbox.y_max))
    return bbox


# OJO con este prompt: la versión anterior traía un ejemplo con valores
# concretos (0, 0, 1000, 1000) y el modelo lo copiaba textual cuando la tabla
# le costaba, devolviendo la imagen entera. Los ejemplos con números son un
# imán para los modelos chicos: van con marcadores de posición, no con valores.
PROMPT_BBOX = (
    "Devolvé el bounding box de la tabla de items/productos de esta factura, "
    "en coordenadas relativas 0-1000 (0 = borde superior o izquierdo, "
    "1000 = borde inferior o derecho).\n"
    "Incluí la fila de encabezados de columna. NO incluyas el logo, los datos "
    "del proveedor, el bloque del cliente ni el recuadro de totales.\n"
    "La tabla puede tener una sola fila y puede ser una franja fina y ancha: "
    "eso es normal, devolvé igual su recuadro ajustado.\n"
    "NO devuelvas el marco de la imagen completa: un recuadro que cubre casi "
    "toda la imagen es una respuesta incorrecta.\n"
    "Si de verdad no distinguís una tabla de items, poné contiene_tabla en false.\n"
    "Respondé SOLO con un objeto JSON con esta forma, sin markdown ni texto:\n"
    '{"contiene_tabla": <true|false>, "x_min": <numero>, "y_min": <numero>, '
    '"x_max": <numero>, "y_max": <numero>}'
)


# --------------------------------------------------------------------------
# Paso 0 — orientación
# --------------------------------------------------------------------------

def normalizar_orientacion(ruta: str | Path, rotacion: int = 0,
                           verbose: bool = True) -> Path:
    """Deja la imagen con el texto derecho ANTES de que la vea nadie.

    Las fotos de celular suelen venir con la etiqueta EXIF de orientación: el
    visor de fotos la respeta, pero PIL y OpenCV leen los píxeles crudos. Así
    que el archivo se ve derecho en el explorador y acostado en el pipeline.
    Acá se aplica la rotación de una vez y se guarda un archivo nuevo, para que
    el LLM y PaddleOCR vean exactamente lo mismo.

    `rotacion` fuerza un giro extra en grados antihorarios (90, 180, 270) para
    los casos en que la foto está girada y no trae EXIF.
    """
    ruta = Path(ruta)
    img = Image.open(ruta)
    original = img.size

    img = ImageOps.exif_transpose(img)     # aplica la etiqueta EXIF si la hay
    giro_exif = img.size != original

    if rotacion % 360:
        img = img.rotate(rotacion, expand=True)

    if not giro_exif and not rotacion % 360:
        return ruta                        # nada que hacer, uso el original

    salida = ruta.with_name(ruta.stem + "_derecha.png")
    img.convert("RGB").save(salida)
    if verbose:
        motivo = []
        if giro_exif:
            motivo.append("EXIF")
        if rotacion % 360:
            motivo.append(f"{rotacion}° manual")
        print(f"[orientacion] {original} -> {img.size} por {' + '.join(motivo)}")
    return salida


def detectar_rotacion(ruta: str | Path, verbose: bool = True) -> int:
    """Prueba las 4 orientaciones y devuelve la que el OCR lee con más confianza.

    Para fotos giradas que NO traen EXIF. Cuesta 4 pasadas de OCR, así que se
    usa sólo cuando `normalizar_orientacion` no alcanzó y el resultado vino mal.
    """
    import tempfile

    base = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    mejor, mejor_score = 0, -1.0
    for ang in (0, 90, 180, 270):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            base.rotate(ang, expand=True).save(f.name)
            res = get_ocr().predict(f.name)[0]
        scores = list(res["rec_scores"])
        # confianza media ponderada por cuánto texto encontró
        score = (sum(scores) / len(scores)) * len(scores) ** 0.5 if scores else 0.0
        if verbose:
            print(f"[rotacion] {ang:3d}°: {len(scores)} cajas, score {score:.2f}")
        if score > mejor_score:
            mejor, mejor_score = ang, score
    if verbose:
        print(f"[rotacion] elegida: {mejor}°")
    return mejor


def _data_url(ruta: Path) -> str:
    mime = mimetypes.guess_type(str(ruta))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(ruta.read_bytes()).decode()}"


def _pedir_bbox(ruta: Path, refuerzo: str = "") -> str:
    completion = get_client().chat.completions.create(
        model=MODELO_VISION,
        messages=[
            {"role": "system", "content": PROMPT_BBOX + refuerzo},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_url(ruta)}}
            ]},
        ],
        max_tokens=512,
        temperature=0.0,
        extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
    )
    return completion.choices[0].message.content.strip()


def detectar_tabla(ruta: str | Path, verbose: bool = True,
                   reintentos: int = 1) -> TablaBBox | None:
    """Imagen → LLM → bbox de la tabla. Devuelve None si no la encuentra."""
    ruta = Path(ruta)
    refuerzo = ""
    for intento in range(reintentos + 1):
        bbox = _parsear_bbox(_pedir_bbox(ruta, refuerzo), verbose=verbose)

        if bbox is None:
            refuerzo = ("\nATENCIÓN: usá EXACTAMENTE las claves contiene_tabla, "
                        "x_min, y_min, x_max, y_max. Nada más.")
            if verbose and intento < reintentos:
                print("[detectar_tabla] reintento con las claves reforzadas...")
            continue

        if not bbox.contiene_tabla:
            if verbose:
                print("[detectar_tabla] el modelo dice que no hay tabla.")
            return None

        ok, motivo = bbox.es_razonable()
        if not ok:
            if verbose:
                print(f"[detectar_tabla] bbox descartado: {motivo}")
            if bbox.cobertura > 0.90:
                refuerzo = ("\nTu respuesta anterior fue el marco de la imagen entera, "
                            "que no sirve. Mirá dónde están las filas de items con sus "
                            "códigos y precios, y devolvé SÓLO ese rectángulo, ajustado. "
                            "Si no distinguís la tabla, poné contiene_tabla en false.")
            else:
                refuerzo = ("\nEl recuadro anterior era demasiado chico. Devolvé el "
                            "rectángulo COMPLETO de la tabla de items.")
            continue

        return bbox

    if verbose:
        print("[detectar_tabla] sin recuadro válido después de "
              f"{reintentos + 1} intento(s)")
    return None


def _margenes(margen: float | tuple[float, float, float, float]):
    """Acepta un número (los 4 lados iguales) o (izq, arriba, der, abajo)."""
    if isinstance(margen, (int, float)):
        return (float(margen),) * 4
    izq, arr, der, aba = margen
    return float(izq), float(arr), float(der), float(aba)


def recortar_zoom(
    ruta: str | Path,
    bbox: TablaBBox,
    margen: float | tuple[float, float, float, float] = MARGEN_DEFAULT,
    zoom: int = ZOOM_DEFAULT,
    borde: int = BORDE_DEFAULT,
) -> Path:
    """bbox → imagen recortada, ampliada y con borde blanco.

    `margen` puede ser un número (los 4 lados iguales) o una tupla
    (izquierda, arriba, derecha, abajo) en fracción del alto/ancho original.
    `borde` son píxeles de blanco que se agregan alrededor después del zoom,
    para que ningún texto quede pegado al borde de la imagen.
    """
    ruta = Path(ruta)
    img = Image.open(ruta)
    w, h = img.size
    m_izq, m_arr, m_der, m_aba = _margenes(margen)

    x0 = max(0.0, bbox.x_min / 1000 - m_izq) * w
    y0 = max(0.0, bbox.y_min / 1000 - m_arr) * h
    x1 = min(1.0, bbox.x_max / 1000 + m_der) * w
    y1 = min(1.0, bbox.y_max / 1000 + m_aba) * h

    crop = img.crop((x0, y0, x1, y1))
    crop = crop.resize((int((x1 - x0) * zoom), int((y1 - y0) * zoom)), Image.LANCZOS)
    if borde:
        crop = ImageOps.expand(crop.convert("RGB"), border=int(borde), fill="white")

    # PNG y no JPEG: es una imagen intermedia y recomprimir en JPEG le agrega
    # ruido justo en los bordes de los caracteres, que es lo que lee el OCR.
    salida = ruta.with_name(ruta.stem + "_tabla.png")
    crop.save(salida)
    return salida


def preparar_tabla(
    ruta: str | Path,
    margen: float | tuple[float, float, float, float] = MARGEN_DEFAULT,
    zoom: int = ZOOM_DEFAULT,
    borde: int = BORDE_DEFAULT,
    rotacion: int = 0,
    verbose: bool = True,
) -> Path:
    """Paso 1 completo: imagen → orientación → llm → tabla recortada.

    Si el LLM no encuentra la tabla, devuelve la imagen original para que el
    pipeline siga funcionando igual.
    """
    ruta = normalizar_orientacion(ruta, rotacion=rotacion, verbose=verbose)
    bbox = detectar_tabla(ruta, verbose=verbose)
    if bbox is None:
        if verbose:
            print("[preparar_tabla] sin bbox: sigo con la imagen completa.")
        return ruta
    if verbose:
        print(
            f"[preparar_tabla] bbox ({bbox.x_min:.0f}, {bbox.y_min:.0f}) - "
            f"({bbox.x_max:.0f}, {bbox.y_max:.0f})"
        )
    return recortar_zoom(ruta, bbox, margen=margen, zoom=zoom, borde=borde)


# --------------------------------------------------------------------------
# Paso 2 — tabla recortada → OCR
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def get_ocr(lang: str = "es", unwarping: bool = False, rotacion: bool = False):
    """Instancia de PaddleOCR cacheada (crearla es lento: carga los modelos).

    `unwarping` viene en False a propósito. El modelo de unwarping (UVDoc) está
    pensado para páginas enteras curvadas o arrugadas; aplicado a un recorte
    ajustado de la tabla, deforma la imagen y le come píxeles de los bordes.
    Eso es lo que estaba truncando el primer dígito de la columna ARTICULO.
    Si trabajás con la factura completa y sale curvada, ahí sí puede servir.
    """
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        text_det_unclip_ratio=2.5,
        enable_mkldnn=False,
        use_doc_orientation_classify=rotacion,   # True si la foto puede venir rotada
        use_doc_unwarping=unwarping,
        use_textline_orientation=False,
    )


def ocr_tabla(ruta: str | Path, guardar_en: str | None = "output", **kw):
    """Tabla recortada → OCR. Devuelve el objeto result de PaddleOCR."""
    res = get_ocr(**kw).predict(str(ruta))[0]
    if guardar_en:
        res.save_to_img(guardar_en)    # imagen con las cajas dibujadas
        res.save_to_json(guardar_en)   # texto + coordenadas + score
    return res


def texto_con_score(res) -> pd.DataFrame:
    """Tabla plana de lo que leyó el OCR, ordenada por score (para debug)."""
    return pd.DataFrame(
        {"texto": res["rec_texts"], "score": res["rec_scores"]}
    ).sort_values("score")


# --------------------------------------------------------------------------
# Paso 3 — OCR → filas
# --------------------------------------------------------------------------

def _cajas(res) -> list[dict]:
    cajas = []
    for poly, txt, score in zip(res["rec_polys"], res["rec_texts"], res["rec_scores"]):
        p = np.array(poly)
        cajas.append({
            "x": float(p[:, 0].min()),
            "y": float(p[:, 1].mean()),
            "alto": float(p[:, 1].max() - p[:, 1].min()),
            "txt": txt,
            "score": float(score),
        })
    return cajas


def agrupar_filas(res, tol: float | None = None) -> list[list[dict]]:
    """Agrupa las cajas de texto en filas por coordenada y.

    Si `tol` es None se calcula como el 60% del alto mediano de las cajas, así
    no hay que reajustarlo a mano cada vez que cambia el zoom del recorte.
    """
    cajas = _cajas(res)
    if not cajas:
        return []

    if tol is None:
        tol = 0.6 * float(np.median([c["alto"] for c in cajas]))

    cajas.sort(key=lambda d: d["y"])

    filas: list[list[dict]] = []
    actual: list[dict] = [cajas[0]]
    for c in cajas[1:]:
        # comparo contra el promedio de la fila en curso, no contra el primer
        # elemento: evita que una fila se corte por una caja apenas desplazada
        y_fila = sum(d["y"] for d in actual) / len(actual)
        if abs(c["y"] - y_fila) <= tol:
            actual.append(c)
        else:
            filas.append(sorted(actual, key=lambda d: d["x"]))
            actual = [c]
    filas.append(sorted(actual, key=lambda d: d["x"]))
    return filas


def filas_a_dataframe(filas: list[list[dict]]) -> pd.DataFrame:
    """Filas → DataFrame crudo (una columna por posición, todavía sin mapear)."""
    return pd.DataFrame([[c["txt"] for c in f] for f in filas])


def imprimir_filas(filas: list[list[dict]]) -> None:
    for f in filas:
        print(" | ".join(c["txt"] for c in f))


# --------------------------------------------------------------------------
# Diagnóstico
# --------------------------------------------------------------------------

def previsualizar_bbox(ruta: str | Path, rotacion: int = 0, verbose: bool = True):
    """Dibuja sobre la imagen el recuadro que devolvió el LLM.

    Sirve para ver qué seleccionó realmente antes de recortar: si agarró toda
    la factura, si se comió el encabezado de columnas o si erró de zona.
    Devuelve (imagen_anotada, bbox) — el bbox es None si no detectó nada.
    """
    from PIL import ImageDraw

    ruta = normalizar_orientacion(ruta, rotacion=rotacion, verbose=verbose)
    bbox = detectar_tabla(ruta, verbose=verbose)
    img = Image.open(ruta).convert("RGB")
    if bbox is None:
        if verbose:
            print("[previsualizar] el LLM no devolvió bbox")
        return img, None

    w, h = img.size
    caja = (bbox.x_min / 1000 * w, bbox.y_min / 1000 * h,
            bbox.x_max / 1000 * w, bbox.y_max / 1000 * h)
    d = ImageDraw.Draw(img)
    d.rectangle(caja, outline=(255, 0, 0), width=max(2, w // 300))

    if verbose:
        alto_rel = (bbox.y_max - bbox.y_min) / 10
        print(f"[previsualizar] la tabla ocupa {alto_rel:.1f}% del alto de la imagen")
        if alto_rel < 12:
            print("  OJO: es una franja fina. El error relativo del modelo pesa "
                  "mucho más acá; conviene subir el zoom o recortar primero la hoja.")
    return img, bbox


def cajas_en_borde(res, ruta_img: str | Path, umbral: int = 3) -> pd.DataFrame:
    """Cajas de texto que tocan el borde de la imagen.

    Si una caja arranca a 0-3 px del borde izquierdo, es muy probable que le
    hayan cortado el primer carácter. Sirve para confirmar el problema en vez
    de adivinarlo.
    """
    w, h = Image.open(ruta_img).size
    filas = []
    for poly, txt in zip(res["rec_polys"], res["rec_texts"]):
        p = np.array(poly)
        x0, x1 = p[:, 0].min(), p[:, 0].max()
        y0, y1 = p[:, 1].min(), p[:, 1].max()
        toca = []
        if x0 <= umbral:
            toca.append("izq")
        if y0 <= umbral:
            toca.append("arriba")
        if x1 >= w - umbral:
            toca.append("der")
        if y1 >= h - umbral:
            toca.append("abajo")
        if toca:
            filas.append({"texto": txt, "toca": ",".join(toca), "x0": int(x0), "x1": int(x1)})
    return pd.DataFrame(filas)


def comparar_variantes(ruta: str | Path, variantes: dict[str, dict],
                       patron: str | None = None) -> pd.DataFrame:
    """Corre el OCR con distintas configuraciones y compara lo que lee.

        comparar_variantes(img, {
            "con unwarping": {"unwarping": True},
            "sin unwarping": {"unwarping": False},
        }, patron=r"\\d{4,8}")
    """
    import re

    salida = {}
    for nombre, kw in variantes.items():
        res = ocr_tabla(ruta, guardar_en=None, **kw)
        textos = list(res["rec_texts"])
        if patron:
            textos = [t for t in textos if re.fullmatch(patron, t)]
        salida[nombre] = textos
    largo = max((len(v) for v in salida.values()), default=0)
    return pd.DataFrame({k: v + [""] * (largo - len(v)) for k, v in salida.items()})


# --------------------------------------------------------------------------
# Conveniencia
# --------------------------------------------------------------------------

def procesar(ruta: str | Path, margen=MARGEN_DEFAULT, zoom: int = ZOOM_DEFAULT,
             borde: int = BORDE_DEFAULT, rotacion: int = 0):
    """Corre el pipeline entero y devuelve (img_tabla, res, filas, df)."""
    img_tabla = preparar_tabla(ruta, margen=margen, zoom=zoom, borde=borde,
                               rotacion=rotacion)
    res = ocr_tabla(img_tabla)
    filas = agrupar_filas(res)
    return img_tabla, res, filas, filas_a_dataframe(filas)
