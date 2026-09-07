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
import functools
import mimetypes
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

MODELO_VISION = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
BASE_URL = "https://integrate.api.nvidia.com/v1"

MARGEN_DEFAULT = 0.02   # % del alto/ancho que se agrega alrededor del bbox
ZOOM_DEFAULT = 2        # factor de ampliación del recorte


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
    contiene_tabla: bool = Field(description="false si no se distingue una tabla clara")
    x_min: float = Field(description="0-1000, borde izquierdo de la tabla de items")
    y_min: float = Field(description="0-1000, arranca en el encabezado de columnas")
    x_max: float
    y_max: float


PROMPT_BBOX = (
    "Devolvé el bounding box de la tabla de items/productos de esta factura, "
    "en coordenadas relativas 0-1000. Incluí la fila de encabezados de columna "
    "pero NO el logo, los datos del proveedor ni el recuadro de totales.\n"
    "Respondé SOLO con este JSON, sin markdown ni texto adicional:\n"
    '{"contiene_tabla": true, "x_min": 0, "y_min": 0, "x_max": 1000, "y_max": 1000}'
)


def _data_url(ruta: Path) -> str:
    mime = mimetypes.guess_type(str(ruta))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(ruta.read_bytes()).decode()}"


def detectar_tabla(ruta: str | Path, verbose: bool = True) -> TablaBBox | None:
    """Imagen → LLM → bbox de la tabla. Devuelve None si no la encuentra."""
    ruta = Path(ruta)
    completion = get_client().chat.completions.create(
        model=MODELO_VISION,
        messages=[
            {"role": "system", "content": PROMPT_BBOX},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_url(ruta)}}
            ]},
        ],
        max_tokens=512,
        temperature=0.0,
        extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
    )

    raw = completion.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        bbox = TablaBBox.model_validate_json(raw)
    except Exception:
        if verbose:
            print(f"[detectar_tabla] no pude parsear el bbox. Crudo: {raw[:200]}")
        return None

    if not bbox.contiene_tabla:
        if verbose:
            print("[detectar_tabla] el modelo dice que no hay tabla.")
        return None
    return bbox


def recortar_zoom(
    ruta: str | Path,
    bbox: TablaBBox,
    margen: float = MARGEN_DEFAULT,
    zoom: int = ZOOM_DEFAULT,
) -> Path:
    """bbox → imagen recortada y ampliada, guardada como <nombre>_tabla.<ext>."""
    ruta = Path(ruta)
    img = Image.open(ruta)
    w, h = img.size
    x0 = max(0.0, bbox.x_min / 1000 - margen) * w
    y0 = max(0.0, bbox.y_min / 1000 - margen) * h
    x1 = min(1.0, bbox.x_max / 1000 + margen) * w
    y1 = min(1.0, bbox.y_max / 1000 + margen) * h

    crop = img.crop((x0, y0, x1, y1))
    crop = crop.resize((int((x1 - x0) * zoom), int((y1 - y0) * zoom)), Image.LANCZOS)

    salida = ruta.with_stem(ruta.stem + "_tabla")
    crop.save(salida)
    return salida


def preparar_tabla(
    ruta: str | Path,
    margen: float = MARGEN_DEFAULT,
    zoom: int = ZOOM_DEFAULT,
    verbose: bool = True,
) -> Path:
    """Paso 1 completo: imagen → llm → tabla recortada.

    Si el LLM no encuentra la tabla, devuelve la imagen original para que el
    pipeline siga funcionando igual.
    """
    ruta = Path(ruta)
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
    return recortar_zoom(ruta, bbox, margen=margen, zoom=zoom)


# --------------------------------------------------------------------------
# Paso 2 — tabla recortada → OCR
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def get_ocr(lang: str = "es", unwarping: bool = True, rotacion: bool = False):
    """Instancia de PaddleOCR cacheada (crearla es lento: carga los modelos)."""
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        text_det_unclip_ratio=2.5,
        enable_mkldnn=False,
        use_doc_orientation_classify=rotacion,   # True si la foto puede venir rotada
        use_doc_unwarping=unwarping,             # True si la hoja está curvada
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
# Conveniencia
# --------------------------------------------------------------------------

def procesar(ruta: str | Path, margen: float = MARGEN_DEFAULT, zoom: int = ZOOM_DEFAULT):
    """Corre el pipeline entero y devuelve (img_tabla, res, filas, df)."""
    img_tabla = preparar_tabla(ruta, margen=margen, zoom=zoom)
    res = ocr_tabla(img_tabla)
    filas = agrupar_filas(res)
    return img_tabla, res, filas, filas_a_dataframe(filas)
