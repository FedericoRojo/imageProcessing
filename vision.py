"""vision.py — imagen → LLM de visión → bounding box de la tabla de items.

La única pregunta que responde este módulo es *dónde está la tabla*. No recorta,
no endereza y no lee texto: devuelve un rectángulo en coordenadas relativas
0-1000 y se va. El recorte lo hace `imagen.py`.

Es también el dueño del cliente de la API, porque es lo único del repo que
habla con un LLM. `esquema.py` le pide `get_client` con un import tardío para
la variante de mapeo de encabezado por LLM.

    from vision import detectar_tabla

    bbox = detectar_tabla("factura2.jpeg")   # None si no la encuentra

Lo que hay que tener presente al tocarlo: el modelo es chico y falla de dos
maneras conocidas — escribe mal las claves del JSON (por eso
`_normalizar_claves`) y devuelve el marco de la imagen entera cuando la tabla
le cuesta (por eso `TablaBBox.es_razonable` y el refuerzo del prompt).
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

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

MODELO_VISION = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
BASE_URL = "https://integrate.api.nvidia.com/v1"


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
# El recuadro y su validación
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
# La llamada
# --------------------------------------------------------------------------

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
