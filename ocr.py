"""ocr.py — imagen de la tabla → texto con coordenadas. Es lo único que corre PaddleOCR.

Todo lo que viene después (agrupar las cajas en filas, ponerle nombre a las
columnas) trabaja sobre el `res` que devuelve `ocr_tabla`, que es un diccionario
con `rec_polys`, `rec_texts` y `rec_scores`. Por eso `cargar_ocr_json` vive acá:
levanta una corrida guardada y produce exactamente lo mismo, así el resto del
pipeline se prueba sin GPU.

`paddleocr` se importa adentro de `get_ocr` a propósito: `import ocr` no
arrastra paddle, y `banco.py` puede replayear salidas guardadas en una máquina
que no lo tiene instalado.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pandas as pd

UNCLIP_DEFAULT = 2.5    # dilatación de las cajas del detector de PaddleOCR


@functools.lru_cache(maxsize=8)
def get_ocr(lang: str = "es", unwarping: bool = False, rotacion: bool = False,
            unclip: float = UNCLIP_DEFAULT):
    """Instancia de PaddleOCR cacheada (crearla es lento: carga los modelos).

    `unwarping` viene en False porque UVDoc estaba truncando el primer dígito
    de la columna ARTICULO. OJO: en ese mismo cambio se agregó el borde blanco
    de 30 px, que ataca la causa que el commit describe — el detector recortando
    las cajas contra el límite de la imagen. Las dos cosas nunca se probaron por
    separado, así que puede que hoy unwarping sea seguro. Vale medirlo, porque
    es lo único que corrige la deformación de perspectiva: en factura2 la
    pendiente de los renglones varía con la altura de la página, y una rotación
    rígida (que es lo que hace `enderezar`) por definición no puede arreglarla.

    `unclip` dilata las cajas del detector. Con el 2.5 actual el alto mediano de
    caja es de 34-38 px contra un paso entre filas de 27 px: las cajas son más
    altas que la separación entre renglones y el 31% de las cajas de factura2 se
    solapan verticalmente con las de otra fila. Eso es lo que hacía que
    `tol = 0.6 * alto` en `agrupar_filas` fuera demasiado permisivo. Bajarlo
    ajusta las cajas, pero puede volver a cortar caracteres: medilo con
    `comparar_variantes` antes de cambiar el default.
    """
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        text_det_unclip_ratio=unclip,
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


def cargar_ocr_json(ruta: str | Path) -> dict:
    """Levanta un `res` guardado por `res.save_to_json()` de una corrida previa.

    Sirve para iterar sobre las capas de agrupado y estandarización sin volver
    a correr PaddleOCR ni la API: el OCR es determinista para una imagen dada,
    así que guardar su salida una vez y replayearla es equivalente y gratis.
    """
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if "res" in datos and isinstance(datos["res"], dict):
        datos = datos["res"]           # algunas versiones envuelven en {"res": ...}
    faltan = [k for k in ("rec_polys", "rec_texts", "rec_scores") if k not in datos]
    if faltan:
        raise ValueError(f"al json le faltan {faltan}: no parece salida de PaddleOCR")
    return datos
