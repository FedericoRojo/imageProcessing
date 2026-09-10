"""Pipeline de extracción de tablas de facturas argentinas.

    imagen original
       ↓  detectar_tabla()   LLM de visión → bounding box de la tabla de items
       ↓  recortar_zoom()    recorte + enderezado + ampliación
    imagen de la tabla
       ↓  ocr_tabla()        PaddleOCR → cajas de texto con coordenadas
       ↓  agrupar_filas()    agrupado por coordenada y
    filas
       ↓  estandarizar()     cortes de columna → items nombrados
    tabla de items + validación aritmética

Uso típico:

    from pipeline import preparar_tabla, ocr_tabla, agrupar_filas, estandarizar

    img_tabla   = preparar_tabla("factura2.jpeg")  # deja el recorte en recortes/
    res         = ocr_tabla(img_tabla)
    filas       = agrupar_filas(res)
    doc, avisos = estandarizar(filas)
    df          = items_a_dataframe(doc)

Los recortes del paso 1 quedan guardados en `recortes/`. `descargar_recortes()`
se los lleva a tu máquina (en Colab dispara la descarga, zipeando si hay varios).

Qué hace este archivo y qué no
------------------------------
Acá quedan sólo dos cosas: la **orquestación** (`preparar_tabla`, `procesar`,
que son los que encadenan pasos de módulos distintos) y la **fachada**, para
que el notebook siga tocando un módulo solo. La lógica vive en:

    vision.py       imagen → LLM → bounding box de la tabla
    imagen.py       orientación, enderezado, recorte + zoom, carpeta recortes/
    ocr.py          PaddleOCR (lo único que importa paddle)
    filas.py        cajas con coordenadas → filas
    diagnostico.py  herramientas para mirar qué pasó cuando algo sale mal
    esquema.py      dónde empieza cada columna + el esquema canónico (paso 5)
    encabezado.py   cómo se llama cada columna (paso 5)

Si vas a cambiar un umbral o un default, el lugar es el módulo, no este archivo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Re-exports. El notebook, el README y `banco.py` importan estos nombres desde
# `pipeline`, y siguen resolviendo desde acá aunque el código viva en otro lado.
# OJO en Colab: `importlib.reload(pipeline)` re-ejecuta estos imports, así que
# `pipeline` tiene que recargarse DESPUÉS de los módulos que trae.
from diagnostico import (cajas_en_borde, comparar_variantes, detectar_rotacion,
                         previsualizar_bbox)
from filas import (agrupar_filas, filas_a_dataframe, imprimir_filas,
                   inclinacion_de_cajas, paso_filas)
from imagen import (BORDE_DEFAULT, MARGEN_DEFAULT, RECORTES_DIR, ZOOM_DEFAULT,
                    descargar_recortes, enderezar, estimar_inclinacion,
                    listar_recortes, normalizar_orientacion, recortar_zoom)
from ocr import (UNCLIP_DEFAULT, cargar_ocr_json, get_ocr, ocr_tabla,
                 texto_con_score)
from vision import (BASE_URL, MODELO_VISION, PROMPT_BBOX, TablaBBox,
                    detectar_tabla, get_client)

# El paso 5 vive afuera: `esquema` y `encabezado` no dependen de paddle ni de
# openai, y así se pueden probar y medir sin GPU ni API key. Se importa el
# módulo, no sus funciones: en Colab un `reload(esquema)` después de un git pull
# se ve desde acá sin tener que recargar también `pipeline`.
# No hay ciclo: `esquema` le pide `get_client` a `vision`, que no importa nada
# del repo.
import esquema

# La superficie pública, explícita: son re-exports, así que sin esto cualquier
# linter los marca como imports sin usar. Sirve además como contrato — lo que
# está acá es lo que el notebook y el README pueden llamar como `pipeline.algo`.
__all__ = [
    # orquestación y fachada del paso 5 (definidos en este archivo)
    "preparar_tabla", "procesar", "estandarizar", "items_a_dataframe",
    "guardar_doc", "validar_aritmetica", "imprimir_avisos",
    # vision.py
    "TablaBBox", "detectar_tabla", "get_client",
    "MODELO_VISION", "BASE_URL", "PROMPT_BBOX",
    # imagen.py
    "normalizar_orientacion", "enderezar", "estimar_inclinacion",
    "recortar_zoom", "listar_recortes", "descargar_recortes",
    "MARGEN_DEFAULT", "ZOOM_DEFAULT", "BORDE_DEFAULT", "RECORTES_DIR",
    # ocr.py
    "get_ocr", "ocr_tabla", "texto_con_score", "cargar_ocr_json",
    "UNCLIP_DEFAULT",
    # filas.py
    "agrupar_filas", "filas_a_dataframe", "imprimir_filas",
    "inclinacion_de_cajas", "paso_filas",
    # diagnostico.py
    "previsualizar_bbox", "cajas_en_borde", "comparar_variantes",
    "detectar_rotacion",
]


# --------------------------------------------------------------------------
# Paso 1 completo — imagen → recorte de la tabla
# --------------------------------------------------------------------------

def preparar_tabla(
    ruta: str | Path,
    margen: float | tuple[float, float, float, float] = MARGEN_DEFAULT,
    zoom: int = ZOOM_DEFAULT,
    borde: int = BORDE_DEFAULT,
    rotacion: int = 0,
    enderezado: bool = True,
    guardar_en: str | Path | None = RECORTES_DIR,
    verbose: bool = True,
) -> Path:
    """Paso 1 completo: imagen → orientación → llm → recorte enderezado.

    El recorte queda en `guardar_en` (por defecto `recortes/`), listo para
    revisarlo o bajarlo con `descargar_recortes()`.

    Si el LLM no encuentra la tabla, devuelve la imagen original para que el
    pipeline siga funcionando igual. En ese caso tampoco se endereza: sin
    recorte, el estimador mira la hoja entera y ahí se equivoca (ver la sección
    "Paso 1.5" de `imagen.py`). Queda para `agrupar_filas`, que estima la
    inclinación residual sobre las coordenadas.
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
    return recortar_zoom(ruta, bbox, margen=margen, zoom=zoom, borde=borde,
                         enderezado=enderezado, guardar_en=guardar_en,
                         verbose=verbose)


# --------------------------------------------------------------------------
# Paso 5 — filas → items nombrados
# --------------------------------------------------------------------------
# Fachada. La lógica está en `esquema.py` (dónde empieza cada columna: geometría
# descubierta de los datos) y en `encabezado.py` (cómo se llama cada columna:
# vocabulario, sin coordenadas). Acá sólo se re-exporta, para que el notebook
# siga tocando un módulo solo.

def estandarizar(filas: list[list[dict]], usar_llm: bool = False, **kw):
    """Filas → (doc, avisos). El doc trae los items ya nombrados.

    OJO con el default: `esquema.estandarizar` viene con `usar_llm=True` y acá
    se invierte a propósito. Este es el camino del notebook y tiene que ser
    offline y determinista; el LLM de encabezado (una llamada de red, sobre los
    títulos y nada más) se pide explícitamente con `usar_llm=True`, y en los
    cinco formatos que ya conocemos no haría falta nunca.

    `avisos` no es decorativo: ahí está la grilla que se usó, las líneas que se
    descartaron por no parecer items y las celdas con dos valores. Se lee cómodo
    con `imprimir_avisos`.
    """
    return esquema.estandarizar(filas, usar_llm=usar_llm, **kw)


def items_a_dataframe(doc: dict) -> pd.DataFrame:
    """Los items del doc como tabla, con las 9 columnas canónicas siempre.

    Las columnas vacías se dejan a la vista: que `codigo_barras` esté vacío en
    factura2 es un dato del formato, y esconderlo hace que dos facturas
    devuelvan tablas de forma distinta.
    """
    return esquema.items_a_dataframe(doc)


def guardar_doc(doc: dict, ruta: str | Path) -> Path:
    """El doc del paso 5 a un .json, para mirarlo fuera de la sesión.

    `banco.py` lo escribe solo para las cinco facturas del corpus; esto es para
    cuando corrés una imagen suelta desde el notebook.
    """
    return esquema.guardar_doc(doc, ruta)


def validar_aritmetica(doc: dict, tol_rel: float = 0.01) -> pd.DataFrame:
    """`cantidad × precio ≈ importe`, línea por línea.

    Es la única validación que no necesita ground truth, y la red que atrapa
    una columna bien mapeada con el número equivocado.
    """
    return esquema.validar_aritmetica(doc, tol_rel=tol_rel)


def imprimir_avisos(avisos: dict, max_lineas: int = 8) -> None:
    """El diagnóstico del paso 5 en una pantalla.

    Lo que hay que mirar, en orden: si la grilla salió de los datos o del
    encabezado (del encabezado es peor y significa que había poco de donde
    medir), qué columna quedó sin nombre, y sobre todo `descartadas` — si ahí
    cayó un item de verdad, la tabla de arriba tiene una fila de menos y nada
    más lo grita.
    """
    if avisos.get("error"):
        print("error:", avisos["error"])
        return

    print("encabezado:", avisos["encabezado"])
    print("grilla:", avisos.get("origen_grilla"), "| nombres:",
          avisos.get("origen_mapa"))
    for col, campo in avisos["mapa"].items():
        print(f"   {col:>22}  {campo if campo else '— sin título'}")

    faltan = sorted(set(esquema.CAMPOS_ESENCIALES) - set(avisos["esenciales"]))
    if faltan:
        print("esenciales que ninguna columna se llevó:", faltan)
    for t in avisos["titulos_partidos"]:
        print(f"título partido: {t['texto']!r} -> {t['en']}")
    if avisos["titulos_sin_datos"]:
        print("títulos sin datos abajo (columna vacía):",
              avisos["titulos_sin_datos"])
    if avisos["titulos_sin_columna"]:
        print("títulos que no se quedaron con ninguna columna:",
              avisos["titulos_sin_columna"])

    if avisos["conflictos"]:
        print(f"\nceldas con dos valores ({len(avisos['conflictos'])}) — "
              f"síntoma de filas fusionadas:")
        for c in avisos["conflictos"][:max_lineas]:
            print("   fila", c["fila"], "->", "; ".join(c["detalle"]))

    if avisos["descartadas"]:
        print(f"\nlíneas descartadas por no parecer items "
              f"({len(avisos['descartadas'])}):")
        for d in avisos["descartadas"][:max_lineas]:
            print("   ", d[:140])


# --------------------------------------------------------------------------
# Conveniencia
# --------------------------------------------------------------------------

def procesar(ruta: str | Path, margen=MARGEN_DEFAULT, zoom: int = ZOOM_DEFAULT,
             borde: int = BORDE_DEFAULT, rotacion: int = 0,
             enderezado: bool = True, guardar_en: str | Path | None = RECORTES_DIR,
             verbose: bool = True, **kw_ocr):
    """Corre los pasos 1 a 4 y devuelve (img_tabla, res, filas, df).

    Llega hasta las filas crudas: el `df` es una columna por posición, todavía
    sin nombre. El paso 5 va aparte, `estandarizar(filas)`, y no se agrega acá
    como quinto valor de retorno para no romper el desempaquetado de cuatro.

    `kw_ocr` va a `get_ocr` (unwarping, unclip, lang), para poder correr una
    variante completa sin tener que rearmar los pasos a mano.
    """
    img_tabla = preparar_tabla(ruta, margen=margen, zoom=zoom, borde=borde,
                               rotacion=rotacion, enderezado=enderezado,
                               guardar_en=guardar_en, verbose=verbose)
    res = ocr_tabla(img_tabla, **kw_ocr)
    filas = agrupar_filas(res, verbose=verbose)
    return img_tabla, res, filas, filas_a_dataframe(filas)
