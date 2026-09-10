"""esquema.py — formato canónico de una factura y traducción de los ground truth.

El problema que resuelve
------------------------
Los seis .json de `imagenes/` se cargaron a mano, uno por uno, y cada uno usa
sus propios nombres y su propia jerarquía para las mismas cosas:

    factura1  comprobante.numero = "0012-00004186"   items[].producto
    factura2  documento.numero   = "00077735"        items[].descripcion
    factura3  numero             = "00432864"        detalle[]
    factura4  comprobante.numero = "00024579"        items[].descripcion
    factura5  comprobante.numero = "0005-00011132"   items[].descripcion
    factura6  factura.numero     = "00386066"        items[].descripcion

Así no se puede medir nada: cualquier comparación campo a campo estaría
midiendo la variación de quien cargó el JSON, no la del pipeline. Este módulo
define UN esquema canónico y traduce a él las dos puntas — el ground truth acá,
y más adelante la salida del pipeline — para que la comparación sea posible.

Dos decisiones de diseño que conviene tener presentes:

1. La traducción es explícita, por tabla de alias. No se adivina por parecido.
   Si un campo del ground truth no tiene alias, NO se pierde en silencio:
   aparece en `avisos["sin_mapear"]`. Un normalizador que descarta callado es
   peor que no tener normalizador.

2. El esquema guarda los valores lo más cerca posible de la fuente (sólo
   normaliza números, fechas y CUIT). Comparar con tolerancia — mayúsculas,
   acentos, espacios — es trabajo de la capa de evaluación, no de esta.

Uso:

    from esquema import desde_ground_truth, informe_normalizacion

    doc, avisos = desde_ground_truth("imagenes/factura2.json")
    print(informe_normalizacion("imagenes"))
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# El vocabulario de títulos de columna vive en su propio módulo: sabe de
# nombres, no de geometría. Acá se lo usa para ponerle nombre a las columnas
# que la grilla descubre a partir de los datos.
from encabezado import (TIER_EXACTO, TIER_PREFIJO, elegir_encabezado,
                        mapear_encabezado, partir_titulo, puntaje_columna)
from texto import clave, sin_acentos

# --------------------------------------------------------------------------
# El esquema canónico
# --------------------------------------------------------------------------

ESQUEMA = {
    "comprobante": ["clase", "letra", "codigo", "punto_venta", "numero",
                    "fecha", "hora", "hoja"],
    "emisor": ["razon_social", "nombre_fantasia", "domicilio", "localidad",
               "cuit", "iibb", "condicion_iva", "inicio_actividades",
               "telefono", "email", "fax", "web", "codigo_postal"],
    "cliente": ["nombre", "domicilio", "localidad", "cuit", "codigo",
                "condicion_iva", "condicion_venta"],
    "totales": ["neto", "descuento", "subtotal", "iva", "iva_10_5", "iva_pct",
                "percepciones", "total"],
    "cae": ["numero", "vencimiento"],
    "otros": ["vendedor", "codigo_barras", "observaciones", "total_en_letras"],
}

CAMPOS_ITEM = ["codigo", "descripcion", "cantidad", "unidad", "precio_unitario",
               "descuento_pct", "importe", "codigo_barras", "codigo_proveedor"]

# Campos que son numéricos: se pasan por `_num` al normalizar.
_NUMERICOS = {
    ("totales", "neto"), ("totales", "descuento"), ("totales", "subtotal"),
    ("totales", "iva"), ("totales", "iva_10_5"), ("totales", "iva_pct"),
    ("totales", "percepciones"), ("totales", "total"),
}
_ITEM_NUMERICOS = {"cantidad", "precio_unitario", "descuento_pct", "importe"}


def documento_vacio() -> dict:
    """Estructura canónica con todos los campos en None. Es el molde."""
    doc = {sec: {campo: None for campo in campos} for sec, campos in ESQUEMA.items()}
    doc["items"] = []
    return doc


# --------------------------------------------------------------------------
# Normalizadores de valores
# --------------------------------------------------------------------------

def _txt(v):
    """Texto limpio, o None si está vacío. No toca mayúsculas ni acentos."""
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    v = re.sub(r"\s+", " ", v).strip()
    return v or None


def _miles(sep: str) -> re.Pattern:
    return re.compile(r"^-?\d{1,3}(" + re.escape(sep) + r"\d{3})+$")


_MILES_PUNTO = _miles(".")
_MILES_COMA = _miles(",")


def _num(v):
    """Número → float, sin asumir cuál separador es el decimal.

    La primera versión asumía que si había coma, la coma era el decimal
    (formato argentino). Se cayó con el corpus real: la factura6 imprime
    `5.829,97` y PaddleOCR lo transcribe como `5,829.97`, con los dos
    separadores dados vuelta. Con la regla vieja eso daba 5.82997 — un número
    perfectamente válido, mil veces más chico, que no rompe nada y no se nota
    hasta que lo comparás contra el ground truth.

    Regla que sí funciona: cuando aparecen los dos separadores, el ÚLTIMO es el
    decimal y el otro es de miles, sin importar cuál sea cuál. Cuando aparece
    uno solo, es de miles únicamente si separa grupos exactos de tres dígitos
    (`1.234` → 1234); si no, es decimal (`450,00` → 450.0).

    El mismo criterio aplica cuando el separador repetido es el MISMO
    carácter: `17,311,55` (factura1, debería ser `17.311,55`) o `3.742.45`
    (debería ser `3.742,45`) son casos donde el OCR imprime el separador de
    miles y el decimal con el mismo glifo. El último es igual el decimal;
    exigir grupos de tres dígitos ahí adentro (como hace el chequeo de
    "es de miles a secas") descarta el número entero en vez de rescatarlo.

    Lección para la tesis: el formato numérico que sale del OCR no es el que
    está impreso en el papel, y un parser que confía en la convención local
    produce errores silenciosos de tres órdenes de magnitud.
    """
    if v is None or v == "" or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip().replace("%", "").replace("$", "").replace(" ", "")
    if not s:
        return None

    tiene_punto, tiene_coma = "." in s, "," in s
    if tiene_punto and tiene_coma:
        decimal = "." if s.rfind(".") > s.rfind(",") else ","
        miles = "," if decimal == "." else "."
        s = s.replace(miles, "").replace(decimal, ".")
    elif tiene_coma and s.count(",") > 1:
        cabeza, _, cola = s.rpartition(",")
        s = cabeza.replace(",", "") + "." + cola
    elif tiene_coma:
        s = s.replace(",", "") if _MILES_COMA.match(s) else s.replace(",", ".")
    elif tiene_punto and s.count(".") > 1:
        cabeza, _, cola = s.rpartition(".")
        s = cabeza.replace(".", "") + "." + cola
    elif tiene_punto and _MILES_PUNTO.match(s):
        s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return None


_RE_FECHA = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")


def _fecha(v):
    """Fecha → DD/MM/AAAA. Deja pasar cualquier cosa que no reconozca."""
    s = _txt(v)
    if not s:
        return None
    m = _RE_FECHA.match(s)
    if not m:
        return s                      # "02/2012", "11/11/96" parcial, etc.
    d, mes, a = m.groups()
    return f"{int(d):02d}/{int(mes):02d}/{a}"


def _cuit(v):
    """CUIT → 99-99999999-9. Unifica '20206014645' con '20-20601464-5'."""
    s = _txt(v)
    if not s:
        return None
    d = re.sub(r"\D", "", s)
    return f"{d[:2]}-{d[2:10]}-{d[10]}" if len(d) == 11 else s


# `sin_acentos` y `clave` viven en texto.py (importados arriba).


# --------------------------------------------------------------------------
# Tablas de alias — la traducción, explícita
# --------------------------------------------------------------------------
# Orden importa: el primer alias presente gana. Ver el caso `importe` abajo.

ALIAS = {
    "comprobante": {
        "clase": ["tipo_comprobante", "tipo_documento", "descripcion"],
        "letra": ["letra", "tipo"],
        "codigo": ["codigo", "cod_tipo", "codigo_tipo"],
        "punto_venta": ["punto_venta"],
        "numero": ["numero"],
        "fecha": ["fecha", "fecha_emision"],
        "hora": ["hora"],
        "hoja": ["hoja", "pagina"],
    },
    "emisor": {
        "razon_social": ["razon_social", "nombre", "empresa"],
        "nombre_fantasia": ["nombre_fantasia", "nombre_comercial", "titular"],
        "domicilio": ["domicilio", "direccion"],
        "localidad": ["localidad"],
        "cuit": ["cuit"],
        "iibb": ["iibb", "ingresos_brutos", "ingresos_brutos_cm", "iib_cm"],
        "condicion_iva": ["condicion_iva", "iva"],
        "inicio_actividades": ["inicio_actividades", "fecha_inicio_actividades"],
        "telefono": ["telefono", "telefono_fax"],
        "email": ["email"],
        "fax": ["fax"],
        "web": ["web", "sitio_web"],
        "codigo_postal": ["codigo_postal"],
    },
    "cliente": {
        "nombre": ["nombre", "razon_social"],
        "domicilio": ["domicilio", "domicilio_comercial", "direccion"],
        "localidad": ["localidad"],
        "cuit": ["cuit"],
        "codigo": ["codigo", "nro_cliente"],
        "condicion_iva": ["condicion_iva"],
        "condicion_venta": ["condicion_venta"],
    },
    "totales": {
        "neto": ["neto_gravado", "gravado", "neto"],
        "descuento": ["descuento"],
        "subtotal": ["subtotal", "sub_total"],
        "iva": ["iva_importe", "iva_monto", "iva", "iva_21"],
        "iva_10_5": ["iva_10_5"],
        "iva_pct": ["iva_porcentaje", "iva_pct"],
        "percepciones": ["percepcion_iibb", "percepcion_ingresos_brutos",
                         "percep_ing_brutos"],
        "total": ["total_a_pagar", "total"],
    },
    "cae": {
        "numero": ["numero", "cae"],
        "vencimiento": ["vencimiento", "fecha_vencimiento", "vencimiento_cae"],
    },
    "otros": {
        "vendedor": ["vendedor", "usuario"],
        "codigo_barras": ["codigo_barras"],
        "observaciones": ["observaciones", "condiciones", "leyenda", "notas"],
        "total_en_letras": ["total_en_letras"],
    },
}

ALIAS_ITEM = {
    "codigo": ["codigo", "articulo"],
    "descripcion": ["descripcion", "producto"],
    "cantidad": ["cantidad", "cant_remitente"],
    "unidad": ["unidad_medida", "unidad"],
    "precio_unitario": ["precio_unitario", "precio"],
    "descuento_pct": ["porcentaje_bonificacion", "porcentaje_descuento",
                      "descuento_porcentaje"],
    # OJO con el orden: en factura3 conviven `total` (900,00 — un cargo suelto)
    # y `total_final` (60.358,31 — el importe de la línea). En factura5 el
    # importe de la línea se llama `total` a secas. Poniendo `total_final`
    # antes que `total`, los dos casos caen bien.
    "importe": ["importe", "neto_total", "total_final", "total"],
    "codigo_barras": ["codigo_barras", "cod_barras"],
    "codigo_proveedor": ["cod_proveedor", "codigo_proveedor"],
}

# Secciones del ground truth: dónde buscar cada bloque canónico.
SECCIONES = {
    "comprobante": ["comprobante", "factura", "documento"],
    "emisor": ["emisor", "empresa"],
    "cliente": ["cliente", "receptor", "destinatario"],
    "totales": ["totales", "pie", "total"],
    "cae": ["cae"],
    "otros": [],          # no tiene contenedor propio: se busca por todo el doc
}
CLAVES_ITEMS = ["items", "detalle", "lineas", "productos"]


# --------------------------------------------------------------------------
# Traducción
# --------------------------------------------------------------------------

class _Fuente:
    """Envuelve el JSON crudo y anota qué claves se consumieron.

    Sin esto no hay forma de saber qué quedó afuera, y un campo que desaparece
    sin ruido es exactamente el tipo de error que después no se encuentra.
    """

    def __init__(self, raw: dict):
        self.raw = raw
        self.consumidas: set[str] = set()

    def buscar(self, d: dict, prefijo: str, alias: list[str]):
        """Primer alias presente en `d`. Devuelve (valor, ruta) o (None, None)."""
        indice = {clave(k): k for k in d}
        for a in alias:
            k = indice.get(clave(a))
            if k is None:
                continue
            ruta = f"{prefijo}.{k}" if prefijo else str(k)
            # Un campo presente pero vacío está reconocido igual: se marca como
            # consumido para que no aparezca como "sin mapear". El ground truth
            # dice que ahí no hay dato, no que el esquema no lo contempla.
            self.consumidas.add(ruta)
            if d[k] not in ("", [], {}, None):
                return d[k], ruta
        return None, None

    def hojas(self) -> set[str]:
        """Todas las rutas hoja del JSON, con las listas colapsadas a `[]`."""
        rutas: set[str] = set()

        def caminar(obj, pref=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    caminar(v, f"{pref}.{k}" if pref else str(k))
            elif isinstance(obj, list):
                for v in obj:
                    caminar(v, f"{pref}[]")
            else:
                rutas.add(pref)

        caminar(self.raw)
        return rutas


def _ubicar(fuente: _Fuente, alias_seccion: list[str]) -> tuple[str, dict]:
    """Devuelve (prefijo, dict) de la sección, o ("", raíz) si viene aplanada.

    factura3 no anida el comprobante: tipo_documento, numero y fecha_emision
    cuelgan de la raíz. Devolver la raíz como fallback es seguro porque el
    mapeo posterior es por alias: sólo levanta las claves que reconoce.
    """
    indice = {clave(k): k for k in fuente.raw}
    for a in alias_seccion:
        k = indice.get(clave(a))
        if isinstance(fuente.raw.get(k), dict):
            return k, fuente.raw[k]
    return "", fuente.raw


def _normalizar_valor(seccion: str, campo: str, valor):
    if (seccion, campo) in _NUMERICOS:
        return _num(valor)
    if campo == "cuit":
        return _cuit(valor)
    if campo in ("fecha", "vencimiento", "inicio_actividades"):
        return _fecha(valor)
    return _txt(valor)


_RE_LETRA = re.compile(r"^[A-Ma-m]$")


def _partir_numero(comp: dict) -> None:
    """'0012-00004186' → punto_venta '0012' + numero '00004186'.

    Facturas 1 y 5 traen el punto de venta pegado al número; las otras los
    traen separados. Sin esto, comparar `numero` entre formatos da siempre
    distinto aunque la lectura sea perfecta.
    """
    num = comp.get("numero")
    if not num or "-" not in num:
        return
    izq, _, der = num.rpartition("-")
    if izq.isdigit() and der.isdigit():
        comp["punto_venta"] = comp.get("punto_venta") or izq
        comp["numero"] = der


def _acomodar_tipo(comp: dict) -> None:
    """Separa la letra de la clase cuando vienen juntas o cruzadas.

    factura2: tipo_comprobante='FACTURA' + letra='A'   → clase / letra
    factura4: tipo_comprobante='FACTURA A'             → hay que partirlo
    factura5: tipo='A' + descripcion='FACTURA - CTA CTE'
    """
    clase, letra = comp.get("clase"), comp.get("letra")

    if letra and not _RE_LETRA.match(letra):
        # 'tipo' trajo algo que no es una letra: en realidad es la clase
        clase, letra = clase or letra, None

    if clase and not letra:
        m = re.match(r"^(.*?)\s+([A-M])$", clase)
        if m:
            clase, letra = m.group(1), m.group(2)

    comp["clase"] = _txt(clase)
    comp["letra"] = letra.upper() if letra else None


def _item(fuente: _Fuente, crudo: dict, prefijo: str) -> dict:
    it = {c: None for c in CAMPOS_ITEM}
    if not isinstance(crudo, dict):
        return it
    for campo, alias in ALIAS_ITEM.items():
        valor, _ = fuente.buscar(crudo, prefijo, alias)
        it[campo] = _num(valor) if campo in _ITEM_NUMERICOS else _txt(valor)
    return it


def desde_ground_truth(ruta: str | Path) -> tuple[dict, dict]:
    """JSON cargado a mano → documento canónico + avisos.

    `avisos` trae:
      sin_mapear  — rutas del JSON que ninguna tabla de alias reconoció
      vacios      — campos canónicos que quedaron en None
      items       — cantidad de líneas leídas
    """
    ruta = Path(ruta)
    raw = json.loads(ruta.read_text(encoding="utf-8"))
    fuente = _Fuente(raw)
    doc = documento_vacio()

    for seccion, campos in ALIAS.items():
        prefijo, origen = _ubicar(fuente, SECCIONES[seccion])
        for campo, alias in campos.items():
            valor, _ = fuente.buscar(origen, prefijo, alias)

            # factura4 anida `comprobante` pero deja `tipo_comprobante` y
            # `cod_tipo` colgando de la raíz. Sin este respaldo, la clase y el
            # código del comprobante se pierden. Se limita a `comprobante` y
            # `otros` a propósito: hacerlo en `emisor` o `cliente` invitaría a
            # que un `cuit` suelto de la raíz se cuele en la sección equivocada.
            if valor is None and seccion in ("comprobante", "otros"):
                valor, _ = fuente.buscar(fuente.raw, "", alias)

            # `otros` no tiene contenedor: vendedor vive en la raíz (factura5),
            # en `pie` (factura2) o en `cliente` (factura6).
            if valor is None and seccion == "otros":
                for k, v in fuente.raw.items():
                    if isinstance(v, dict):
                        valor, _ = fuente.buscar(v, str(k), alias)
                        if valor is not None:
                            break

            doc[seccion][campo] = _normalizar_valor(seccion, campo, valor)

    # el CAE aparece como dict {numero, vencimiento} o como string suelto
    cae_crudo = raw.get("cae")
    if isinstance(cae_crudo, str):
        doc["cae"]["numero"] = _txt(cae_crudo)
        fuente.consumidas.add("cae")
    if doc["cae"]["vencimiento"] is None:
        v, _ = fuente.buscar(raw, "", ["vencimiento_cae"])
        doc["cae"]["vencimiento"] = _fecha(v)
    # factura4 esconde el CAE adentro del bloque comprobante
    if doc["cae"]["numero"] is None:
        pref, comp = _ubicar(fuente, SECCIONES["comprobante"])
        v, _ = fuente.buscar(comp, pref, ["cae"])
        doc["cae"]["numero"] = _txt(v)

    _partir_numero(doc["comprobante"])
    _acomodar_tipo(doc["comprobante"])

    lista, ruta_items = fuente.buscar(raw, "", CLAVES_ITEMS)
    if isinstance(lista, list):
        doc["items"] = [_item(fuente, c, f"{ruta_items}[]") for c in lista]
        # Sacar el contenedor de la lista de consumidas: si queda, el prefijo
        # `items[]` tapa TODOS los campos de item no reconocidos y el informe
        # de "sin mapear" da cero falsamente (así se escondía `despacho` en
        # factura5). Se marcan como consumidos los campos, no el contenedor.
        fuente.consumidas.discard(ruta_items)

    sin_mapear = sorted(r for r in fuente.hojas()
                        if not any(r == c or r.startswith(c + ".") or
                                   r.startswith(c + "[]")
                                   for c in fuente.consumidas))
    vacios = [f"{s}.{c}" for s, campos in ESQUEMA.items()
              for c in campos if doc[s][c] is None]

    return doc, {"archivo": ruta.name, "items": len(doc["items"]),
                 "sin_mapear": sin_mapear, "vacios": vacios}


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

# factura3 queda fuera del banco de items: es un comprobante de transporte
# (remitente, remito de origen, aforo, flete), no una tabla de productos. Sus
# líneas no tienen código ni descripción de artículo, así que medir precisión
# de items sobre ella castiga al pipeline por una tarea que no es la suya.
# El archivo se conserva a propósito: sigue siendo el mejor ejemplo de que
# ninguna heurística puede asumir la forma de la tabla.
EXCLUIDAS_ITEMS = {"factura3"}


def normalizar_carpeta(carpeta: str | Path = "imagenes",
                       excluir: set[str] | None = EXCLUIDAS_ITEMS
                       ) -> dict[str, tuple[dict, dict]]:
    """Normaliza los .json no vacíos de la carpeta.

    `excluir=None` incluye todo, para cuando querés mirar el corpus completo
    y no el banco de pruebas de items.
    """
    carpeta = Path(carpeta)
    excluir = excluir or set()
    salida = {}
    for j in sorted(carpeta.glob("*.json")):
        if j.stat().st_size == 0 or j.stem in excluir:
            continue
        salida[j.stem] = desde_ground_truth(j)
    return salida


def informe_normalizacion(carpeta: str | Path = "imagenes",
                          excluir: set[str] | None = EXCLUIDAS_ITEMS) -> pd.DataFrame:
    """Una fila por factura: qué se pudo mapear y qué quedó afuera."""
    filas = []
    for nombre, (doc, av) in normalizar_carpeta(carpeta, excluir).items():
        completos = sum(1 for s, cs in ESQUEMA.items() for c in cs
                        if doc[s][c] is not None)
        totales = sum(len(cs) for cs in ESQUEMA.values())
        item_ok = {c: sum(1 for it in doc["items"] if it[c] is not None)
                   for c in ("codigo", "descripcion", "cantidad",
                             "precio_unitario", "importe")}
        filas.append({
            "factura": nombre,
            "cabecera": f"{completos}/{totales}",
            "items": av["items"],
            **{f"it_{k}": v for k, v in item_ok.items()},
            "sin_mapear": len(av["sin_mapear"]),
        })
    return pd.DataFrame(filas)


def items_a_dataframe(doc: dict) -> pd.DataFrame:
    """Los items canónicos como tabla — la forma en que se van a comparar."""
    return pd.DataFrame(doc["items"], columns=CAMPOS_ITEM)


# --------------------------------------------------------------------------
# Estandarización: filas del pipeline → items canónicos
# --------------------------------------------------------------------------
# El pipeline entrega columnas por posición, sin nombre, y cada formato usa
# nombres distintos para lo mismo: código es ARTICULO en factura2, Codigo en
# factura1 y Código (tercera columna) en factura5.
#
# Son dos preguntas distintas y se resuelven en dos lugares distintos:
#
#   CÓMO SE LLAMA cada columna  → encabezado.py (vocabulario, sin coordenadas)
#   DÓNDE EMPIEZA cada columna  → acá abajo (geometría, sin vocabulario)
#
# La grilla se descubre de los datos; el encabezado sólo la nombra.

CAMPOS_ESENCIALES = ("codigo", "descripcion", "cantidad", "precio_unitario",
                     "importe")


def _x0(c: dict) -> float:
    return float(c.get("x0", c["x"]))


def _x1(c: dict) -> float:
    return float(c.get("x1", c.get("x0", c["x"])))


def _limites_columnas(fila: list[dict]) -> list[tuple[float, float]]:
    """Franja horizontal de cada columna, a partir de dónde ARRANCA su encabezado.

    Primera versión de esto: el límite entre dos columnas era el punto medio
    entre el fin de un encabezado y el arranque del siguiente. Suena razonable
    y está mal, porque el encabezado es una palabra corta y el dato debajo es
    ancho: 'DESCRIPCION' ocupa 220px pero la descripción del producto ocupa
    500px, y el pedazo que sobra cae del otro lado del punto medio y aterriza
    en la columna de precio. Es justo el caso de una descripción que el OCR
    parte en dos cajas.

    Lo que sí es estable es dónde arranca cada columna: una tabla se arma de
    izquierda a derecha y ninguna columna invade el arranque de la siguiente.
    Así que la franja de la columna i va desde el arranque de su encabezado
    hasta el arranque del siguiente, y la primera y la última se extienden
    hasta el infinito para no perder nada por los costados.
    """
    orden = sorted(fila, key=_x0)
    arranques = [_x0(c) for c in orden]
    franjas = []
    for i, a in enumerate(arranques):
        izq = float("-inf") if i == 0 else a
        der = arranques[i + 1] if i + 1 < len(arranques) else float("inf")
        franjas.append((izq, der))
    return franjas


def _bandas_desde_datos(filas: list[list[dict]], min_hueco: float = 3.0,
                        frac_umbral: float = 0.15) -> list[tuple[float, float]]:
    """Descubre las columnas mirando dónde NO hay texto en ninguna fila.

    Por qué no alcanza con los encabezados: en factura5 el título 'Descripción'
    está centrado sobre su columna y arranca 190px a la derecha de donde
    arrancan las descripciones, así que usar su posición como límite mete la
    descripción entera en la columna de código. Y en factura4 hay una columna
    de código de barras que directamente no tiene encabezado: sin descubrirla
    desde los datos, su contenido se pega a la descripción.

    El método: se proyectan todas las cajas sobre el eje x contando EN CUÁNTAS
    FILAS está ocupado cada píxel. Una columna es una franja ocupada en muchas
    filas; un separador es un hueco ocupado en casi ninguna. Contar por filas y
    no por cajas es lo que hace que una línea rara no arruine la grilla: la
    línea 'DESPACHO: S/FACT102141' de factura5 cruza el hueco entre código y
    descripción, pero lo cruza en una sola fila de cinco.

    Con una o dos filas de datos no hay estadística posible y el umbral baja a
    cero: cualquier hueco cuenta. Es lo correcto para una factura de un ítem.
    """
    cajas = [c for f in filas for c in f]
    if not cajas:
        return []

    x_ini = int(min(_x0(c) for c in cajas))
    x_fin = int(max(_x1(c) for c in cajas)) + 1
    ancho = x_fin - x_ini + 1
    if ancho <= 1:
        return []

    n = len(filas)
    cobertura = np.zeros(ancho, dtype=int)
    for fila in filas:
        marca = np.zeros(ancho, dtype=bool)
        for c in fila:
            marca[int(_x0(c)) - x_ini: int(_x1(c)) - x_ini + 1] = True
        cobertura += marca

    umbral = 0 if n <= 2 else max(1, int(frac_umbral * n))
    ocupado = cobertura > umbral

    bandas: list[list[float]] = []
    dentro = False
    for i, v in enumerate(ocupado):
        if v and not dentro:
            bandas.append([x_ini + i, x_ini + i])
            dentro = True
        elif v:
            bandas[-1][1] = x_ini + i
        else:
            dentro = False

    # huecos demasiado finos no son separadores de columna, son el espacio
    # entre dos palabras que el OCR separó en cajas distintas
    unidas: list[list[float]] = []
    for b in bandas:
        if unidas and b[0] - unidas[-1][1] < min_hueco:
            unidas[-1][1] = b[1]
        else:
            unidas.append(list(b))

    # Las bandas quedan con sus extremos reales, sin extender a ±infinito: una
    # caja que cae afuera de todas igual aterriza en la más cercana (el solape
    # negativo más chico), y así "afuera de todas" sigue siendo distinguible,
    # que es lo que permite descartar el título de una columna vacía.
    return [(a, b) for a, b in unidas]


def _bandas_tocadas(caja: dict, bandas: list[tuple[float, float]],
                    min_frac: float = 0.15) -> list[int]:
    """Bandas sobre las que la caja se apoya de verdad.

    Con "solape positivo" a secas no alcanza: los títulos sobresalen unos
    píxeles sobre la columna vecina y eso basta para que parezcan abarcarla.
    'Cód. Barras Cód. Prov' de factura1 mide 305px, pisa 21 de la columna de
    código, y con el criterio laxo terminaba adueñándose de ella y dejando el
    código sin leer en las 12 filas. Se exige que el solape sea al menos una
    fracción del ancho del propio título.
    """
    a, b = _x0(caja), _x1(caja)
    minimo = max(1.0, min_frac * (b - a))
    return [k for k, (izq, der) in enumerate(bandas)
            if min(b, der) - max(a, izq) >= minimo]


def _subdividir_bandas(bandas: list[tuple[float, float]],
                       filas: list[list[dict]], min_frac: float = 0.7,
                       max_sd: float = 0.05) -> list[tuple[float, float]]:
    """Parte una banda que en realidad son dos columnas pegadas.

    La proyección de blancos no separa dos columnas cuyas cajas se tocan. En
    factura1 pasa dos veces: el código de barras arranca donde termina el
    código de artículo, y 'Neto Unit.' con 'Neto Total' se rozan en algunas
    filas. La banda sale una sola y adentro viven dos columnas.

    Lo que sí delata la estructura es la repetición: si casi todas las filas
    ponen exactamente k cajas dentro de la banda, y el hueco entre ellas cae
    siempre en el mismo lugar, entonces son k columnas. La exigencia de que el
    corte sea CONSISTENTE (desvío chico frente al ancho de la banda) es la que
    evita partir una descripción que el OCR corta en dos pedazos cada vez en un
    lugar distinto.

    El "casi todas" se mide contra las filas que traen evidencia (2+ cajas
    dentro de la banda), no contra todas las que caen dentro de la banda. Una
    fila donde el OCR pegó las dos columnas en una sola caja no vota en contra
    del corte: no vota nada, porque no hay corte que medir ahí. Contarla en el
    denominador castiga al corte por un error de OCR ajeno a si la columna
    existe. En factura1 esto hundía el corte código/descripción por 0.4 filas
    (8 de 12) a pesar de que las 8 filas que sí podían opinar coincidían con
    2.5px de desvío contra un margen de 27.85px.
    """
    salida: list[tuple[float, float]] = []
    for izq, der in bandas:
        ancho = der - izq
        conteos, cortes = [], []
        for f in filas:
            dentro = sorted((c for c in f
                             if min(_x1(c), der) - max(_x0(c), izq) > 0), key=_x0)
            if dentro:
                conteos.append(len(dentro))
            if len(dentro) >= 2:
                cortes.append([(_x1(dentro[i]) + _x0(dentro[i + 1])) / 2
                               for i in range(len(dentro) - 1)])

        if not conteos:
            salida.append((izq, der))
            continue
        k = max(set(conteos), key=conteos.count)
        utiles = [c for c in cortes if len(c) == k - 1]
        if k < 2 or len(utiles) < min_frac * len(cortes):
            salida.append((izq, der))
            continue

        arr = np.array(utiles, dtype=float)
        if (arr.std(axis=0) > max_sd * ancho).any():
            salida.append((izq, der))       # el corte no cae siempre igual
            continue

        puntos = [izq, *arr.mean(axis=0).tolist(), der]
        salida.extend((puntos[i], puntos[i + 1]) for i in range(len(puntos) - 1))
    return salida


def _columna_de(caja: dict, franjas: list[tuple[float, float]]) -> int:
    """Columna con la que la caja se superpone más.

    Por superposición y no por el centro ni por el borde izquierdo: una columna
    numérica alineada a la derecha puede tener un número que arranca antes que
    su propio encabezado (el número es más largo que el título), y un texto
    largo puede terminar bien adentro de la columna siguiente. La superposición
    acierta en los dos casos; el centro y el borde fallan cada uno en uno.
    """
    return _columna_y_solape(caja, franjas)[0]


def _columna_y_solape(caja: dict, franjas: list[tuple[float, float]]):
    """(columna, solape). Solape negativo = la caja cae en un hueco.

    El arranque en -inf importa: con `mejor_sup = -1` la función devolvía la
    columna 0 por defecto cada vez que TODOS los solapes eran negativos, que es
    justo lo que pasa con el título de una columna vacía en todas las filas
    ('% DTO' en factura2). El título terminaba pisando la columna de código.
    """
    a, b = _x0(caja), _x1(caja)
    mejor, mejor_sup = 0, float("-inf")
    for k, (izq, der) in enumerate(franjas):
        sup = min(b, der) - max(a, izq)
        if sup > mejor_sup:
            mejor, mejor_sup = k, sup
    return mejor, mejor_sup


# Campos donde una celda tiene que traer un solo valor. Si trae dos, no es que
# el OCR partió el texto: es que dos items quedaron pegados en la misma fila.
_CAMPOS_ATOMICOS = ("codigo", "cantidad", "precio_unitario", "importe")


def _fila_a_item(fila: list[dict], franjas: list[tuple[float, float]],
                 mapa: dict[int, str]) -> tuple[dict, list[str]]:
    celdas: dict[int, list[dict]] = {}
    for caja in fila:
        celdas.setdefault(_columna_de(caja, franjas), []).append(caja)

    it = {c: None for c in CAMPOS_ITEM}
    conflictos = []
    for k, cajas in celdas.items():
        campo = mapa.get(k)
        if campo not in CAMPOS_ITEM:
            continue                       # columna sin nombre, o `iva`
        cajas = sorted(cajas, key=_x0)
        # Varias cajas en la misma celda: en la descripción es el OCR que partió
        # el texto y se unen. En una columna atómica es el síntoma de dos filas
        # fusionadas, y conviene que grite en vez de elegir una y seguir.
        if len(cajas) > 1 and campo in _CAMPOS_ATOMICOS:
            conflictos.append(f"{campo}: {[c['txt'] for c in cajas]}")
        texto = " ".join(c["txt"] for c in cajas)
        it[campo] = _num_suelto(texto) if campo in _ITEM_NUMERICOS else _txt(texto)
    return it, conflictos


def _num_suelto(texto: str):
    """Como `_num`, pero rescata el número cuando viene con texto pegado.

    El OCR une celdas vecinas cada tanto: en factura5 la cantidad y la unidad
    salen como una sola caja, '5,0000 U'. Devolver None ahí es perder un dato
    que está entero y a la vista. Se prueba el texto completo y, si no parsea,
    el primer token que sí lo haga. 'DESPACHO: S/FACT102141' sigue dando None,
    que es lo correcto.
    """
    v = _num(texto)
    if v is not None or not texto:
        return v
    for token in str(texto).split():
        v = _num(token)
        if v is not None:
            return v
    return None


def _es_item(it: dict) -> bool:
    """Una línea es un item si tiene código, o cantidad e importe juntos."""
    if it["codigo"]:
        return True
    return it["cantidad"] is not None and it["importe"] is not None


def _nombrar_bandas(encabezado: list[dict], bandas: list[tuple[float, float]],
                    avisos: dict) -> dict[int, str]:
    """Pone nombre canónico a cada columna descubierta en los datos.

    Un título puede apoyarse sobre varias bandas (el OCR pegó dos o tres
    títulos en una caja): ahí se lo desarma y cada pedazo va a su banda, de
    izquierda a derecha. Un título que no se apoya sobre ninguna banda es el de
    una columna vacía en todas las filas y se descarta, en vez de empujarlo a
    la columna de al lado.
    """
    candidatos: list[tuple[float, int, int, str]] = []
    for i, celda in enumerate(encabezado):
        tocadas = _bandas_tocadas(celda, bandas)
        if len(tocadas) > 1:
            partes = partir_titulo(celda["txt"])
            if len(partes) > 1:
                # Se proponen todas las partes para todas las bandas que toca,
                # con preferencia por el orden pero sin imponerlo: el título
                # suele sobresalir un poco sobre la columna vecina, así que la
                # cantidad de partes y la de bandas no siempre coincide, y una
                # asignación posicional rígida se corre entera.
                escala = ((len(partes) - 1) / (len(tocadas) - 1)
                          if len(tocadas) > 1 else 0)
                for ib, b in enumerate(tocadas):
                    for ip, campo in enumerate(partes):
                        castigo = 0.1 * abs(ip - ib * escala)
                        candidatos.append((TIER_EXACTO - castigo, i, b, campo))
                avisos.setdefault("titulos_partidos", []).append(
                    {"texto": celda["txt"], "en": partes})
                continue
        b, solape = _columna_y_solape(celda, bandas)
        if solape <= 0:
            avisos["titulos_sin_datos"].append(celda["txt"])
            continue
        for campo, p in puntaje_columna(celda["txt"]).items():
            candidatos.append((p, i, b, campo))

    # Primero los matches más fuertes; a igual fuerza gana el título de más a
    # la izquierda. En factura1 conviven 'P. Unit.' (precio de lista) y
    # 'Neto Unit' (después de la bonificación), los dos empatan en puntaje, y
    # el ground truth guarda el de lista: el de la izquierda.
    candidatos.sort(key=lambda t: (-t[0], t[1]))
    mapa: dict[int, str] = {}
    usados: set[str] = set()
    ganadores: set[int] = set()
    for _, i, b, campo in candidatos:
        if b in mapa or campo in usados:
            continue
        mapa[b], _ = campo, usados.add(campo)
        ganadores.add(i)

    # Un título que compitió y perdió todas sus bandas no deja rastro en `mapa`,
    # y es el aviso que más importa: en factura1 'Cód. Barras Cód. Prov' se
    # queda sin columna porque el código de barras vive pegado a la descripción,
    # y el síntoma —12 descripciones con el código de barras adelante— aparece
    # recién al comparar contra el ground truth. Distinto de `titulos_sin_datos`,
    # que son los que ni siquiera llegaron a competir.
    perdedores = set(range(len(encabezado))) - ganadores
    avisos["titulos_sin_columna"] = [
        encabezado[i]["txt"] for i in sorted(perdedores)
        if encabezado[i]["txt"] not in avisos["titulos_sin_datos"]]
    return mapa


def _borde(v: float) -> str:
    return "·" if v in (float("-inf"), float("inf")) else f"{v:.0f}"


def _mapa_legible(limites: list[tuple[float, float]],
                  mapa: dict[int, str]) -> dict[str, str | None]:
    """La grilla tal como quedó: una entrada por columna, con su rango.

    Con nombre de columna y no de título a propósito. Una columna puede no tener
    título (la banda [1397,1526] de factura1) y un título puede no tener columna
    (`codigo_barras` en esa misma factura, que vive pegado a la descripción);
    las dos cosas se ven acá y ninguna se ve en un dict indexado por título.
    """
    return {f"{k} [{_borde(izq)},{_borde(der)}]": mapa.get(k)
            for k, (izq, der) in enumerate(limites)}


def desde_filas(filas: list[list[dict]], modo: str = "x",
                mapa_forzado: dict[int, str] | None = None
                ) -> tuple[dict, dict]:
    """Filas del pipeline → documento canónico (sólo la sección `items`).

    `modo`:
      "x"           asigna cada caja a la columna por su posición horizontal.
                    Aguanta columnas vacías (en factura2 `descuento` e `iva`
                    están vacías en las 44 filas).
      "posicional"  la caja k-ésima es la columna k-ésima. Es la línea de base
                    barata; se rompe con la primera celda vacía. Está para
                    medir cuánto peor es, no para usarla.

    Devuelve (doc, avisos). El doc trae sólo items: el recorte que produce el
    pipeline contiene la tabla, no la cabecera de la factura.

    Sobre los dos mapas que salen en los avisos: `mapa_alias` es lo que el
    vocabulario dedujo mirando sólo los títulos, y `mapa` es la grilla que de
    verdad se usó para repartir las cajas. Casi nunca son iguales, y la que
    importa es la segunda.
    """
    doc = documento_vacio()
    avisos = {"encabezado": None, "mapa": {}, "mapa_alias": {},
              "sin_nombre": [], "descartadas": [], "conflictos": [],
              "titulos_sin_datos": [], "titulos_partidos": [],
              "titulos_sin_columna": [], "columnas_sin_titulo": [],
              "modo": modo, "origen_mapa": "alias"}

    idx, mapa, punt = elegir_encabezado(filas)
    if mapa_forzado is not None:
        idx = idx if idx is not None else 0
        mapa, avisos["origen_mapa"] = dict(mapa_forzado), "forzado"
    if idx is None:
        avisos["error"] = "no encontré la fila de encabezados"
        return doc, avisos

    encabezado = sorted(filas[idx], key=_x0)
    avisos["encabezado"] = [c["txt"] for c in encabezado]
    avisos["mapa_alias"] = {encabezado[i]["txt"]: campo
                            for i, campo in mapa.items() if i < len(encabezado)}
    avisos["sin_nombre"] = [c["txt"] for i, c in enumerate(encabezado)
                            if i not in mapa]

    filas_datos = filas[idx + 1:]

    # La grilla sale de los datos; el encabezado sólo la nombra. Si no se puede
    # descubrir (pocas filas, todo pegado), se cae al reparto por encabezado.
    limites = _subdividir_bandas(_bandas_desde_datos(filas_datos), filas_datos)
    if len(limites) >= 2 and modo != "posicional":
        avisos["origen_grilla"] = f"datos ({len(limites)} columnas)"
        mapa = _nombrar_bandas(encabezado, limites, avisos)
        avisos["columnas_sin_titulo"] = sorted(
            set(range(len(limites))) - set(mapa))
    else:
        limites = _limites_columnas(encabezado)
        avisos["origen_grilla"] = "encabezado"

    # Después de nombrar las bandas y no antes: `_nombrar_bandas` reemplaza el
    # mapa del vocabulario por completo, y `esenciales` es el número con el que
    # `estandarizar` decide si le pide ayuda al LLM. Medirlo sobre el mapa que
    # ya se descartó es decidir sobre algo que no se usó.
    avisos["mapa"] = _mapa_legible(limites, mapa)
    avisos["esenciales"] = sorted(set(mapa.values()) & set(CAMPOS_ESENCIALES))

    for fila in filas_datos:
        if modo == "posicional":
            orden = sorted(fila, key=_x0)
            it = {c: None for c in CAMPOS_ITEM}
            for k, caja in enumerate(orden):
                campo = mapa.get(k)
                if campo in CAMPOS_ITEM:
                    it[campo] = (_num(caja["txt"]) if campo in _ITEM_NUMERICOS
                                 else _txt(caja["txt"]))
        else:
            it, conflictos = _fila_a_item(fila, limites, mapa)
            if conflictos:
                avisos["conflictos"].append(
                    {"fila": len(doc["items"]), "detalle": conflictos})

        if _es_item(it):
            doc["items"].append(it)
        else:
            # No se descarta en silencio: la línea 'DESPACHO: S/FACT 102141'
            # de factura5 no es un item, y tampoco lo son las de totales.
            avisos["descartadas"].append(" | ".join(c["txt"] for c in fila))
    return doc, avisos


# --------------------------------------------------------------------------
# Nivel 2 — el LLM, sólo sobre el encabezado
# --------------------------------------------------------------------------
# Se le mandan únicamente los textos de la fila de encabezados: una llamada de
# ~100 tokens, independiente de la cantidad de filas (factura2 con 44 items
# cuesta lo mismo que factura4 con 1). Los números nunca pasan por el modelo,
# que es donde un modelo chico alucina caro y sin dejar rastro.

MODELO_TEXTO = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

PROMPT_ENCABEZADO = (
    "Te paso los encabezados de columna de la tabla de items de una factura "
    "argentina, numerados de izquierda a derecha, tal como los leyó un OCR "
    "(pueden tener errores de lectura).\n"
    "Asigná a cada uno el campo canónico que le corresponde, eligiendo de "
    "esta lista: {campos}.\n"
    "Si un encabezado no corresponde a ninguno, omitilo. No inventes campos.\n"
    "Respondé SOLO con un objeto JSON {{\"<indice>\": \"<campo>\"}}, "
    "sin markdown ni explicación.\n\n"
    "Encabezados:\n{encabezados}"
)


def mapear_encabezado_llm(fila: list[dict], modelo: str = MODELO_TEXTO,
                          verbose: bool = True) -> dict[int, str]:
    """Encabezados → mapeo, pidiéndoselo a un LLM de texto."""
    from vision import get_client            # import tardío: esquema.py solo no
                                             # necesita openai ni paddle
    orden = sorted(fila, key=_x0)
    listado = "\n".join(f"{i}: {c['txt']}" for i, c in enumerate(orden))
    campos = ", ".join(CAMPOS_ITEM)

    rta = get_client().chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": PROMPT_ENCABEZADO.format(
            campos=campos, encabezados=listado)}],
        max_tokens=256, temperature=0.0,
        extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
    ).choices[0].message.content

    m = re.search(r"\{.*\}", rta.replace("```json", "").replace("```", ""), re.S)
    if not m:
        if verbose:
            print(f"[encabezado-llm] no hay JSON en la respuesta: {rta[:200]}")
        return {}
    try:
        crudo = json.loads(m.group())
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[encabezado-llm] JSON inválido ({e})")
        return {}

    mapa, usados = {}, set()
    for k, v in crudo.items():
        if not str(k).strip().isdigit():
            continue
        campo = clave(v)
        campo = next((c for c in CAMPOS_ITEM if clave(c) == campo), None)
        if campo and campo not in usados:
            mapa[int(k)] = campo
            usados.add(campo)
    return mapa


def estandarizar(filas: list[list[dict]], usar_llm: bool = True,
                 minimo_esenciales: int = 3, verbose: bool = True):
    """Nivel 1 con respaldo en el nivel 2.

    El LLM entra sólo si la tabla de alias no encontró encabezado o dejó menos
    de `minimo_esenciales` de los cinco campos que importan. En los formatos
    que ya conocemos no debería dispararse nunca; está para el formato nuevo.
    """
    doc, avisos = desde_filas(filas)
    if not usar_llm:
        return doc, avisos

    esenciales = len(avisos.get("esenciales", []))
    if avisos.get("error") is None and esenciales >= minimo_esenciales:
        return doc, avisos

    idx, _, _ = elegir_encabezado(filas, minimo=1)
    if idx is None:
        idx = 0
    if verbose:
        print(f"[estandarizar] alias mapeó {esenciales} esenciales; pido al LLM")
    mapa = mapear_encabezado_llm(filas[idx], verbose=verbose)
    if not mapa:
        return doc, avisos

    doc2, avisos2 = desde_filas(filas, mapa_forzado=mapa)
    avisos2["origen_mapa"] = "llm"
    return doc2, avisos2


# --------------------------------------------------------------------------
# Validación sin ground truth
# --------------------------------------------------------------------------

def validar_aritmetica(doc: dict, tol_rel: float = 0.01) -> pd.DataFrame:
    """`cantidad × precio ≈ importe`, línea por línea.

    No necesita ground truth: sirve para ver dónde está roto el pipeline antes
    de compararlo contra nada, y es el filtro que decide qué filas valdría la
    pena mandarle a un LLM.
    """
    filas = []
    for i, it in enumerate(doc["items"]):
        cant, precio, imp = it["cantidad"], it["precio_unitario"], it["importe"]
        esperado = None
        if cant is not None and precio is not None:
            esperado = cant * precio
            if it["descuento_pct"]:
                esperado *= 1 - it["descuento_pct"] / 100
        if esperado is None or imp is None:
            estado = "incompleta"
            dif = None
        else:
            dif = imp - esperado
            estado = "ok" if abs(dif) <= max(0.02, tol_rel * abs(imp)) else "falla"
        filas.append({"i": i, "codigo": it["codigo"], "cantidad": cant,
                      "precio": precio, "importe": imp, "esperado": esperado,
                      "dif": dif, "estado": estado})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Comparación contra el ground truth
# --------------------------------------------------------------------------

def _comparable(campo: str, v):
    """Valor listo para comparar: los números por valor, el texto sin ruido."""
    if v is None:
        return None
    if campo in _ITEM_NUMERICOS:
        return round(float(v), 2)
    return re.sub(r"\s+", " ", sin_acentos(str(v)).upper()).strip()


def comparar_items(doc_pipeline: dict, doc_truth: dict,
                   campos: tuple[str, ...] = CAMPOS_ESENCIALES) -> dict:
    """Empareja items y mide acierto por campo.

    Emparejamiento por `codigo` cuando los dos lados lo tienen (es el campo más
    discriminante y el que menos se parte); las líneas sin código se emparejan
    por orden entre las que quedaron sueltas. Las que no emparejan se cuentan
    aparte: una fila fusionada aparece como un item de más y uno faltante, y
    eso no se puede esconder dentro de un promedio por campo.
    """
    pend_p = list(enumerate(doc_pipeline["items"]))
    pend_t = list(enumerate(doc_truth["items"]))
    pares: list[tuple[dict, dict]] = []

    por_codigo: dict[str, list[int]] = {}
    for j, it in pend_t:
        c = _comparable("codigo", it["codigo"])
        if c:
            por_codigo.setdefault(c, []).append(j)

    usados_t: set[int] = set()
    resto_p = []
    for _, it in pend_p:
        c = _comparable("codigo", it["codigo"])
        libres = [j for j in por_codigo.get(c, []) if j not in usados_t] if c else []
        if libres:
            j = libres[0]
            usados_t.add(j)
            pares.append((it, doc_truth["items"][j]))
        else:
            resto_p.append(it)

    resto_t = [it for j, it in pend_t if j not in usados_t]
    for it_p, it_t in zip(resto_p, resto_t):
        pares.append((it_p, it_t))

    conteo = {c: {"ok": 0, "contiene": 0, "mal": 0, "faltante": 0} for c in campos}
    errores = []
    for it_p, it_t in pares:
        for c in campos:
            vp, vt = _comparable(c, it_p[c]), _comparable(c, it_t[c])
            if vt is None:
                continue
            if vp is None:
                conteo[c]["faltante"] += 1
            elif vp == vt:
                conteo[c]["ok"] += 1
            else:
                # "contiene": el valor correcto está adentro de lo leído, con
                # algo pegado. Separa perder un dato de arrastrar basura, que
                # son dos problemas distintos con dos soluciones distintas.
                # En factura1 la descripción trae el código de barras adelante
                # porque el OCR devuelve las dos cajas tocándose: el dato está,
                # la columna no.
                if isinstance(vt, str) and isinstance(vp, str) and vt in vp:
                    conteo[c]["contiene"] += 1
                else:
                    conteo[c]["mal"] += 1
                errores.append({"campo": c, "esperado": it_t[c], "leido": it_p[c],
                                "codigo": it_t["codigo"]})

    filas_res = []
    for c, v in conteo.items():
        total = max(1, v["ok"] + v["contiene"] + v["mal"] + v["faltante"])
        filas_res.append({"campo": c, **v,
                          "exacta": round(v["ok"] / total, 3),
                          "laxa": round((v["ok"] + v["contiene"]) / total, 3)})
    resumen = pd.DataFrame(filas_res)

    return {"resumen": resumen,
            "errores": pd.DataFrame(errores),
            "items_pipeline": len(doc_pipeline["items"]),
            "items_truth": len(doc_truth["items"]),
            "emparejados": len(pares),
            "sin_emparejar_pipeline": max(0, len(resto_p) - len(resto_t)),
            "sin_emparejar_truth": max(0, len(resto_t) - len(resto_p))}


def detalle_sin_mapear(carpeta: str | Path = "imagenes",
                       excluir: set[str] | None = None) -> pd.DataFrame:
    """Qué campos del ground truth no entraron al esquema, y en cuál factura."""
    filas = []
    for nombre, (_, av) in normalizar_carpeta(carpeta, excluir).items():
        for ruta in av["sin_mapear"]:
            filas.append({"factura": nombre, "campo": ruta})
    return pd.DataFrame(filas)
