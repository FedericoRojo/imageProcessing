"""encabezado.py — el vocabulario de títulos de columna de la tabla de items.

Lo único que este módulo sabe es CÓMO SE LLAMAN las cosas en el encabezado de
una factura. No sabe de geometría: no mira coordenadas, no arma columnas, no
toca los datos. Le entra texto y le sale un campo canónico.

Responde tres preguntas, de abajo hacia arriba:

    puntaje_columna(texto)    ¿a qué campo se parece este título, y cuánto?
    mapear_encabezado(fila)   dada una fila de títulos, ¿qué campo es cada uno?
    elegir_encabezado(filas)  ¿cuál de las primeras filas ES el encabezado?

y una cuarta, auxiliar, para cuando el OCR pega varios títulos en una caja:

    partir_titulo(texto)      '% Bonif. Neto Unit Neto Total' → 3 campos

Depende sólo de `texto.clave`. Quien lo usa es la capa de grilla, que sí sabe
de geometría y le pide a este módulo únicamente los nombres.

Nota sobre la salida de `elegir_encabezado`: en el pipeline actual, de sus tres
valores el que de verdad sobrevive es el ÍNDICE. El mapeo se recalcula después
contra la grilla real de columnas, porque un título puede estar centrado sobre
su columna, sobresalir sobre la vecina o pertenecer a una columna vacía, y eso
sólo se puede resolver con coordenadas — que acá no hay. El mapeo que sale de
este módulo es la mejor respuesta posible SIN mirar dónde caen los datos.
"""

from __future__ import annotations

import difflib

from texto import clave

__all__ = [
    "ALIAS_ENCABEZADO", "TIER_EXACTO", "TIER_PREFIJO", "TIER_DIFUSO",
    "CORTE_DIFUSO", "LARGO_MIN_PREFIJO",
    "puntaje_columna", "mapear_encabezado", "elegir_encabezado",
    "partir_titulo",
]


# --------------------------------------------------------------------------
# El vocabulario
# --------------------------------------------------------------------------
# Todas las formas en que vimos escrito cada campo, ya en forma de `clave()`
# (minúsculas, sin acentos, sin separadores). Agregar un formato nuevo es
# agregar una línea acá, no tocar lógica.

ALIAS_ENCABEZADO = {
    "codigo": ["codigo", "articulo", "art", "cod", "codart", "codigoarticulo",
               "item", "nro", "codproducto"],
    "codigo_barras": ["codbarras", "codigobarras", "codigodebarras", "ean",
                      "barras"],
    "codigo_proveedor": ["codproveedor", "codigoproveedor", "codprov"],
    "descripcion": ["descripcion", "producto", "detalle", "concepto",
                    "articulodescripcion", "denominacion"],
    "cantidad": ["cantidad", "cant", "ctd", "cdad", "cantid"],
    "unidad": ["umedid", "unidad", "unidadmedida", "um", "medida", "umed"],
    "precio_unitario": ["preciounitario", "precio", "punitario", "punit",
                        "preciounit", "netounitario", "netounit", "unitario",
                        "pruni", "preciolista"],
    "descuento_pct": ["bonif", "porcbonif", "bonificacion", "desc", "descuento",
                      "dto", "descto", "dcto"],
    "importe": ["importe", "netototal", "total", "importetotal", "subtotal",
                "montototal"],
    # `iva` NO es un campo canónico de item: cuando se arma la fila se descarta.
    # Está acá para que la columna quede TOMADA. La asignación de más abajo es
    # con unicidad y una columna sólo puede tener un nombre; si `IVA` no
    # estuviera en el vocabulario, esa columna quedaría libre y el primer campo
    # que ande buscando dónde caer por parecido difuso se la llevaría.
    # Es una defensa contra un caso conocido, no una solución general: una
    # columna 'LOTE' o 'VENCIMIENTO' vuelve a dejar el agujero abierto.
    "iva": ["iva", "alicuota", "aliciva", "ivapct"],
}


# --------------------------------------------------------------------------
# Los tres niveles de evidencia
# --------------------------------------------------------------------------
# Los valores están elegidos para que los rangos NUNCA se solapen:
#
#     exacto   3.0
#     prefijo  2.0
#     difuso   1.00 .. 1.18   (= TIER_DIFUSO + (ratio - CORTE_DIFUSO))
#
# Es decir: un match difuso jamás le gana a uno por prefijo, y un prefijo jamás
# le gana a un exacto, por bueno que sea el parecido. La alternativa ingenua
# —un solo `SequenceMatcher` con umbral— deja que el tercer decimal de un
# algoritmo de similitud decida si 'Desc.' es descripción o descuento.

TIER_EXACTO, TIER_PREFIJO, TIER_DIFUSO = 3.0, 2.0, 1.0

# Umbral del match difuso. Existe SÓLO para sobrevivir errores del OCR
# ('DESCRIPCICN' contra 'descripcion' da 0.91). Es un número calibrado a mano:
# no hay nada que diga que 0.80 o 0.85 sean peores.
CORTE_DIFUSO = 0.82

# Un título de menos de 3 caracteres no puede matchear por prefijo: 'C' es
# prefijo de codigo, cantidad, concepto, cod y ctd a la vez.
LARGO_MIN_PREFIJO = 3

# Cuánto se premia a `descuento_pct` cuando el título trae un '%'. Alcanza para
# ponerlo primero en la cola del greedy.
BONO_PORCENTAJE = 2.0


def puntaje_columna(texto: str) -> dict[str, float]:
    """Cuánto se parece un título a cada campo canónico.

    Devuelve un dict, no un único campo: un mismo título puede puntuar contra
    varios a la vez, y eso es información, no ambigüedad. 'Desc.' puntúa 3.0
    contra `descuento_pct` (alias exacto) y 2.0 contra `descripcion` (prefijo);
    quién se lo queda lo decide `mapear_encabezado` mirando el encabezado
    entero, no esta función mirando un título suelto.
    """
    k = clave(texto)
    if not k:
        return {}

    puntajes: dict[str, float] = {}
    for campo, alias in ALIAS_ENCABEZADO.items():
        mejor = 0.0
        for a in alias:
            mejor = max(mejor, _puntaje_alias(k, a))
        if mejor:
            puntajes[campo] = mejor

    return _pista_porcentaje(texto, puntajes)


def _puntaje_alias(k: str, alias: str) -> float:
    """Puntaje de una clave contra UN alias, en los tres niveles."""
    if k == alias:
        return TIER_EXACTO

    # Bidireccional a propósito: cubre el título abreviado ('cant' ⊂ 'cantidad')
    # y el ampliado ('cantidadfacturada' ⊃ 'cantidad'). Con una sola dirección
    # se pierde la mitad de los casos.
    if len(k) >= LARGO_MIN_PREFIJO and (alias.startswith(k) or k.startswith(alias)):
        return TIER_PREFIJO

    ratio = difflib.SequenceMatcher(None, k, alias).ratio()
    if ratio >= CORTE_DIFUSO:
        return TIER_DIFUSO + (ratio - CORTE_DIFUSO)
    return 0.0


def _pista_porcentaje(texto: str, puntajes: dict[str, float]) -> dict[str, float]:
    """El signo '%' es una pista fuerte y gratis: '% Desc.' es un porcentaje.

    Mira el texto CRUDO porque `clave()` borra el '%'.

    En el corpus actual esto no cambia nada: clave('% Desc.') es 'desc', que ya
    es alias exacto de `descuento_pct` (3.0) y sólo prefijo de `descripcion`
    (2.0). Es un seguro para el caso en que la columna de descripción de verdad
    se lea mal y sólo alcance un match difuso: sin la regla, la columna del
    porcentaje podría llevarse `descripcion` y dejar sin nombre a la buena.
    Barato y sin efectos laterales, pero es una heurística sobre un carácter.
    """
    if "%" not in texto:
        return puntajes
    puntajes.pop("descripcion", None)
    if "descuento_pct" in puntajes:
        puntajes["descuento_pct"] += BONO_PORCENTAJE
    return puntajes


# --------------------------------------------------------------------------
# De una fila de títulos al mapeo de columnas
# --------------------------------------------------------------------------

def mapear_encabezado(fila: list[dict]) -> tuple[dict[int, str], dict[int, float]]:
    """Fila de encabezados → ({índice de columna: campo}, {índice: puntaje}).

    Asignación greedy con UNICIDAD por los dos lados: una columna no puede
    tener dos nombres y un campo no puede quedar en dos columnas.

    La unicidad es lo que salva el caso de factura1, donde conviven 'P. Unit.'
    (precio de lista) y 'Neto Unit' (después de la bonificación). `clave()` los
    deja en 'punit' y 'netounit', que son AMBOS alias exactos de
    `precio_unitario`: empatan en 3.0. Sin unicidad las dos columnas se llaman
    igual, la segunda pisa a la primera al armar el item, y te queda un solo
    precio sin que nada avise. Con unicidad, la segunda queda sin nombre — y
    eso sí es visible en `avisos["sin_nombre"]`.

    PARCHE, y conviene saberlo: el desempate a igual puntaje es por índice, o
    sea gana la columna de MÁS A LA IZQUIERDA. No hay principio detrás; está
    calibrado contra factura1, donde el ground truth guarda el precio de lista,
    que es el de la izquierda. En una factura donde el precio relevante sea el
    de la derecha esto falla en silencio: devuelve un número plausible que es
    el precio equivocado. La red que lo atrapa es `validar_aritmetica`
    (cantidad × precio ≈ importe), no este módulo.
    """
    candidatos = [(p, i, campo)
                  for i, celda in enumerate(fila)
                  for campo, p in puntaje_columna(celda["txt"]).items()]
    candidatos.sort(key=lambda t: (-t[0], t[1]))   # mejor puntaje; luego, izquierda

    mapa: dict[int, str] = {}
    puntajes: dict[int, float] = {}
    campos_usados: set[str] = set()
    for p, i, campo in candidatos:
        if i in mapa or campo in campos_usados:
            continue
        mapa[i] = campo
        puntajes[i] = p
        campos_usados.add(campo)
    return mapa, puntajes


def elegir_encabezado(filas: list[list[dict]], max_filas: int = 5,
                      minimo: int = 2) -> tuple[int | None, dict[int, str],
                                                dict[int, float]]:
    """Busca la fila de encabezados entre las primeras `max_filas`.

    Devuelve (índice, mapa, puntajes), o (None, {}, {}) si ninguna califica.

    NO se asume la fila 0. El recorte que devuelve el LLM de visión lleva un
    margen del 2% arriba —puesto a propósito, porque recortar justo se come la
    fila de títulos— y ahí suele quedar texto suelto del bloque de arriba. O
    sea: el margen que evita perder el encabezado es el mismo que impide
    asumir que es la primera fila.

    Gana la fila con más columnas reconocidas con match FUERTE (exacto o
    prefijo; el difuso no cuenta acá). La asimetría es deliberada: para decidir
    si una fila es el encabezado hace falta evidencia dura, porque cualquier
    fila de datos acumula matches difusos por casualidad. Para nombrar una
    columna que ya se sabe que es un título, el difuso alcanza.

    `minimo` es el que decide el fracaso: por debajo devuelve None, y eso es lo
    que dispara el nivel 2 (el LLM sobre el encabezado).

    `max_filas=5` y `minimo=2` son empíricos.
    """
    mejor_idx: int | None = None
    mejor_mapa: dict[int, str] = {}
    mejor_puntajes: dict[int, float] = {}
    mejor_fuertes = -1

    for i, fila in enumerate(filas[:max_filas]):
        mapa, puntajes = mapear_encabezado(fila)
        fuertes = sum(1 for p in puntajes.values() if p >= TIER_PREFIJO)
        # Estricto: a igual cantidad de matches fuertes gana la fila de más
        # arriba, que es donde está el encabezado de verdad.
        if fuertes > mejor_fuertes:
            mejor_idx, mejor_mapa, mejor_puntajes, mejor_fuertes = (
                i, mapa, puntajes, fuertes)

    if mejor_fuertes < minimo:
        return None, {}, {}
    return mejor_idx, mejor_mapa, mejor_puntajes


# --------------------------------------------------------------------------
# Títulos que el OCR pegó
# --------------------------------------------------------------------------

def partir_titulo(texto: str, max_tokens: int = 4) -> list[str]:
    """Desarma un título que el OCR pegó, usando el vocabulario de columnas.

    En factura1 el OCR devuelve '% Bonif. Neto Unit Neto Total' como UNA caja:
    tres títulos de columna en una. Sin desarmarlo, `importe` nunca recibe
    nombre y la factura entera queda sin importes.

    Segmentación greedy de izquierda a derecha: en cada posición se prueban
    todos los cortes posibles de hasta `max_tokens` palabras y gana el de MEJOR
    PUNTAJE, con el más largo como desempate. Los tokens que no matchean nada
    se saltean.

    Tomar directamente el corte más largo que supere el umbral es un error, y
    fue un bug real: 'Neto Unit Neto Total' entero matchea 'netounit' por
    prefijo, se come los cuatro tokens y 'Neto Total' nunca llega a existir.
    El match exacto tiene que ganarle al prefijo aunque sea más corto.

    Vive en este módulo y no en la capa de grilla porque es puro vocabulario:
    no mira una sola coordenada. Quién decide que un título hay que partirlo
    —eso sí es geometría— es la capa de grilla, que llama acá.
    """
    tokens = str(texto).split()
    campos: list[str] = []
    i = 0
    while i < len(tokens):
        mejor = _mejor_corte(tokens, i, max_tokens)
        if mejor is None:
            i += 1
            continue
        _, i, campo = mejor
        if campo not in campos:
            campos.append(campo)
    return campos


def _mejor_corte(tokens: list[str], desde: int,
                 max_tokens: int) -> tuple[float, int, str] | None:
    """(puntaje, hasta, campo) del mejor grupo de palabras que arranca en `desde`.

    Sólo se aceptan matches fuertes (>= TIER_PREFIJO): partir un título por un
    parecido difuso inventaría columnas que no existen.
    """
    mejor: tuple[float, int, str] | None = None
    for hasta in range(desde + 1, min(len(tokens), desde + max_tokens) + 1):
        puntajes = puntaje_columna(" ".join(tokens[desde:hasta]))
        if not puntajes:
            continue
        campo = max(puntajes, key=puntajes.get)
        p = puntajes[campo]
        # (p, hasta) > (mejor_p, mejor_hasta): primero el puntaje, después el
        # corte más largo. Este orden es el que arregla 'Neto Unit Neto Total'.
        if p >= TIER_PREFIJO and (mejor is None or (p, hasta) > mejor[:2]):
            mejor = (p, hasta, campo)
    return mejor
