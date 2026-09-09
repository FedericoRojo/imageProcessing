"""imagen.py — todo lo que le pasa a los píxeles antes de que los vea el OCR.

Orientación (paso 0), enderezado (paso 1.5) y recorte + zoom (paso 1). No sabe
de facturas ni de LLMs: recibe una imagen o un bbox ya calculado y devuelve
otra imagen. Quién decide *dónde* recortar es `vision.py`; quién orquesta los
pasos es `pipeline.py`.

Es también el dueño de `recortes/`: acá se define dónde se escriben los
recortes del paso 1 y desde acá se los lista o se los baja.

    from imagen import normalizar_orientacion, enderezar, recortar_zoom

Por qué hay un paso de enderezado
---------------------------------
Medido sobre el corpus: las tablas vienen inclinadas entre 0.14° y 0.53°. Suena
despreciable y no lo es. Sobre los ~1490 px de ancho de la tabla de factura2,
medio grado son 13 px de deriva vertical, y el paso entre filas es de 27 px. La
mitad de un renglón: alcanza para que el agrupado por coordenada y encadene una
fila con la siguiente. En factura2 eso hacía que el OCR entregara 38 filas donde
hay 44 items, y una fila fusionada no se pierde en silencio — se convierte en un
item mal leído.

El ángulo NO se puede leer de las cajas del OCR: PaddleOCR devuelve los
polígonos alineados al eje (146 de 234 cajas de factura2 tienen y0 == y1 exacto)
y la mediana del ángulo por caja da 0.00°. La cuantización al píxel borra medio
grado. Sólo es recuperable a nivel layout, donde la deriva se acumula sobre todo
el ancho, o midiéndolo sobre la imagen antes del OCR — que es lo que se hace acá.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:            # sólo para el chequeo de tipos: en runtime este
    from vision import TablaBBox   # módulo no depende de `vision` ni de openai

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

MARGEN_DEFAULT = 0.02   # % del alto/ancho que se agrega alrededor del bbox
ZOOM_DEFAULT = 2        # factor de ampliación del recorte
BORDE_DEFAULT = 30      # px de borde blanco agregados DESPUÉS del zoom

# Los recortes van todos a una carpeta en vez de quedar sueltos al lado de la
# imagen original: en Colab /content se llena de intermedios y bajarlos uno por
# uno es incómodo. Con la carpeta, `descargar_recortes()` se los lleva todos.
RECORTES_DIR = Path("recortes")

# Inclinación: pendiente (px de y por px de x), no grados. 0.045 son ~2.6°.
# Más que eso ya no es una tabla inclinada sino una foto torcida, y eso lo
# resuelve `rotacion`/`detectar_rotacion`, no este paso.
INCLINACION_MAX = 0.045
INCLINACION_MIN = 0.0008   # ~0.05°: por debajo, rotar sólo agrega interpolación

# Por qué el borde blanco: el detector de texto de PaddleOCR recorta las cajas
# contra el límite de la imagen, y el preprocesador (unwarping) puede correr la
# imagen unos píxeles. Cualquiera de las dos cosas se come el primer carácter de
# la columna más a la izquierda. Con un borde blanco, ningún texto queda pegado
# al borde y el problema desaparece.


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


def _margenes(margen: float | tuple[float, float, float, float]):
    """Acepta un número (los 4 lados iguales) o (izq, arriba, der, abajo)."""
    if isinstance(margen, (int, float)):
        return (float(margen),) * 4
    izq, arr, der, aba = margen
    return float(izq), float(arr), float(der), float(aba)


# --------------------------------------------------------------------------
# Paso 1.5 — enderezado de la tabla recortada
# --------------------------------------------------------------------------
# Se hace DESPUÉS del recorte y ANTES del zoom, y las dos cosas importan.
#
# Después del recorte, porque el estimador necesita ver sólo la tabla. Medido
# sobre las cinco facturas, estimando sobre la hoja entera contra estimando
# sobre la zona de la tabla, y comparando contra el ángulo que efectivamente
# arregla el agrupado de filas:
#
#     factura    ángulo real   hoja entera   zona de la tabla
#     factura2     -0.0087       -0.0010          -0.0074
#     factura5     -0.0074       -0.0010          -0.0088
#     factura4     -0.0093       -0.0096          -0.0094
#
# Sobre la hoja entera el estimador recupera el 11% del ángulo en factura2. El
# logo, el bloque del emisor y el recuadro de totales tienen su propia
# distribución vertical y arrastran la estimación. Sobre el recorte, acierta.
#
# Antes del zoom, porque así la rotación interpola a resolución nativa y el
# LANCZOS del zoom queda como última operación, en vez de ampliar ×2 y después
# repartir cada píxel entre vecinos.


def _tinta(img: Image.Image, lado_max: int = 1600):
    """Coordenadas (x, y) de los píxeles de tinta, en escala reducida.

    El umbral es deliberadamente grosero: para estimar una inclinación no hace
    falta segmentar bien el texto, alcanza con que los renglones se distingan
    del fondo. Se usa el percentil 2 y no el mínimo porque un solo píxel negro
    (una mota, el borde de un sello) corre el umbral para toda la imagen.
    """
    g = ImageOps.grayscale(img)
    esc = min(1.0, lado_max / max(g.size))
    if esc < 1.0:
        g = g.resize((max(1, int(g.width * esc)), max(1, int(g.height * esc))),
                     Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    med, piso = float(np.median(a)), float(np.percentile(a, 2))
    ys, xs = np.nonzero(a < med - 0.35 * (med - piso))
    return xs.astype(np.float32), ys.astype(np.float32), a.shape[0]


def estimar_inclinacion(img: Image.Image, b_max: float = INCLINACION_MAX,
                        paso: float = 0.0002, lado_max: int = 1600) -> float:
    """Pendiente de los renglones, por perfil de proyección.

    Devuelve `b` tal que `y - b*x` deja las filas horizontales — la misma
    convención que usa `agrupar_filas`.

    El método clásico es rotar la imagen por cada ángulo candidato y quedarse
    con el que da el perfil de proyección más "picudo" (renglones alineados =
    picos altos y valles vacíos). Rotar 450 veces es caro, así que en vez de
    rotar se acumulan los píxeles de tinta en bins de `y - b*x`, que para
    ángulos chicos es lo mismo y cuesta un bincount por candidato.

    El puntaje es la suma de cuadrados del perfil: crece cuando la misma
    cantidad de tinta se concentra en menos bins.
    """
    xs, ys, alto = _tinta(img, lado_max)
    if len(xs) < 50:
        return 0.0
    xs = xs - xs.mean()
    bs = np.arange(-b_max, b_max + paso / 2, paso, dtype=np.float32)
    mejor, mejor_p = 0.0, -1.0
    for b in bs:
        yp = ys - b * xs
        h = np.bincount((yp - yp.min()).astype(np.int32), minlength=alto + 2)
        p = float(np.dot(h, h))
        if p > mejor_p:
            mejor, mejor_p = float(b), p
    return mejor


def enderezar(img: Image.Image, b: float | None = None,
              verbose: bool = True) -> tuple[Image.Image, float]:
    """Rota la imagen para dejar los renglones horizontales.

    `b=None` lo estima. Devuelve (imagen, b_aplicado); b_aplicado es 0.0 si no
    hizo falta rotar, así que la función es idempotente: aplicada dos veces, la
    segunda no toca nada.

    El relleno va en blanco y no en negro porque las esquinas que deja la
    rotación entran igual al detector de texto, y una cuña negra pegada al
    borde es exactamente el tipo de cosa que le hace recortar la primera caja.
    """
    if b is None:
        b = estimar_inclinacion(img)
    if abs(b) < INCLINACION_MIN:
        if verbose:
            print(f"[enderezar] inclinación {b:+.4f}: por debajo del umbral, no roto")
        return img, 0.0

    # Convención: una fila cumple y = c + b*x, o sea que con b<0 el renglón sube
    # hacia la derecha (y crece hacia abajo). PIL rota antihorario con ángulos
    # positivos, así que el ángulo a aplicar es directamente atan(b).
    ang = math.degrees(math.atan(b))
    salida = img.convert("RGB").rotate(ang, resample=Image.BICUBIC, expand=True,
                                       fillcolor="white")
    if verbose:
        print(f"[enderezar] inclinación {b:+.4f} ({ang:+.2f}°) corregida")
    return salida, b


# --------------------------------------------------------------------------
# Paso 1 — recorte, zoom y borde
# --------------------------------------------------------------------------

def _ruta_recorte(origen: Path, guardar_en: str | Path | None) -> Path:
    """Dónde escribir el recorte: `<guardar_en>/<nombre>_tabla.png`.

    Del nombre se saca el sufijo `_derecha` que agrega `normalizar_orientacion`:
    lo que se descarga es el recorte de tal factura, no interesa por cuántos
    pasos intermedios pasó. `guardar_en=None` lo deja al lado de la imagen
    original, que es como se guardaba antes.
    """
    nombre = origen.stem
    if nombre.endswith("_derecha"):
        nombre = nombre[: -len("_derecha")]
    nombre += "_tabla.png"
    if guardar_en is None:
        return origen.with_name(nombre)
    carpeta = Path(guardar_en)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / nombre


def recortar_zoom(
    ruta: str | Path,
    bbox: TablaBBox,
    margen: float | tuple[float, float, float, float] = MARGEN_DEFAULT,
    zoom: int = ZOOM_DEFAULT,
    borde: int = BORDE_DEFAULT,
    enderezado: bool = True,
    guardar_en: str | Path | None = RECORTES_DIR,
    verbose: bool = True,
) -> Path:
    """bbox → imagen recortada, enderezada, ampliada y con borde blanco.

    `bbox` es un `vision.TablaBBox`; acá sólo se leen sus cuatro coordenadas,
    así que este módulo no necesita importar `vision`.

    `margen` puede ser un número (los 4 lados iguales) o una tupla
    (izquierda, arriba, derecha, abajo) en fracción del alto/ancho original.
    `borde` son píxeles de blanco que se agregan alrededor después del zoom,
    para que ningún texto quede pegado al borde de la imagen.
    `enderezado=False` saltea la corrección de inclinación, para poder medir
    contra la línea de base con `comparar_variantes`.
    `guardar_en` es la carpeta donde queda el recorte; `None` lo deja al lado de
    la imagen original. Reprocesar la misma factura pisa el archivo anterior.
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
    if enderezado:
        crop, _ = enderezar(crop, verbose=verbose)
    crop = crop.resize((int(crop.width * zoom), int(crop.height * zoom)),
                       Image.LANCZOS)
    if borde:
        crop = ImageOps.expand(crop.convert("RGB"), border=int(borde), fill="white")

    # PNG y no JPEG: es una imagen intermedia y recomprimir en JPEG le agrega
    # ruido justo en los bordes de los caracteres, que es lo que lee el OCR.
    salida = _ruta_recorte(ruta, guardar_en)
    crop.save(salida)
    if verbose:
        print(f"[recortar_zoom] recorte guardado en {salida}")
    return salida


# --------------------------------------------------------------------------
# La carpeta de recortes
# --------------------------------------------------------------------------

def listar_recortes(carpeta: str | Path = RECORTES_DIR) -> list[Path]:
    """Los recortes que hay guardados, del más nuevo al más viejo."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    return sorted(carpeta.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


def descargar_recortes(carpeta: str | Path = RECORTES_DIR, verbose: bool = True):
    """Se lleva los recortes a tu máquina.

    En Colab dispara la descarga del navegador: si hay un solo recorte baja el
    PNG, y si hay varios los zipea primero (Colab abre un diálogo por archivo,
    y con diez recortes eso es insoportable). Fuera de Colab no hay nada que
    descargar — los archivos ya están en el disco — así que sólo los lista.

    Devuelve la lista de recortes.
    """
    carpeta = Path(carpeta)
    recortes = listar_recortes(carpeta)
    if not recortes:
        if verbose:
            print(f"[descargar] no hay recortes en {carpeta.resolve()}")
        return []

    if verbose:
        print(f"[descargar] {len(recortes)} recorte(s) en {carpeta.resolve()}:")
        for r in recortes:
            print(f"   {r.name}")

    try:
        from google.colab import files      # type: ignore[import-not-found]
    except ImportError:
        return recortes                     # local: ya están donde tienen que estar

    if len(recortes) == 1:
        files.download(str(recortes[0]))
    else:
        import shutil

        zip_path = shutil.make_archive(str(carpeta), "zip", root_dir=carpeta)
        files.download(zip_path)
    return recortes
