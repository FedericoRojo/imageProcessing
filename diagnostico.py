"""diagnostico.py — herramientas para mirar qué pasó, no para producir resultados.

Nada de acá corre en el camino normal del pipeline. Son las cuatro preguntas
que uno se hace cuando el resultado vino mal:

    ¿qué recuadro eligió el LLM?          previsualizar_bbox
    ¿la foto está girada?                 detectar_rotacion
    ¿el detector cortó texto en el borde?  cajas_en_borde
    ¿qué cambia si toco el OCR?           comparar_variantes

Todas son caras (una llamada a la API o pasadas de OCR de más), por eso viven
aparte: que sea evidente que se piden a mano y no se pagan por default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from imagen import normalizar_orientacion
from ocr import get_ocr, ocr_tabla
from vision import detectar_tabla


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
