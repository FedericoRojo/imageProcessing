"""filas.py — cajas de texto con coordenadas → filas.

La pregunta es una sola: **qué cajas son el mismo renglón**. No sabe de
facturas, de columnas ni de PaddleOCR — trabaja sobre `rec_polys`/`rec_texts`
y coordenadas. Por eso no importa `paddleocr` y se puede medir sobre corridas
guardadas (`ocr.cargar_ocr_json`), que es lo que hace `banco.py`.

Lo difícil acá no es el agrupado sino sus dos parámetros: cuánta inclinación
residual queda (`inclinacion_de_cajas`) y cuál es la separación real entre
renglones (`paso_filas`). Los dos están medidos contra el corpus y la prosa de
cada función dice contra qué.

    from filas import agrupar_filas, imprimir_filas
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cajas(res) -> list[dict]:
    cajas = []
    for poly, txt, score in zip(res["rec_polys"], res["rec_texts"], res["rec_scores"]):
        p = np.array(poly)
        cajas.append({
            "x": float(p[:, 0].min()),      # alias histórico de x0
            "x0": float(p[:, 0].min()),
            "x1": float(p[:, 0].max()),     # hace falta para asignar columnas
            "y": float(p[:, 1].mean()),
            "alto": float(p[:, 1].max() - p[:, 1].min()),
            "txt": txt,
            "score": float(score),
        })
    return cajas


def _agrupar(cajas: list[dict], tol: float) -> list[list[dict]]:
    """Encadena por `yp` con tolerancia `tol`. `cajas` ya viene ordenada."""
    filas: list[list[dict]] = []
    actual: list[dict] = [cajas[0]]
    for c in cajas[1:]:
        # comparo contra el promedio de la fila en curso, no contra el primer
        # elemento: evita que una fila se corte por una caja apenas desplazada
        y_fila = sum(d["yp"] for d in actual) / len(actual)
        if abs(c["yp"] - y_fila) <= tol:
            actual.append(c)
        else:
            filas.append(sorted(actual, key=lambda d: d["x"]))
            actual = [c]
    filas.append(sorted(actual, key=lambda d: d["x"]))
    return filas


def inclinacion_de_cajas(cajas: list[dict], tol: float,
                         b_max: float = 0.02, paso: float = 0.0002) -> float:
    """Inclinación residual leída de las coordenadas del OCR.

    Busca la pendiente que deja las filas más compactas. Es la red de seguridad
    de `enderezar`: si la imagen ya vino derecha devuelve ~0 y no cambia nada,
    y si el recorte no pasó por el enderezado (el LLM no encontró la tabla)
    rescata igual la mayor parte del problema.

    No se puede hacer leyendo el ángulo de cada polígono: PaddleOCR los devuelve
    alineados al eje y la mediana del ángulo por caja da 0.00°. Hay que mirar
    todas las filas juntas, donde la deriva se acumula sobre el ancho completo.
    """
    xs = np.array([(c["x0"] + c["x1"]) / 2 for c in cajas], dtype=float)
    ys = np.array([c["y"] for c in cajas], dtype=float)
    xs = xs - xs.mean()

    mejor, mejor_k = 0.0, None
    for b in np.arange(-b_max, b_max + paso / 2, paso):
        yp = ys - b * xs
        orden = np.argsort(yp)
        # costo = dispersión interna total; mínimo cuando las filas son planas
        k, actual = 0.0, [yp[orden[0]]]
        for i in orden[1:]:
            v = yp[i]
            if abs(v - sum(actual) / len(actual)) <= tol:
                actual.append(v)
            else:
                k += max(actual) - min(actual)
                actual = [v]
        k += max(actual) - min(actual)
        if mejor_k is None or k < mejor_k:
            mejor, mejor_k = float(b), k
    return mejor


def paso_filas(cajas: list[dict], lag_min: int = 8,
               lag_max: int = 400) -> float | None:
    """Separación típica entre renglones, por autocorrelación de los centros.

    Por qué no alcanza con la mediana de las diferencias entre centros
    consecutivos: dentro de una fila esa diferencia es casi cero, y si hay filas
    fusionadas la mediana sale al doble del paso real. La autocorrelación mira
    la periodicidad de todo el bloque, así que aguanta filas fusionadas, filas
    faltantes y líneas sueltas que no son de la tabla.

    Validado contra el corpus: da 26.0 px donde el paso real es 26.9 (factura2),
    63.0 contra 62.9 (factura4) y 43.0 contra 43.5 (factura1).
    """
    yp = np.array([c["yp"] for c in cajas], dtype=float)
    yp = yp - yp.min()
    n = int(yp.max()) + 1
    if n < 2 * lag_min:
        return None
    perfil = np.zeros(n + 2)
    np.add.at(perfil, np.rint(yp).astype(int), 1.0)
    perfil -= perfil.mean()
    ac = np.correlate(perfil, perfil, mode="full")[len(perfil) - 1:]
    hi = min(lag_max, len(ac) - 1)
    if hi <= lag_min:
        return None
    return float(int(np.argmax(ac[lag_min:hi])) + lag_min)


def agrupar_filas(res, tol: float | None = None, b: float | None = None,
                  verbose: bool = False) -> list[list[dict]]:
    """Agrupa las cajas de texto en filas por coordenada y.

    `b` es la inclinación residual: con None se estima de las coordenadas, con
    0.0 se desactiva. El agrupado se hace sobre `y - b*x`, que es lo que deja
    las filas horizontales.

    `tol` con None se resuelve en dos pasadas. La primera usa el 60% del alto
    mediano de caja — el criterio viejo — sólo para poder medir el paso entre
    renglones; la segunda usa una fracción de ese paso. El cambio importa
    porque el alto de caja NO es la altura del renglón: `text_det_unclip_ratio`
    dilata las cajas y en factura2 el alto mediano (38 px) supera el paso real
    (27 px). Una tolerancia del 60% de 38 es el 85% del paso, y con ese margen
    cualquier desvío encadena la fila siguiente.

    Se toma el mínimo entre los dos criterios y no el del paso a secas: en las
    tablas de renglón alto (factura1 y factura4 tienen paso de 43 y 63 px) la
    fracción del paso se vuelve más permisiva que el criterio viejo, y ahí no
    hay nada que ganar.

    El 0.65 sale de barrer el factor contra el corpus. No es un número elegido
    al filo: entre 0.55 y 0.75 el resultado no se mueve (factura2 da 48-49 filas
    contra las 48 esperadas, y las otras cuatro quedan clavadas), así que se
    toma el centro de esa meseta. Por debajo de 0.50 empieza a partir
    descripciones que ocupan dos renglones; a partir de 0.85 vuelve la fusión
    (factura2 cae a 42 filas con 7 dobles).
    """
    cajas = _cajas(res)
    if not cajas:
        return []

    alto = float(np.median([c["alto"] for c in cajas]))
    tol_alto = 0.6 * alto

    if b is None:
        b = inclinacion_de_cajas(cajas, tol_alto)
    for c in cajas:
        c["yp"] = c["y"] - b * (c["x0"] + c["x1"]) / 2
    cajas.sort(key=lambda d: d["yp"])

    if tol is None:
        pas = paso_filas(cajas)
        tol = min(tol_alto, 0.65 * pas) if pas else tol_alto
        if verbose:
            print(f"[agrupar_filas] b={b:+.4f} alto={alto:.0f} "
                  f"paso={pas if pas else float('nan'):.0f} tol={tol:.1f}")

    return _agrupar(cajas, tol)


def filas_a_dataframe(filas: list[list[dict]]) -> pd.DataFrame:
    """Filas → DataFrame crudo (una columna por posición, todavía sin mapear)."""
    return pd.DataFrame([[c["txt"] for c in f] for f in filas])


def imprimir_filas(filas: list[list[dict]]) -> None:
    for f in filas:
        print(" | ".join(c["txt"] for c in f))
