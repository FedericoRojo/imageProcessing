"""banco.py — medición end-to-end sobre el corpus real.

Levanta las salidas de OCR guardadas en `data/ocr/`, las pasa por el agrupado
y la estandarización, y las compara contra los ground truth normalizados.

    python3 banco.py            # resumen de las 5
    python3 banco.py factura2   # detalle de una

El OCR es determinista para una imagen dada, así que replayear su salida
guardada equivale a correrlo, y cuesta cero llamadas y cero GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from esquema import (CAMPOS_ESENCIALES, EXCLUIDAS_ITEMS, comparar_items,
                     desde_filas, desde_ground_truth, validar_aritmetica)
from pipeline import agrupar_filas, cargar_ocr_json

# Las salidas de OCR viven afuera del repo, una carpeta por corrida
# (`respuestaGenerada1`, y las que vengan). Las variables de entorno permiten
# medir una corrida vieja contra el mismo ground truth sin tocar el código:
#
#     DIR_OCR=../imagenes/respuetaGenerada python3 banco.py
DIR_OCR = Path(os.environ.get("DIR_OCR", "../imagenes/respuestaGenerada1"))
DIR_GT = Path(os.environ.get("DIR_GT", "../imagenes/respuestaEsperada"))


def facturas() -> list[str]:
    nombres = sorted(p.name.split("_")[0] for p in DIR_OCR.glob("*_res.json"))
    return [n for n in nombres if n not in EXCLUIDAS_ITEMS]


def procesar(nombre: str, modo: str = "x"):
    res = cargar_ocr_json(DIR_OCR / f"{nombre}_tabla_res.json")
    filas = agrupar_filas(res)
    doc, avisos = desde_filas(filas, modo=modo)
    truth, _ = desde_ground_truth(DIR_GT / f"{nombre}.json")
    return filas, doc, avisos, truth


def resumen(modo: str = "x") -> pd.DataFrame:
    out = []
    for n in facturas():
        filas, doc, avisos, truth = procesar(n, modo=modo)
        r = comparar_items(doc, truth)
        v = validar_aritmetica(doc)
        fila = {"factura": n, "filas_ocr": len(filas),
                "items": len(doc["items"]), "esperados": len(truth["items"]),
                "conflictos": len(avisos["conflictos"]),
                "arit_ok": int((v.estado == "ok").sum()) if len(v) else 0}
        f = r["resumen"].set_index("campo")
        for c in CAMPOS_ESENCIALES:
            fila[c] = f.loc[c, "exacta"] if c in f.index else None
            laxa = f.loc[c, "laxa"] if c in f.index else None
            if laxa is not None and laxa != fila[c]:
                fila[c] = f"{fila[c]} ({laxa})"
        out.append(fila)
    return pd.DataFrame(out)


def detalle(nombre: str, modo: str = "x") -> None:
    filas, doc, avisos, truth = procesar(nombre, modo=modo)
    print(f"=== {nombre} — {len(filas)} filas de OCR, modo '{modo}'\n")
    print("encabezado leído:", avisos["encabezado"])
    print("mapeo:", avisos["mapa"])
    if avisos["sin_nombre"]:
        print("columnas sin nombre:", avisos["sin_nombre"])
    print(f"\nitems: {len(doc['items'])} (ground truth: {len(truth['items'])})")

    if avisos["conflictos"]:
        print(f"\nfilas con celdas dobles (síntoma de fusión): "
              f"{len(avisos['conflictos'])}")
        for c in avisos["conflictos"][:8]:
            print("  fila", c["fila"], "->", "; ".join(c["detalle"]))

    if avisos["descartadas"]:
        print(f"\ndescartadas ({len(avisos['descartadas'])}):")
        for d in avisos["descartadas"][:8]:
            print("  ", d[:140])

    v = validar_aritmetica(doc)
    if len(v):
        print("\naritmética:", dict(v.estado.value_counts()))

    r = comparar_items(doc, truth)
    print("\nprecisión por campo:")
    print(r["resumen"].to_string(index=False))
    print(f"emparejados {r['emparejados']} | sin emparejar: "
          f"pipeline {r['sin_emparejar_pipeline']}, truth {r['sin_emparejar_truth']}")
    if len(r["errores"]):
        print(f"\nprimeros errores ({len(r['errores'])} en total):")
        print(r["errores"].head(12).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    if len(sys.argv) > 1:
        detalle(sys.argv[1], modo=sys.argv[2] if len(sys.argv) > 2 else "x")
    else:
        print("modo x (asignación por superposición con la franja de columna)")
        print(resumen("x").to_string(index=False))
        print("\nmodo posicional (línea de base)")
        print(resumen("posicional").to_string(index=False))
