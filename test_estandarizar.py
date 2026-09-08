"""Pruebas de la capa de estandarización con filas sintéticas.

Por qué sintéticas: hacen falta cajas con coordenadas reales para probar la
asignación por x, y correr PaddleOCR + la API para cada cambio es lento y caro.
Estas filas reproducen a mano la geometría de los tres formatos que ya
conocemos, incluidos los casos que sabemos que rompen: columnas vacías en todas
las filas, descripciones que el OCR parte en pedazos, líneas que no son items y
ruido arriba del encabezado.

Cuando existan los .json de OCR guardados, esto se reemplaza por el corpus real
sin tocar el código que se está probando.

    python3 test_estandarizar.py
"""

import sys

from esquema import (CAMPOS_ITEM, comparar_items, desde_filas, desde_ground_truth,
                     elegir_encabezado, estandarizar, mapear_encabezado,
                     validar_aritmetica)

FALLOS = []


def check(nombre, condicion, detalle=""):
    if condicion:
        print(f"  ok   {nombre}")
    else:
        print(f"  FALLA {nombre}  {detalle}")
        FALLOS.append(nombre)


def caja(txt, x0, x1, y, alto=22):
    return {"txt": txt, "x": x0, "x0": float(x0), "x1": float(x1),
            "y": float(y), "alto": float(alto), "score": 0.99}


def fila(y, celdas):
    return [caja(t, a, b, y) for t, a, b in celdas]


# --------------------------------------------------------------------------
# factura2 — ARTICULO / CANT. / DESCRIPCION / PRECIO / % DTO / IMPORTE / IVA
# Las columnas % DTO e IVA están vacías en las 44 filas: es el caso que rompe
# el modo posicional.
# --------------------------------------------------------------------------
F2_ENCABEZADO = [("ARTICULO", 30, 150), ("CANT.", 200, 280),
                 ("DESCRIPCION", 340, 560), ("PRECIO", 900, 1000),
                 ("% DTO", 1050, 1120), ("IMPORTE", 1250, 1380),
                 ("IVA", 1430, 1490)]

FILAS_F2 = [
    fila(100, F2_ENCABEZADO),
    fila(140, [("0412024", 30, 150), ("2", 250, 270),
               ("BASTIDOR ENTELADO BASIC FIME DE 18 X 24 CM.", 340, 820),
               ("5.564,23", 900, 1010), ("11.128,46", 1250, 1390)]),
    # el OCR partió la descripción en dos cajas
    fila(170, [("0720249", 30, 150), ("6", 250, 270),
               ("CINTA P/EMBALAJE STIKO", 340, 640),
               ("TRANSPARENTE 48 X 100 MT", 650, 900),
               ("3.138,91", 900, 1010), ("18.833,46", 1250, 1390)]),
    fila(200, [("2840810", 30, 150), ("1", 250, 270),
               ("CINTA P/REGALO PACKERS DE 20 MM X 50 MTS LISO O", 340, 830),
               ("2.278,15", 900, 1010), ("2.278,15", 1250, 1390)]),
    # fila de pie: no es un item
    fila(240, [("TRANSPORTE", 900, 1050), ("391.868,27", 1250, 1390)]),
]


def prueba_factura2():
    print("\nfactura2 — columnas vacías en todas las filas")
    mapa, _ = mapear_encabezado(FILAS_F2[0])
    nombres = {FILAS_F2[0][i]["txt"]: c for i, c in mapa.items()}
    check("ARTICULO -> codigo", nombres.get("ARTICULO") == "codigo", nombres)
    check("CANT. -> cantidad", nombres.get("CANT.") == "cantidad", nombres)
    check("DESCRIPCION -> descripcion", nombres.get("DESCRIPCION") == "descripcion")
    check("PRECIO -> precio_unitario", nombres.get("PRECIO") == "precio_unitario")
    check("% DTO -> descuento_pct", nombres.get("% DTO") == "descuento_pct", nombres)
    check("IMPORTE -> importe", nombres.get("IMPORTE") == "importe")

    doc, av = desde_filas(FILAS_F2)
    check("3 items, la de TRANSPORTE descartada", len(doc["items"]) == 3,
          f'{len(doc["items"])} items, descartadas={av["descartadas"]}')
    it = doc["items"][0]
    check("importe en su columna, no en % DTO", it["importe"] == 11128.46,
          f'importe={it["importe"]} descuento={it["descuento_pct"]}')
    check("descuento vacío queda None", it["descuento_pct"] is None)
    check("precio parseado del formato AR", it["precio_unitario"] == 5564.23)
    check("descripción partida se reune",
          doc["items"][1]["descripcion"] ==
          "CINTA P/EMBALAJE STIKO TRANSPARENTE 48 X 100 MT",
          doc["items"][1]["descripcion"])

    # la línea de base posicional: acá es donde se rompe
    doc_pos, _ = desde_filas(FILAS_F2, modo="posicional")
    mal = doc_pos["items"][0]["importe"] != 11128.46
    check("el modo posicional falla, como se esperaba", mal,
          f'posicional dio importe={doc_pos["items"][0]["importe"]}')

    v = validar_aritmetica(doc)
    check("aritmética ok en las 3", (v.estado == "ok").sum() == 3,
          v.to_string(index=False))


# --------------------------------------------------------------------------
# factura5 — el código está en la TERCERA columna, y hay una línea DESPACHO
# que no es un item.
# --------------------------------------------------------------------------
F5_ENCABEZADO = [("Cantidad", 30, 140), ("U.Medid.", 170, 270),
                 ("Código", 300, 400), ("Descripción", 430, 620),
                 ("% Desc.", 900, 990), ("Precio Unitario", 1050, 1250),
                 ("Total", 1350, 1450)]

FILAS_F5 = [
    fila(100, F5_ENCABEZADO),
    fila(140, [("4,00", 30, 140), ("U", 200, 220), ("A09492", 300, 400),
               ("ALB. NEW 15X22X200 LISO", 430, 780), ("0,00", 900, 990),
               ("23.317,4667", 1050, 1260), ("93.269,87", 1350, 1460)]),
    fila(170, [("2,00", 30, 140), ("U", 200, 220), ("A09345", 300, 400),
               ("ALB.MINI MAX 15X22X100 LISO", 430, 800), ("0,00", 900, 990),
               ("11.521,0667", 1050, 1260), ("23.042,13", 1350, 1460)]),
    fila(200, [("5,00", 30, 140), ("U", 200, 220), ("P63082", 300, 400),
               ("PILA VINNIC CR 1220", 430, 720), ("0,00", 900, 990),
               ("450,00", 1050, 1260), ("2.250,00", 1350, 1460)]),
    # línea que no es item: sólo descripción
    fila(230, [("DESPACHO: S/FACT102141", 430, 800)]),
]


def prueba_factura5():
    print("\nfactura5 — código en la tercera columna, línea DESPACHO")
    mapa, _ = mapear_encabezado(FILAS_F5[0])
    nombres = {FILAS_F5[0][i]["txt"]: c for i, c in mapa.items()}
    check("Código -> codigo (3ra columna)", nombres.get("Código") == "codigo", nombres)
    check("U.Medid. -> unidad", nombres.get("U.Medid.") == "unidad", nombres)
    check("% Desc. -> descuento_pct, no descripcion",
          nombres.get("% Desc.") == "descuento_pct", nombres)
    check("Precio Unitario -> precio_unitario",
          nombres.get("Precio Unitario") == "precio_unitario", nombres)
    check("Total -> importe", nombres.get("Total") == "importe", nombres)

    doc, av = desde_filas(FILAS_F5)
    check("3 items", len(doc["items"]) == 3, f'{len(doc["items"])}')
    check("DESPACHO descartada y reportada",
          any("DESPACHO" in d for d in av["descartadas"]), av["descartadas"])
    check("decimales largos", doc["items"][0]["precio_unitario"] == 23317.4667,
          doc["items"][0]["precio_unitario"])


# --------------------------------------------------------------------------
# factura1 — Neto Unit. vs Neto Total, que se pelean por el mismo campo.
# Y ruido arriba del encabezado.
# --------------------------------------------------------------------------
F1_ENCABEZADO = [("Codigo", 30, 130), ("Cod. Barras", 160, 320),
                 ("Producto", 350, 520), ("Cant.", 800, 880),
                 ("P. Unitario", 920, 1080), ("% Bonif", 1120, 1240),
                 ("Neto Unit.", 1280, 1420), ("Neto Total", 1460, 1600)]

FILAS_F1 = [
    fila(60, [("Original", 30, 140)]),               # ruido arriba
    fila(100, F1_ENCABEZADO),
    fila(140, [("1335487", 30, 130), ("1335487", 160, 320),
               ("BRILLANTINA X 9 E/BL C.PEN 514023", 350, 780),
               ("2,00", 800, 880), ("1.680,20", 920, 1080), ("15,00", 1120, 1240),
               ("1.428,17", 1280, 1420), ("2.856,34", 1460, 1600)]),
]


def prueba_factura1():
    print("\nfactura1 — ruido arriba, Neto Unit. vs Neto Total")
    idx, mapa, _ = elegir_encabezado(FILAS_F1)
    check("saltea la fila de ruido", idx == 1, f"idx={idx}")
    nombres = {FILAS_F1[1][i]["txt"]: c for i, c in mapa.items()}
    check("P. Unitario -> precio_unitario",
          nombres.get("P. Unitario") == "precio_unitario", nombres)
    check("Neto Total -> importe", nombres.get("Neto Total") == "importe", nombres)
    check("Cod. Barras -> codigo_barras",
          nombres.get("Cod. Barras") == "codigo_barras", nombres)
    check("Codigo -> codigo", nombres.get("Codigo") == "codigo", nombres)
    check("% Bonif -> descuento_pct", nombres.get("% Bonif") == "descuento_pct",
          nombres)

    doc, _ = desde_filas(FILAS_F1)
    it = doc["items"][0]
    check("importe = neto total", it["importe"] == 2856.34, it["importe"])
    v = validar_aritmetica(doc)
    check("aritmética con bonificación cierra", v.estado.iloc[0] == "ok",
          v.to_string(index=False))


# --------------------------------------------------------------------------
# Comparación contra el ground truth real de factura2
# --------------------------------------------------------------------------

def prueba_comparacion():
    print("\ncomparación contra el ground truth de factura2")
    doc_p, _ = desde_filas(FILAS_F2)
    doc_t, _ = desde_ground_truth("../imagenes/factura2.json")
    r = comparar_items(doc_p, doc_t)
    print(r["resumen"].to_string(index=False))
    check("empareja las 3 por código", r["emparejados"] == 3, r["emparejados"])
    check("detecta los 41 items que faltan",
          r["sin_emparejar_truth"] == 41, r["sin_emparejar_truth"])
    perfectos = (r["resumen"].mal.sum() == 0 and r["resumen"].faltante.sum() == 0)
    check("sin errores de campo en las emparejadas", perfectos,
          r["errores"].to_string(index=False) if len(r["errores"]) else "")


def prueba_filas_fusionadas():
    """El problema abierto: dos items en la misma fila.

    No lo resuelve esta capa, pero tiene que DETECTARLO: dos códigos en la
    celda de código no es el OCR partiendo un texto, es un agrupado mal hecho.
    """
    print("\nfilas fusionadas — no se resuelve acá, pero tiene que gritar")
    # Con las filas sanas delante: la grilla se descubre de los datos, así que
    # una sola fila no alcanza para que haya grilla que contradecir.
    filas = FILAS_F2[:-1] + [
        fila(230, [("1203020", 30, 110), ("1203019", 115, 195),
                   ("20", 250, 270), ("20", 275, 295),
                   ("CARTULINA ESCOLAR LUMA VIOLETA 45 X 63 CM.", 340, 800),
                   ("376,86", 900, 1010), ("6.783,27", 1250, 1390)]),
    ]
    doc, av = desde_filas(filas)
    check("detecta el conflicto", len(av["conflictos"]) == 1, av["conflictos"])
    detalle = " ".join(av["conflictos"][0]["detalle"]) if av["conflictos"] else ""
    check("señala la columna de código", "codigo" in detalle, detalle)
    v = validar_aritmetica(doc)
    check("la aritmética también lo delata", v.estado.iloc[-1] == "falla",
          v.to_string(index=False))


def prueba_encabezado_ilegible():
    print("\nencabezado ilegible — el disparador del nivel 2")
    filas = [fila(100, [("XXXX", 30, 150), ("###", 200, 280)]),
             fila(140, [("0412024", 30, 150), ("2", 250, 270)])]
    idx, mapa, _ = elegir_encabezado(filas)
    check("no reconoce encabezado", idx is None, f"idx={idx} mapa={mapa}")
    doc, av = desde_filas(filas)
    check("avisa el error", "error" in av, av)
    doc, av = estandarizar(filas, usar_llm=False)
    check("sin LLM devuelve vacío, no basura", len(doc["items"]) == 0)


if __name__ == "__main__":
    prueba_factura2()
    prueba_factura5()
    prueba_factura1()
    prueba_comparacion()
    prueba_filas_fusionadas()
    prueba_encabezado_ilegible()
    print(f"\n{'TODO OK' if not FALLOS else str(len(FALLOS)) + ' FALLAS: ' + ', '.join(FALLOS)}")
    sys.exit(1 if FALLOS else 0)
