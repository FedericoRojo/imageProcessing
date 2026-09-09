# tesis-facturas

Extracción automática de la tabla de items de fotos de facturas argentinas.

```
imagen original
   ↓  LLM de visión   →  bounding box de la tabla de items
imagen recortada + ampliada
   ↓  PaddleOCR       →  texto + coordenadas
   ↓  agrupado por y  →  filas
   ↓  cortes de columna → items nombrados (código, descripción, cantidad, precio, importe)
   ↓  cantidad × precio ≈ importe  →  validación sin ground truth
```

La idea es que el LLM haga sólo una cosa (ubicar la tabla) y que el OCR trabaje
sobre una imagen limpia, sin logo, datos del proveedor ni recuadro de totales.

## Estructura

Un módulo por pregunta. Los dos últimos pasos, por ejemplo, son dos preguntas
distintas y viven en dos módulos distintos: **cómo se llama** cada columna es
vocabulario y no mira una sola coordenada; **dónde empieza** es geometría y no
sabe de facturas.

| Archivo | Qué es |
|---|---|
| `pipeline.py` | **Fachada y orquestación.** Encadena los pasos (`preparar_tabla`, `procesar`) y re-exporta todo, para que el notebook toque un módulo solo. No tiene lógica propia. |
| `vision.py` | **Dónde está la tabla**: imagen → LLM de visión → bounding box. Lo único que habla con la API. |
| `imagen.py` | **Los píxeles antes del OCR**: orientación (EXIF), enderezado, recorte + zoom + borde. Dueño de `recortes/`. |
| `ocr.py` | PaddleOCR, y nada más. Lo único que importa `paddleocr`. |
| `filas.py` | **Qué cajas son el mismo renglón**: inclinación residual, paso entre filas, agrupado. Geometría pura, sin paddle. |
| `esquema.py` | **Dónde empieza** cada columna: la grilla se descubre de los datos, no de los títulos. Y el esquema canónico del documento. |
| `encabezado.py` | **Cómo se llama** cada columna: `ARTICULO`→`codigo`, `CANT.`→`cantidad`. Sin coordenadas. |
| `texto.py` | Primitivas de comparación de strings. No sabe de nada. |
| `diagnostico.py` | Herramientas para mirar qué pasó cuando algo sale mal. Nada de esto corre en el camino normal. |
| `banco.py` | Medición contra ground truth, replayeando salidas de OCR guardadas. |
| `test_estandarizar.py` | Pruebas del paso 5 con filas sintéticas. |
| `notebooks/factura_a_tabla.ipynb` | Notebook flaco: orquesta y muestra resultados. |
| `requirements.txt` | Dependencias. |
| `data/` | Facturas de prueba y salidas de OCR guardadas (ignoradas por git). |
| `recortes/` | Recortes que devuelve el paso 1, para revisar o descargar (ignorada por git). |

`esquema.py`, `encabezado.py`, `texto.py` y `filas.py` no importan `paddleocr` ni
`openai` a propósito: del agrupado en adelante, todo se prueba y se mide sin GPU
y sin API key, replayeando salidas de OCR guardadas. `ocr.py` importa paddle
adentro de `get_ocr`, así que ni siquiera él lo arrastra al importarse.

Las dependencias van en una sola dirección: `vision`, `imagen`, `ocr` y `filas`
no importan nada del repo; `diagnostico` usa a los tres primeros; `pipeline` los
usa a todos. No hay ciclos.

## Flujo de trabajo

El notebook **no** contiene lógica: hace `git pull`, importa los módulos y muestra
resultados. Los cambios se hacen en los `.py`, se pushean, y en Colab alcanza
con volver a correr la primera celda.

```
editar los .py  →  commit + push  →  en Colab: !git pull + correr la celda de import
```

Sin descargar ni volver a subir el `.ipynb`. La celda de import recarga los
módulos en orden de dependencia y con `pipeline` último; sin eso, un cambio en
`esquema.py` no se vería aunque el `git pull` lo haya traído, y `pipeline` —que
es una fachada— se quedaría con los nombres viejos.

## Correr en Colab

1. Abrir el notebook desde GitHub: `Archivo → Abrir cuaderno → GitHub`.
2. Cargar `NVIDIA_API_KEY` en los secrets de Colab (ícono de la llave).
3. Correr desde la primera celda.

## Correr local

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export NVIDIA_API_KEY=...                        # Windows: set NVIDIA_API_KEY=...
```

```python
from pipeline import procesar, estandarizar, items_a_dataframe, imprimir_avisos

img_tabla, res, filas, df = procesar("data/factura2.jpeg")   # pasos 1 a 4
doc, avisos = estandarizar(filas)                            # paso 5
items_a_dataframe(doc)          # las 9 columnas canónicas
imprimir_avisos(avisos)         # la grilla que se usó y lo que se descartó
```

`procesar` llega hasta las filas crudas; el paso 5 va aparte porque es el único
que se puede correr sin imagen, replayeando una salida de OCR guardada.

El recorte del paso 1 queda en `recortes/factura2_tabla.png`. En Colab,
`pipeline.descargar_recortes()` baja todos los recortes de la sesión (uno solo
como PNG, varios como `.zip`).

## Antes de commitear el notebook

Borrar los outputs (`Editar → Borrar todos los resultados` en Colab). Un notebook
con los resultados del OCR embebidos pesa 20x más y ensucia todos los diffs.
