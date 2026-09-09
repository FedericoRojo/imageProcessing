# tesis-facturas

Extracción automática de la tabla de items de fotos de facturas argentinas.

```
imagen original
   ↓  LLM de visión   →  bounding box de la tabla de items
imagen recortada + ampliada
   ↓  PaddleOCR       →  texto + coordenadas
   ↓  agrupado por y  →  filas / DataFrame
```

La idea es que el LLM haga sólo una cosa (ubicar la tabla) y que el OCR trabaje
sobre una imagen limpia, sin logo, datos del proveedor ni recuadro de totales.

## Estructura

| Archivo | Qué es |
|---|---|
| `pipeline.py` | Toda la lógica. Es lo único que se edita. |
| `notebooks/factura_a_tabla.ipynb` | Notebook flaco: orquesta y muestra resultados. |
| `requirements.txt` | Dependencias. |
| `data/` | Facturas de prueba (ignoradas por git). |
| `recortes/` | Recortes que devuelve el paso 1, para revisar o descargar (ignorada por git). |

## Flujo de trabajo

El notebook **no** contiene lógica: hace `git pull`, importa `pipeline` y muestra
resultados. Los cambios se hacen en `pipeline.py`, se pushean, y en Colab alcanza
con volver a correr la primera celda.

```
editar pipeline.py  →  commit + push  →  en Colab: !git pull + Runtime > Restart
```

Sin descargar ni volver a subir el `.ipynb`.

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
from pipeline import procesar
img_tabla, res, filas, df = procesar("data/factura2.jpeg")
```

El recorte del paso 1 queda en `recortes/factura2_tabla.png`. En Colab,
`pipeline.descargar_recortes()` baja todos los recortes de la sesión (uno solo
como PNG, varios como `.zip`).

## Antes de commitear el notebook

Borrar los outputs (`Editar → Borrar todos los resultados` en Colab). Un notebook
con los resultados del OCR embebidos pesa 20x más y ensucia todos los diffs.
