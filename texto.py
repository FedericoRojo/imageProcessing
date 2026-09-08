"""texto.py — primitivas de comparación de texto.

Acá vive lo que no sabe nada del dominio: ni de facturas, ni de columnas, ni
de OCR. Sólo cómo dejar dos strings en una forma en la que compararlos
signifique algo.

Es el módulo más abajo de todos: no importa nada del proyecto, y lo importan
tanto la capa de encabezados como la de normalización de valores y la de
comparación contra el ground truth.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["sin_acentos", "clave"]


def sin_acentos(s: str) -> str:
    """Saca los diacríticos: 'Código' → 'Codigo', 'Descripción' → 'Descripcion'.

    Descompone cada carácter en su forma base + marca (NFD) y tira las marcas
    (categoría Unicode 'Mn' = Mark, nonspacing). Es más robusto que una tabla
    de reemplazos á→a: funciona con la diéresis, la tilde de la ñ y cualquier
    otro diacrítico que aparezca, incluidos los que el OCR inventa.
    """
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def clave(k) -> str:
    """Clave comparable: minúsculas, sin acentos, sin separadores.

        'Cód. Barras'  → 'codbarras'
        '% Bonif.'     → 'bonif'
        'P. Unit.'     → 'punit'
        'PRECIO UNIT.' → 'preciounit'

    Sin esta normalización, cada tabla de alias tendría que enumerar todas las
    variantes de mayúsculas, acentos y puntuación de cada nombre. Con ella,
    cada nombre se escribe una sola vez y en una sola forma.

    OJO: borra el '%'. Si un criterio depende de ese signo — y hay uno, en
    `puntaje_columna` — tiene que mirar el texto crudo, no la clave.
    """
    return re.sub(r"[^a-z0-9]", "", sin_acentos(str(k)).lower())
