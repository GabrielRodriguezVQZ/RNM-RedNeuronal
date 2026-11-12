"""
datos.py — Lectura del Excel del DANE y cálculo del RNM.

Qué hace este archivo:
  1. Lee la hoja con la población por año, área, sexo y edad simple.
  2. Calcula el RNM (Relación Niños-Mujeres) para cada año y área.
  3. Construye las variables (features) que usará la red neuronal.

Uso:  python datos.py   ->  genera data/rnm_dataset.csv
"""
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
HOJA = "PobNacionalxÁreaSexoEdad"   # nombre de la hoja del Excel que tiene los datos

# El Excel llama a las áreas con nombres largos; aquí los dejamos más cortos.
AREAS = {
    "Total": "Total",
    "Cabecera": "Cabecera",
    "Centros Poblados y Rural Disperso": "Rural",
}

# El RNM se expresa "por 100 mujeres". Si tu materia lo pide por 1.000, cambia a 1000.
FACTOR = 100


# ----------------------------------------------------------------------------
# 1. LECTURA DEL EXCEL
# ----------------------------------------------------------------------------
def cargar_excel(ruta):
    """Lee la hoja del Excel y deja solo las filas de datos (sin títulos ni notas).

    Devuelve:
      raw : tabla con una fila por (año, área) y las columnas de población
      idx : diccionario {nombre de columna -> posición}, para ubicar columnas por nombre
    """
    # header=None: no usamos ninguna fila como encabezado automático.
    # skiprows=7: saltamos los títulos que están antes de la tabla.
    raw = pd.read_excel(ruta, sheet_name=HOJA, header=None, skiprows=7)

    # La fila 1 (segunda) contiene los nombres reales, p. ej. "Hombres 0 años".
    cols = [str(c).strip() for c in raw.iloc[1]]

    # Las dos primeras filas son encabezados; los datos empiezan en la tercera.
    raw = raw.iloc[2:].copy()
    raw.columns = [f"c{i}" for i in range(raw.shape[1])]  # nombres temporales c0, c1, ...

    # Nos quedamos solo con las filas de "Total Nacional" (descarta notas al pie).
    raw = raw[raw["c0"] == "Total Nacional"].copy()

    raw["anio"] = raw["c1"].astype(int)   # la columna 1 es el año
    raw["area"] = raw["c2"].map(AREAS)    # la columna 2 es el área (Total/Cabecera/Rural)

    # Diccionario para encontrar la posición de una columna a partir de su nombre.
    idx = {n: i for i, n in enumerate(cols)}
    return raw, idx


def poblacion_larga(ruta):
    """Devuelve una tabla ordenada con la población por edad simple.

    Columnas resultantes:
      anio, area,
      h0..h100 -> hombres de 0 a 100 años (100 = "100 y más")
      m0..m100 -> mujeres de 0 a 100 años
    """
    raw, idx = cargar_excel(ruta)
    out = raw[["anio", "area"]].reset_index(drop=True)

    datos = {}
    for e in range(101):  # edades 0, 1, 2, ..., 100
        # Así se escriben los nombres en el Excel: "1 año", "2 años", "100 años y más"
        suf = "100 años y más" if e == 100 else (f"{e} año" if e == 1 else f"{e} años")
        for pref, k in (("Hombres", "h"), ("Mujeres", "m")):
            nombre = f"{pref} {suf}"
            # Buscamos la columna cuyo nombre coincide (" ".join(split()) limpia espacios dobles).
            col = [i for n, i in idx.items() if " ".join(n.split()) == nombre][0]
            datos[f"{k}{e}"] = pd.to_numeric(raw.iloc[:, col].values)

    return pd.concat([out, pd.DataFrame(datos)], axis=1)


# ----------------------------------------------------------------------------
# 2. CÁLCULO DEL RNM Y DE LAS VARIABLES DEL MODELO
# ----------------------------------------------------------------------------
def construir_dataset(ruta):
    """Calcula el RNM y las variables de entrada de la red neuronal."""
    p = poblacion_larga(ruta)

    # Niños de 0 a 4 años = hombres + mujeres de 0, 1, 2, 3 y 4 años.
    ninos = p[[f"h{e}" for e in range(5)] + [f"m{e}" for e in range(5)]].sum(axis=1)

    # Mujeres en edad fértil = mujeres de 15 a 49 años.
    mf = p[[f"m{e}" for e in range(15, 50)]].sum(axis=1)

    d = p[["anio", "area"]].copy()
    d["ninos_0_4"] = ninos
    d["mujeres_15_49"] = mf

    # ---- VARIABLE OBJETIVO: lo que la red debe aprender a predecir ----
    d["RNM"] = FACTOR * ninos / mf

    # ---- VARIABLES DE ENTRADA ----
    # IMPORTANTE: ninguna usa a los niños de 0-4 años. Si las usáramos, la red
    # solo "copiaría" el numerador del RNM (eso se llama fuga de información).

    # (a) Proporción de las mujeres fértiles que hay en cada grupo quinquenal
    #     (15-19, 20-24, ..., 45-49). Cada edad tiene una fecundidad distinta.
    for a in range(15, 50, 5):
        d[f"m_{a}_{a+4}"] = p[[f"m{e}" for e in range(a, a + 5)]].sum(axis=1) / mf

    # (b) Niñas de 5-9 y 10-14 años, relativas a las mujeres fértiles (futuras madres).
    d["m_5_9_rel"] = p[[f"m{e}" for e in range(5, 10)]].sum(axis=1) / mf
    d["m_10_14_rel"] = p[[f"m{e}" for e in range(10, 15)]].sum(axis=1) / mf

    # (c) Edad media de las mujeres de 15 a 49 años (promedio ponderado por población).
    d["edad_media_mf"] = sum(p[f"m{e}"] * e for e in range(15, 50)) / mf

    return d.sort_values(["area", "anio"]).reset_index(drop=True)


# Lista de columnas que entran a la red (se importa desde modelo.py).
FEATURES = [f"m_{a}_{a+4}" for a in range(15, 50, 5)] + ["m_5_9_rel", "m_10_14_rel", "edad_media_mf"]


# ----------------------------------------------------------------------------
# 3. EJECUCIÓN DIRECTA: python datos.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    d = construir_dataset("data/PPED-AreaSexoEdadNac-2018-2070.xlsx")
    d.to_csv("data/rnm_dataset.csv", index=False)       # guardamos el resultado
    print(d.groupby("area").anio.agg(["count", "min", "max"]))   # años disponibles por área
    print(d[d.area == "Total"][["anio", "ninos_0_4", "mujeres_15_49", "RNM"]].iloc[::8])