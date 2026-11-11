"""Carga del Excel DANE (PPED) y cálculo del RNM."""
import pandas as pd
import numpy as np

HOJA = "PobNacionalxÁreaSexoEdad"
AREAS = {"Total": "Total", "Cabecera": "Cabecera", "Centros Poblados y Rural Disperso": "Rural"}
FACTOR = 100  # RNM por 100 mujeres (cambiar a 1000 si tu materia lo pide)


def cargar_excel(ruta):
    raw = pd.read_excel(ruta, sheet_name=HOJA, header=None, skiprows=7)
    cols = [str(c).strip() for c in raw.iloc[1]]
    raw = raw.iloc[2:].copy()
    raw.columns = [f"c{i}" for i in range(raw.shape[1])]
    raw = raw[raw["c0"] == "Total Nacional"].copy()
    raw["anio"] = raw["c1"].astype(int)
    raw["area"] = raw["c2"].map(AREAS)
    idx = {n: i for i, n in enumerate(cols)}
    return raw, idx


def poblacion_larga(ruta):
    """Devuelve DataFrame con anio, area y columnas h0..h100, m0..m100 (edad simple)."""
    raw, idx = cargar_excel(ruta)
    out = raw[["anio", "area"]].reset_index(drop=True)
    datos = {}
    for e in range(101):
        suf = "100 años y más" if e == 100 else (f"{e} año" if e == 1 else f"{e} años")
        for pref, k in (("Hombres", "h"), ("Mujeres", "m")):
            nombre = f"{pref} {suf}"
            col = [i for n, i in idx.items() if " ".join(n.split()) == nombre][0]
            datos[f"{k}{e}"] = pd.to_numeric(raw.iloc[:, col].values)
    return pd.concat([out, pd.DataFrame(datos)], axis=1)


def construir_dataset(ruta):
    p = poblacion_larga(ruta)
    ninos = p[[f"h{e}" for e in range(5)] + [f"m{e}" for e in range(5)]].sum(axis=1)
    mf = p[[f"m{e}" for e in range(15, 50)]].sum(axis=1)
    d = p[["anio", "area"]].copy()
    d["ninos_0_4"] = ninos
    d["mujeres_15_49"] = mf
    d["RNM"] = FACTOR * ninos / mf
    # Características demográficas (NO incluyen a los niños 0-4 → sin fuga de información)
    for a in range(15, 50, 5):
        d[f"m_{a}_{a+4}"] = p[[f"m{e}" for e in range(a, a + 5)]].sum(axis=1) / mf
    d["m_5_9_rel"] = p[[f"m{e}" for e in range(5, 10)]].sum(axis=1) / mf
    d["m_10_14_rel"] = p[[f"m{e}" for e in range(10, 15)]].sum(axis=1) / mf
    d["edad_media_mf"] = sum(p[f"m{e}"] * e for e in range(15, 50)) / mf
    return d.sort_values(["area", "anio"]).reset_index(drop=True)


FEATURES = [f"m_{a}_{a+4}" for a in range(15, 50, 5)] + ["m_5_9_rel", "m_10_14_rel", "edad_media_mf"]

if __name__ == "__main__":
    d = construir_dataset("data/PPED-AreaSexoEdadNac-2018-2070.xlsx")
    d.to_csv("data/rnm_dataset.csv", index=False)
    print(d.groupby("area").anio.agg(["count", "min", "max"]))
    print(d[d.area == "Total"][["anio", "ninos_0_4", "mujeres_15_49", "RNM"]].iloc[::8])
