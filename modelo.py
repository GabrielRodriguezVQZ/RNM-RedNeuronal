"""Entrena la red neuronal (Keras) y los baselines; guarda predicciones y métricas.
Split temporal: entrenamiento <= ANIO_CORTE, prueba > ANIO_CORTE."""
import os, json, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datos import construir_dataset, FEATURES

SEED, ANIO_CORTE = 42, 2040
RUTA = "data/PPED-AreaSexoEdadNac-2018-2070.xlsx"


def fijar_semillas(s=SEED):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def red(n_in):
    return tf.keras.Sequential([
        tf.keras.layers.Input((n_in,)),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),
    ])


def metricas(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "MAPE_%": 100 * mean_absolute_percentage_error(y, p)}


def main():
    d = construir_dataset(RUTA)
    d["es_rural"] = (d.area == "Rural").astype(float)
    d["es_total"] = (d.area == "Total").astype(float)
    X_cols = FEATURES + ["es_rural", "es_total"]
    tr, te = d[d.anio <= ANIO_CORTE], d[d.anio > ANIO_CORTE]
    sx, sy = StandardScaler().fit(tr[X_cols]), StandardScaler().fit(tr[["RNM"]])
    Xtr, Xte = sx.transform(tr[X_cols]), sx.transform(te[X_cols])
    ytr = sy.transform(tr[["RNM"]]).ravel()

    # --- Red neuronal: promedio de 10 semillas (reduce varianza) ---
    pred_nn = np.zeros(len(d)); Xall = sx.transform(d[X_cols])
    N = 10
    for k in range(N):
        fijar_semillas(SEED + k)
        m = red(Xtr.shape[1]); m.compile(tf.keras.optimizers.Adam(5e-3), "mse")
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True)
        m.fit(Xtr, ytr, validation_split=0.2, epochs=800, batch_size=16, verbose=0, callbacks=[es], shuffle=True)
        pred_nn += sy.inverse_transform(m.predict(Xall, verbose=0)).ravel() / N
    d["pred_RNN"] = pred_nn

    # --- Baselines con las mismas variables ---
    for nombre, mod in {"Regresion_lineal": LinearRegression(), "Ridge": Ridge(1.0),
                        "Random_Forest": RandomForestRegressor(300, random_state=SEED, min_samples_leaf=2)}.items():
        mod.fit(Xtr, tr.RNM); d[f"pred_{nombre}"] = mod.predict(Xall)

    # --- Baseline de series de tiempo (solo usa el pasado de cada serie) ---
    d["pred_ETS"] = np.nan
    for area, g in d.groupby("area"):
        s = g[g.anio <= ANIO_CORTE].set_index("anio").RNM
        fit = ExponentialSmoothing(s.values, trend="add", damped_trend=True).fit()
        h = g.anio.max() - ANIO_CORTE
        d.loc[g.index, "pred_ETS"] = np.r_[fit.fittedvalues, fit.forecast(h)]

    modelos = [c for c in d.columns if c.startswith("pred_")]
    te = d[d.anio > ANIO_CORTE]; tr = d[d.anio <= ANIO_CORTE]
    filas = []
    for c in modelos:
        for conj, df in (("prueba", te), ("entrenamiento", tr)):
            filas.append({"modelo": c[5:], "conjunto": conj, **metricas(df.RNM, df[c])})
    res = pd.DataFrame(filas)
    # prueba solo en el horizonte común (áreas hasta 2050) y solo Total hasta 2070
    extra = te[(te.area == "Total") & (te.anio > 2050)]
    for c in modelos:
        res = pd.concat([res, pd.DataFrame([{"modelo": c[5:], "conjunto": "prueba Total 2051-2070", **metricas(extra.RNM, extra[c])}])])
    d.to_csv("data/predicciones.csv", index=False)
    res.to_csv("data/metricas.csv", index=False)
    json.dump({"anio_corte": ANIO_CORTE, "semillas": N, "features": X_cols}, open("data/config.json", "w"))
    print(res.pivot(index="modelo", columns="conjunto", values="MAPE_%").round(2))
    print(res[res.conjunto == "prueba"].round(3).sort_values("RMSE"))

if __name__ == "__main__":
    main()
