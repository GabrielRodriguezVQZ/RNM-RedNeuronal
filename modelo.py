"""
modelo.py — Entrena la red neuronal (Keras) y los modelos de comparación.

Qué hace este archivo:
  1. Carga el dataset (RNM + variables) desde datos.py.
  2. Separa los datos por TIEMPO: entrenamiento (<= 2040) y prueba (> 2040).
  3. Entrena la red neuronal 10 veces con semillas distintas y promedia.
  4. Entrena modelos de comparación (regresión lineal, Ridge, Random Forest, ETS).
  5. Calcula las métricas de error y guarda todo en data/ para la app.

Uso:  python modelo.py
"""
import os, json, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # oculta mensajes informativos de TensorFlow
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datos import construir_dataset, FEATURES

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
SEED = 42            # semilla base: hace que los resultados sean reproducibles
ANIO_CORTE = 2040    # hasta este año se entrena; después se evalúa (prueba)
RUTA = "data/PPED-AreaSexoEdadNac-2018-2070.xlsx"


def fijar_semillas(s=SEED):
    """Fija las semillas aleatorias de Python, NumPy y TensorFlow."""
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


# ----------------------------------------------------------------------------
# ARQUITECTURA DE LA RED NEURONAL
# ----------------------------------------------------------------------------
def red(n_in):
    """Crea la red: n_in entradas -> 16 neuronas -> 8 neuronas -> 1 salida (el RNM)."""
    return tf.keras.Sequential([
        tf.keras.layers.Input((n_in,)),                       # entrada: una columna por variable
        # Capa oculta 1: 16 neuronas con ReLU. L2 penaliza pesos grandes (evita sobreajuste).
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        # Dropout: durante el entrenamiento apaga al azar el 10 % de las neuronas.
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation="relu"),          # Capa oculta 2: 8 neuronas
        tf.keras.layers.Dense(1),                             # Salida: un solo número (RNM)
    ])


def metricas(y, p):
    """Calcula las tres métricas de error entre valores reales (y) y predichos (p)."""
    return {
        "MAE": mean_absolute_error(y, p),                        # error absoluto medio
        "RMSE": mean_squared_error(y, p) ** 0.5,                 # raíz del error cuadrático medio
        "MAPE_%": 100 * mean_absolute_percentage_error(y, p),    # error porcentual medio
    }


def main():
    # ------------------------------------------------------------------
    # 1. Datos y partición temporal
    # ------------------------------------------------------------------
    d = construir_dataset(RUTA)
    # Convertimos el área en dos columnas 0/1 para que la red pueda usarla.
    # (Cabecera queda cuando ambas valen 0.)
    d["es_rural"] = (d.area == "Rural").astype(float)
    d["es_total"] = (d.area == "Total").astype(float)
    X_cols = FEATURES + ["es_rural", "es_total"]   # las 12 variables de entrada

    # Partición por TIEMPO (nunca aleatoria): el modelo no ve el futuro al entrenar.
    tr, te = d[d.anio <= ANIO_CORTE], d[d.anio > ANIO_CORTE]

    # Estandarizamos (media 0, desviación 1) usando SOLO datos de entrenamiento,
    # para no filtrar información de los años de prueba.
    sx, sy = StandardScaler().fit(tr[X_cols]), StandardScaler().fit(tr[["RNM"]])
    Xtr, Xte = sx.transform(tr[X_cols]), sx.transform(te[X_cols])
    ytr = sy.transform(tr[["RNM"]]).ravel()

    # ------------------------------------------------------------------
    # 2. Red neuronal: se entrenan N redes y se promedian sus predicciones.
    #    (Con pocos datos, cada red cambia según sus pesos iniciales; el promedio
    #     reduce esa variación y da un resultado más estable.)
    # ------------------------------------------------------------------
    pred_nn = np.zeros(len(d)); Xall = sx.transform(d[X_cols])
    N = 10
    for k in range(N):
        fijar_semillas(SEED + k)                     # una semilla distinta por red
        m = red(Xtr.shape[1])
        m.compile(tf.keras.optimizers.Adam(5e-3), "mse")   # Adam + error cuadrático medio
        # Early stopping: detiene el entrenamiento si la validación deja de mejorar
        # durante 60 épocas y recupera los mejores pesos.
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True)
        # validation_split=0.2: reserva el 20 % de los datos de entrenamiento para vigilar el sobreajuste.
        m.fit(Xtr, ytr, validation_split=0.2, epochs=800, batch_size=16, verbose=0, callbacks=[es], shuffle=True)
        # Predecimos todos los años y volvemos a la escala original del RNM.
        pred_nn += sy.inverse_transform(m.predict(Xall, verbose=0)).ravel() / N
    d["pred_RNN"] = pred_nn

    # ------------------------------------------------------------------
    # 3. Modelos de comparación (usan las mismas variables que la red)
    # ------------------------------------------------------------------
    modelos_base = {
        "Regresion_lineal": LinearRegression(),
        "Ridge": Ridge(1.0),                                              # regresión lineal regularizada
        "Random_Forest": RandomForestRegressor(300, random_state=SEED, min_samples_leaf=2),
    }
    for nombre, mod in modelos_base.items():
        mod.fit(Xtr, tr.RNM)
        d[f"pred_{nombre}"] = mod.predict(Xall)

    # ------------------------------------------------------------------
    # 4. ETS (suavizado exponencial): modelo clásico de series de tiempo.
    #    Solo usa el pasado de cada serie (no usa las demás variables).
    # ------------------------------------------------------------------
    d["pred_ETS"] = np.nan
    for area, g in d.groupby("area"):
        s = g[g.anio <= ANIO_CORTE].set_index("anio").RNM       # serie histórica de entrenamiento
        fit = ExponentialSmoothing(s.values, trend="add", damped_trend=True).fit()
        h = g.anio.max() - ANIO_CORTE                           # años a pronosticar
        d.loc[g.index, "pred_ETS"] = np.r_[fit.fittedvalues, fit.forecast(h)]

    # ------------------------------------------------------------------
    # 5. Métricas de error en entrenamiento y en prueba
    # ------------------------------------------------------------------
    modelos = [c for c in d.columns if c.startswith("pred_")]
    te = d[d.anio > ANIO_CORTE]; tr = d[d.anio <= ANIO_CORTE]
    filas = []
    for c in modelos:
        for conj, df in (("prueba", te), ("entrenamiento", tr)):
            filas.append({"modelo": c[5:], "conjunto": conj, **metricas(df.RNM, df[c])})
    res = pd.DataFrame(filas)

    # Evaluación extra: solo el Total en 2051-2070 (la extrapolación más lejana).
    extra = te[(te.area == "Total") & (te.anio > 2050)]
    for c in modelos:
        res = pd.concat([res, pd.DataFrame([{"modelo": c[5:], "conjunto": "prueba Total 2051-2070", **metricas(extra.RNM, extra[c])}])])

    # ------------------------------------------------------------------
    # 6. Guardado de resultados (los lee app.py)
    # ------------------------------------------------------------------
    d.to_csv("data/predicciones.csv", index=False)
    res.to_csv("data/metricas.csv", index=False)
    json.dump({"anio_corte": ANIO_CORTE, "semillas": N, "features": X_cols}, open("data/config.json", "w"))

    print(res.pivot(index="modelo", columns="conjunto", values="MAPE_%").round(2))
    print(res[res.conjunto == "prueba"].round(3).sort_values("RMSE"))


if __name__ == "__main__":
    main()