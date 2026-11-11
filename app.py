import json
import pandas as pd, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="RNM Colombia", page_icon="📈", layout="wide")
d = pd.read_csv("data/predicciones.csv"); m = pd.read_csv("data/metricas.csv")
cfg = json.load(open("data/config.json")); corte = cfg["anio_corte"]
NOMBRES = {"RNN": "Red neuronal (Keras)", "Regresion_lineal": "Regresión lineal", "Ridge": "Ridge",
           "Random_Forest": "Random Forest", "ETS": "Suavizado exponencial (ETS)"}

st.title("Relación Niños-Mujeres (RNM) en Colombia, 2018-2070")
st.caption("RNM = (población 0-4 años ÷ mujeres 15-49 años) × 100. Fuente: DANE, Proyecciones de población (PPED, act. julio 2025).")

with st.sidebar:
    st.header("Controles")
    area = st.selectbox("Área geográfica", ["Total", "Cabecera", "Rural"])
    modelos = st.multiselect("Modelos a comparar", list(NOMBRES), default=["RNN", "Regresion_lineal"], format_func=NOMBRES.get)
    st.info(f"Entrenamiento: 2018–{corte}\n\nPrueba: {corte+1} en adelante (la red no ve esos años).")

g = d[d.area == area]
t1, t2, t3, t4 = st.tabs(["Proyección", "Comparación de modelos", "Datos", "Metodología"])

with t1:
    fig = go.Figure()
    fig.add_scatter(x=g.anio, y=g.RNM, name="RNM oficial (calculado del DANE)", line=dict(color="#1f2937", width=3))
    for k in modelos:
        fig.add_scatter(x=g.anio, y=g[f"pred_{k}"], name=NOMBRES[k], line=dict(dash="dash"))
    fig.add_vrect(x0=corte + .5, x1=g.anio.max(), fillcolor="#f59e0b", opacity=.10, line_width=0,
                  annotation_text="Zona de prueba", annotation_position="top left")
    fig.update_layout(xaxis_title="Año", yaxis_title="RNM (niños 0-4 por 100 mujeres 15-49)", hovermode="x unified", height=480, legend=dict(orientation="h", y=-.2))
    st.plotly_chart(fig, use_container_width=True)
    anio = st.slider("Año", int(g.anio.min()), int(g.anio.max()), 2030)
    r = g[g.anio == anio].iloc[0]
    c = st.columns(4)
    c[0].metric("RNM oficial", f"{r.RNM:.2f}")
    c[1].metric("Red neuronal", f"{r.pred_RNN:.2f}", f"{r.pred_RNN - r.RNM:+.2f}")
    c[2].metric("Niños 0-4", f"{r.ninos_0_4:,.0f}")
    c[3].metric("Mujeres 15-49", f"{r.mujeres_15_49:,.0f}")

with t2:
    conj = st.radio("Conjunto de evaluación", ["prueba", "prueba Total 2051-2070", "entrenamiento"], horizontal=True)
    mm = m[m.conjunto == conj].copy(); mm["modelo"] = mm.modelo.map(NOMBRES)
    st.dataframe(mm.drop(columns="conjunto").sort_values("RMSE").round(3), hide_index=True, use_container_width=True)
    b = go.Figure(go.Bar(x=mm.sort_values("MAPE_%").modelo, y=mm.sort_values("MAPE_%")["MAPE_%"], marker_color="#2563eb"))
    b.update_layout(yaxis_title="MAPE (%) — menor es mejor", height=360)
    st.plotly_chart(b, use_container_width=True)
    st.caption("Si la regresión lineal iguala o supera a la red, se reporta tal cual: con ~100 observaciones una red no tiene por qué ganar.")

with t3:
    st.dataframe(g[["anio", "ninos_0_4", "mujeres_15_49", "RNM", "pred_RNN"]].round(3), hide_index=True, use_container_width=True)
    st.download_button("Descargar CSV", g.to_csv(index=False), f"rnm_{area}.csv")

with t4:
    st.markdown(f"""
**Datos.** Hoja *PobNacionalxÁreaSexoEdad* del PPED: población por año, área, sexo y edad simple. Por área solo hay datos 2018–2050; el Total llega a 2070.

**Variable objetivo.** RNM por 100 mujeres, calculado por año y área.

**Variables de entrada.** Estructura de las mujeres en edad fértil (distribución por grupos quinquenales 15-19…45-49), mujeres de 5-9 y 10-14 relativas, edad media de las mujeres 15-49 y el área. **No se usa la población 0-4**, para evitar fuga de información.

**Red.** Keras: Dense(16, ReLU, L2) → Dropout(0,1) → Dense(8) → Dense(1). Early stopping, promedio de {cfg['semillas']} semillas.

**Validación.** Split temporal (entrena ≤ {corte}). Comparada con regresión lineal, Ridge, Random Forest y ETS usando MAE, RMSE y MAPE.

**Limitación.** Los datos 2018–2070 son proyecciones oficiales, no observaciones: el modelo aprende la dinámica demográfica de la proyección del DANE.
""")
