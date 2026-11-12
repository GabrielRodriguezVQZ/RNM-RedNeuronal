"""
app.py — Aplicación web (Streamlit) para explorar los resultados.

Lee los archivos que generó modelo.py (predicciones, métricas y configuración)
y los muestra en cinco pestañas: Guía de inicio, Proyección, Comparación,
Datos y Metodología.

Uso:  streamlit run app.py
"""
import json
import pandas as pd, plotly.graph_objects as go, streamlit as st

# Configuración general de la página (título de la pestaña del navegador, ancho, etc.)
st.set_page_config(page_title="RNM Colombia · Red Neuronal", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# CARGA DE RESULTADOS (generados antes por modelo.py)
# ----------------------------------------------------------------------------
d = pd.read_csv("data/predicciones.csv")    # RNM real y predicción de cada modelo
m = pd.read_csv("data/metricas.csv")        # errores de cada modelo
cfg = json.load(open("data/config.json"))   # año de corte, número de semillas, variables
corte = cfg["anio_corte"]

# Nombres "bonitos" para mostrar en gráficos y tablas.
NOMBRES = {"RNN": "Red neuronal (Keras)", "Regresion_lineal": "Regresión lineal", "Ridge": "Ridge",
           "Random_Forest": "Random Forest", "ETS": "Suavizado exponencial (ETS)"}

# Un ejemplo real para la guía: el Total nacional en 2018.
ej = d[(d.area == "Total") & (d.anio == 2018)].iloc[0]

# ----------------------------------------------------------------------------
# ENCABEZADO DEL PROYECTO
# ----------------------------------------------------------------------------
st.title("📈 Red Neuronal para la Relación Niños-Mujeres (RNM) de la población de Colombia")
st.markdown("**Proyecto de Ciencia de Datos** · Proyecciones de población del DANE, 2018-2070")
st.caption("RNM = (población 0-4 años ÷ mujeres 15-49 años) × 100. Fuente: DANE, Proyecciones de población (PPED, act. julio 2025).")
st.info("👋 **¿Primera vez aquí?** Empieza por la pestaña **🧭 Guía de inicio**: en 2 minutos entenderás qué muestra cada parte.")

# ----------------------------------------------------------------------------
# BARRA LATERAL: controles que elige el usuario
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controles")
    st.caption("Estos filtros cambian los gráficos de las pestañas **Proyección** y **Datos**.")
    area = st.selectbox(
        "Área geográfica", ["Total", "Cabecera", "Rural"],
        help="Total = todo el país. Cabecera = zonas urbanas (cabeceras municipales). "
             "Rural = centros poblados y rural disperso. Por área solo hay datos hasta 2050.")
    modelos = st.multiselect(
        "Modelos a comparar", list(NOMBRES), default=["RNN", "Regresion_lineal"], format_func=NOMBRES.get,
        help="Elige qué modelos dibujar junto al RNM oficial. RNN es la red neuronal; "
             "los demás son modelos de comparación.")
    st.info(f"**Entrenamiento:** 2018–{corte}\n\n**Prueba:** {corte+1} en adelante (la red no ve esos años al entrenar).")

g = d[d.area == area]   # solo las filas del área elegida
t0, t1, t2, t3, t4 = st.tabs(["🧭 Guía de inicio", "📊 Proyección", "🏆 Comparación de modelos", "🗃️ Datos", "📚 Metodología"])

# ----------------------------------------------------------------------------
# PESTAÑA 0: GUÍA DE INICIO (para quien usa la app por primera vez)
# ----------------------------------------------------------------------------
with t0:
    st.header("Guía rápida para usar esta aplicación")

    st.subheader("1. ¿Qué es el RNM?")
    st.markdown(f"""
La **Relación Niños-Mujeres** compara cuántos niños pequeños hay con cuántas mujeres están en edad de tener hijos.
Sirve como una medida **indirecta de la fecundidad**: si baja, hay menos niños por cada mujer fértil.

**Ejemplo con datos reales (Colombia, 2018):**
- Niños de 0 a 4 años: **{ej.ninos_0_4:,.0f}**
- Mujeres de 15 a 49 años: **{ej.mujeres_15_49:,.0f}**
- RNM = {ej.ninos_0_4:,.0f} ÷ {ej.mujeres_15_49:,.0f} × 100 = **{ej.RNM:.1f}**

Es decir, en 2018 había unos **{ej.RNM:.0f} niños de 0 a 4 años por cada 100 mujeres en edad fértil**.
""")

    st.subheader("2. ¿Qué hace la red neuronal?")
    st.markdown(f"""
Una **red neuronal** es un modelo matemático que aprende patrones a partir de ejemplos.
Aquí aprende a **estimar el RNM** a partir de cómo están distribuidas las mujeres por edades
(cuántas hay de 15-19 años, de 20-24, etc.). Para comprobar que realmente aprendió, se entrena con los años
**2018 a {corte}** y luego se le pide predecir los años siguientes, que nunca vio.
""")

    st.subheader("3. Cómo usar la app, paso a paso")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**Paso 1**\n\n👈 En la **barra lateral** elige el *área* (Total, Cabecera o Rural) y los *modelos* que quieres ver.")
    c2.markdown("**Paso 2**\n\nVe a **📊 Proyección**. Verás el RNM oficial (línea negra) y lo que predijo cada modelo (líneas punteadas).")
    c3.markdown("**Paso 3**\n\nMueve el **deslizador de año** para ver los valores exactos de un año concreto.")
    c4.markdown("**Paso 4**\n\nEntra a **🏆 Comparación** para saber qué modelo se equivoca menos, y a **🗃️ Datos** para descargar la tabla.")

    st.subheader("4. Cómo leer el gráfico de la proyección")
    st.markdown(f"""
- **Línea negra gruesa:** el RNM "real", calculado directamente con los datos del DANE.
- **Líneas punteadas de colores:** lo que predice cada modelo. *Mientras más pegada a la línea negra, mejor.*
- **Franja naranja ("Zona de prueba"):** años que el modelo **no vio** al entrenar (desde {corte+1}). Es la parte
  más importante: ahí se ve si el modelo realmente sirve para predecir.
""")

    st.subheader("5. Glosario")
    with st.expander("Ver términos importantes"):
        st.markdown("""
| Término | Qué significa |
|---|---|
| **Red neuronal** | Modelo que aprende patrones ajustando muchos números (pesos) a partir de ejemplos. |
| **Entrenamiento** | Etapa en la que el modelo aprende, usando los años 2018 a 2040. |
| **Prueba** | Etapa en la que se evalúa el modelo con años que no vio (2041 en adelante). |
| **Modelo de comparación (baseline)** | Modelo más simple con el que se compara la red, para saber si la red realmente aporta. |
| **MAE** | Error absoluto medio: cuánto se equivoca en promedio, en puntos de RNM. |
| **RMSE** | Parecido al MAE, pero castiga más los errores grandes. |
| **MAPE** | Error porcentual medio: en promedio, qué % se equivoca. *Menor es mejor.* |
| **Proyección** | Estimación oficial del DANE hacia el futuro; no son datos observados. |
| **Cabecera / Rural** | Zonas urbanas vs. centros poblados y rural disperso. |
""")

    st.subheader("6. Preguntas frecuentes")
    with st.expander("¿Por qué algunos modelos se equivocan más que otros?"):
        st.write("Cada modelo captura distinto la tendencia. Con pocos datos, los modelos simples (como la regresión lineal) "
                 "pueden ser tan buenos como la red neuronal. Por eso se comparan: así se sabe si la red realmente mejora.")
    with st.expander("¿Estos datos son del futuro real?"):
        st.write("No. Los valores de 2018 a 2070 son **proyecciones oficiales del DANE**, no observaciones. "
                 "La red aprende la dinámica demográfica que esa proyección contiene; no predice el futuro real del país.")
    with st.expander("¿Por qué por área solo hay datos hasta 2050?"):
        st.write("Porque el DANE publica la desagregación Cabecera/Rural solo hasta 2050. El Total nacional llega hasta 2070.")
    with st.expander("¿Qué tan confiable es la red?"):
        st.write("Mira la pestaña **Comparación de modelos**: allí se muestra su error en los años de prueba. "
                 "Un MAPE bajo (por ejemplo, cercano al 2-3 %) indica que se acerca mucho al valor oficial.")

# ----------------------------------------------------------------------------
# PESTAÑA 1: gráfico de la proyección
# ----------------------------------------------------------------------------
with t1:
    st.subheader(f"RNM proyectado — área: {area}")
    fig = go.Figure()
    # Línea sólida: el RNM "oficial" calculado directamente del Excel del DANE.
    fig.add_scatter(x=g.anio, y=g.RNM, name="RNM oficial (calculado del DANE)", line=dict(color="#1f2937", width=3))
    # Líneas punteadas: la predicción de cada modelo seleccionado.
    for k in modelos:
        fig.add_scatter(x=g.anio, y=g[f"pred_{k}"], name=NOMBRES[k], line=dict(dash="dash"))
    # Franja naranja: años de prueba (los que el modelo no vio al entrenar).
    fig.add_vrect(x0=corte + .5, x1=g.anio.max(), fillcolor="#f59e0b", opacity=.10, line_width=0,
                  annotation_text="Zona de prueba", annotation_position="top left")
    fig.update_layout(xaxis_title="Año", yaxis_title="RNM (niños 0-4 por 100 mujeres 15-49)", hovermode="x unified", height=480, legend=dict(orientation="h", y=-.2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Pasa el cursor sobre el gráfico para ver los valores de cada año. Haz clic en la leyenda para ocultar o mostrar una línea.")

    with st.expander("ℹ️ ¿Cómo leer este gráfico?"):
        st.markdown(f"""
La línea negra es el RNM oficial. Las líneas punteadas son las predicciones de los modelos que elegiste.
En la **zona naranja** (desde {corte+1}) los modelos predicen años que no vieron al entrenar: cuanto más cerca estén
de la línea negra, mejor funcionan. Un RNM que **baja** significa menos niños de 0-4 años por cada mujer en edad fértil.
""")

    st.markdown("#### Consulta un año específico")
    # Selector de año y tarjetas con los valores de ese año.
    anio = st.slider("Año", int(g.anio.min()), int(g.anio.max()), 2030,
                     help="Mueve el deslizador para ver los valores exactos de ese año.")
    r = g[g.anio == anio].iloc[0]
    c = st.columns(4)
    c[0].metric("RNM oficial", f"{r.RNM:.2f}", help="Niños 0-4 por cada 100 mujeres de 15-49, según el DANE.")
    c[1].metric("Red neuronal", f"{r.pred_RNN:.2f}", f"{r.pred_RNN - r.RNM:+.2f}",
                help="Lo que predice la red. El número pequeño es la diferencia con el valor oficial.")
    c[2].metric("Niños 0-4", f"{r.ninos_0_4:,.0f}")
    c[3].metric("Mujeres 15-49", f"{r.mujeres_15_49:,.0f}")
    # Mensaje que indica si el año consultado estuvo en entrenamiento o en prueba.
    if anio <= corte:
        st.caption(f"📘 {anio} es un año de **entrenamiento**: la red lo vio al aprender.")
    else:
        st.caption(f"📙 {anio} es un año de **prueba**: la red NO lo vio al entrenar, así que la diferencia mide su capacidad real de predecir.")

# ----------------------------------------------------------------------------
# PESTAÑA 2: comparación de modelos con sus métricas de error
# ----------------------------------------------------------------------------
with t2:
    st.subheader("¿Qué modelo se equivoca menos?")
    st.markdown("Aquí se comparan todos los modelos con los mismos datos. **Menor error = mejor modelo.**")
    conj = st.radio(
        "Conjunto de evaluación", ["prueba", "prueba Total 2051-2070", "entrenamiento"], horizontal=True,
        help="prueba: años 2041 en adelante, no vistos al entrenar (lo más importante). "
             "prueba Total 2051-2070: solo el Total nacional en los años más lejanos. "
             "entrenamiento: los años con los que aprendió el modelo.")
    mm = m[m.conjunto == conj].copy(); mm["modelo"] = mm.modelo.map(NOMBRES)
    mejor = mm.sort_values("MAPE_%").iloc[0]
    st.success(f"🏅 En este conjunto, el modelo con menor error porcentual es **{mejor.modelo}** (MAPE {mejor['MAPE_%']:.2f} %).")
    st.dataframe(mm.drop(columns="conjunto").sort_values("RMSE").round(3), hide_index=True, use_container_width=True)
    # Gráfico de barras ordenado por MAPE (menor = mejor).
    b = go.Figure(go.Bar(x=mm.sort_values("MAPE_%").modelo, y=mm.sort_values("MAPE_%")["MAPE_%"], marker_color="#2563eb"))
    b.update_layout(yaxis_title="MAPE (%) — menor es mejor", height=360)
    st.plotly_chart(b, use_container_width=True)
    with st.expander("ℹ️ ¿Qué significan MAE, RMSE y MAPE?"):
        st.markdown("""
- **MAE:** en promedio, cuántos puntos de RNM se equivoca el modelo.
- **RMSE:** como el MAE, pero penaliza más los errores grandes.
- **MAPE:** el error promedio expresado en porcentaje del valor real.

Un modelo es mejor cuanto **más bajos** sean estos tres valores.
""")
    st.caption("Si la regresión lineal iguala o supera a la red, se reporta tal cual: con ~100 observaciones una red no tiene por qué ganar.")

# ----------------------------------------------------------------------------
# PESTAÑA 3: tabla de datos y botón de descarga
# ----------------------------------------------------------------------------
with t3:
    st.subheader(f"Datos del área: {area}")
    st.caption("Una fila por año. **RNM** es el valor oficial y **pred_RNN** la predicción de la red neuronal.")
    st.dataframe(g[["anio", "ninos_0_4", "mujeres_15_49", "RNM", "pred_RNN"]].round(3), hide_index=True, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", g.to_csv(index=False), f"rnm_{area}.csv")

# ----------------------------------------------------------------------------
# PESTAÑA 4: explicación de la metodología (texto)
# ----------------------------------------------------------------------------
with t4:
    st.subheader("Metodología")
    st.markdown(f"""
**Datos.** Hoja *PobNacionalxÁreaSexoEdad* del PPED: población por año, área, sexo y edad simple. Por área solo hay datos 2018–2050; el Total llega a 2070.

**Variable objetivo.** RNM por 100 mujeres, calculado por año y área.

**Variables de entrada.** Estructura de las mujeres en edad fértil (distribución por grupos quinquenales 15-19…45-49), mujeres de 5-9 y 10-14 relativas, edad media de las mujeres 15-49 y el área. **No se usa la población 0-4**, para evitar fuga de información.

**Red.** Keras: Dense(16, ReLU, L2) → Dropout(0,1) → Dense(8) → Dense(1). Early stopping, promedio de {cfg['semillas']} semillas.

**Validación.** Split temporal (entrena ≤ {corte}). Comparada con regresión lineal, Ridge, Random Forest y ETS usando MAE, RMSE y MAPE.

**Limitación.** Los datos 2018–2070 son proyecciones oficiales, no observaciones: el modelo aprende la dinámica demográfica de la proyección del DANE.
""")