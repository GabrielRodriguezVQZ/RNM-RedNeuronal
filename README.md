# 📈 RNM Colombia — Red neuronal para proyectar la Relación Niños-Mujeres

Proyecto de **Ciencia de Datos** que construye una red neuronal (Keras/TensorFlow) para estimar y proyectar la **Relación Niños-Mujeres (RNM)** de la población de Colombia entre 2018 y 2070, con una **app web interactiva en Streamlit** para explorar los resultados.

![Captura de la aplicación](docs/captura_app.png)

## 📌 ¿Qué es el RNM?

La Relación Niños-Mujeres es un indicador indirecto de fecundidad:

$$RNM = \frac{\text{Población de 0 a 4 años}}{\text{Mujeres de 15 a 49 años}} \times 100$$

Se expresa como niños de 0-4 años por cada 100 mujeres en edad fértil. (Si tu curso lo usa por 1.000 mujeres, cambia `FACTOR` en `datos.py`.)

En los datos del DANE, el RNM nacional pasa de **30,4 en 2018** a **16,7 en 2066**, reflejo del descenso de la fecundidad.

## 🗂️ Estructura del repositorio

```
RNM-RedNeuronal/
├── data/
│   └── PPED-AreaSexoEdadNac-2018-2070.xlsx   # datos oficiales del DANE
├── datos.py            # lectura del Excel, cálculo del RNM y variables
├── modelo.py           # entrenamiento de la red y de los modelos de comparación
├── app.py              # aplicación web (Streamlit + Plotly)
├── requirements.txt
└── README.md
```

Al ejecutar los scripts se generan en `data/`: `rnm_dataset.csv`, `predicciones.csv`, `metricas.csv` y `config.json`.

## 📊 Datos

- **Fuente:** DANE, *Proyecciones de población y estudios demográficos (PPED)*, actualización de julio de 2025. Serie nacional 2018-2070.
- **Desagregación:** año, área geográfica (Total, Cabecera, Centros poblados y rural disperso), sexo y edad simple (0 a 100+).
- **Cobertura:** el Total llega hasta 2070; por área solo hay datos de 2018 a 2050.

> ⚠️ **Limitación importante:** los valores 2018-2070 son **proyecciones oficiales**, no observaciones. El modelo aprende la dinámica demográfica contenida en esa proyección; no se está prediciendo el futuro real.

## 🧠 Metodología

### Variable objetivo
RNM por 100 mujeres, calculado por año y por área.

### Variables de entrada (12)
| Variable | Descripción |
|---|---|
| `m_15_19` … `m_45_49` (7) | Proporción de las mujeres de 15-49 en cada grupo quinquenal |
| `m_5_9_rel`, `m_10_14_rel` | Niñas de 5-9 y 10-14 años relativas a las mujeres 15-49 (futuras madres) |
| `edad_media_mf` | Edad media de las mujeres de 15-49 |
| `es_rural`, `es_total` | Indicadores del área geográfica |

No se usa la población de 0-4 años como entrada, para evitar fuga de información (el RNM se calcula con ella).

### Arquitectura de la red

```
12 entradas → Dense(16, ReLU, L2) → Dropout(0.1) → Dense(8, ReLU) → Dense(1) → RNM
```

- 353 parámetros: una red pequeña a propósito, porque hay pocas observaciones (~120).
- Optimizador Adam, pérdida MSE, *early stopping* con 20 % de validación.
- Se entrenan **10 redes con semillas distintas** y se promedian sus predicciones para reducir la varianza.
- Variables y objetivo estandarizados con `StandardScaler`.

### Validación
- **Partición temporal:** entrenamiento 2018-2040, prueba 2041 en adelante (nunca aleatoria).
- **Modelos de comparación:** regresión lineal, Ridge, Random Forest (scikit-learn) y suavizado exponencial ETS (statsmodels).
- **Métricas:** MAE, RMSE y MAPE.

## 🏆 Resultados (conjunto de prueba, 2041 en adelante)

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Regresión lineal | 0,395 | 0,442 | 2,22 % |
| **Red neuronal (Keras)** | 0,463 | 0,542 | 2,61 % |
| Ridge | 0,500 | 0,613 | 2,79 % |
| Random Forest | 0,956 | 1,121 | 5,50 % |
| ETS | 1,230 | 1,592 | 7,07 % |

En la extrapolación más lejana (Total, 2051-2070) la red es el mejor modelo: **MAPE 2,27 %** frente a 3,50 % de la regresión lineal.

**Conclusión:** con pocos datos y una relación demográfica casi lineal, un modelo lineal simple es muy competitivo en el periodo de prueba completo. La red neuronal lo supera cuando extrapola a los años más lejanos, y ambos superan con claridad a Random Forest y ETS.

## 🚀 Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
# 1. Clonar e instalar dependencias
git clone https://github.com/GabrielRodriguezVQZ/RNM-RedNeuronal.git
cd RNM-RedNeuronal
pip install -r requirements.txt

# 2. Leer el Excel y calcular el RNM
python datos.py

# 3. Entrenar la red y los modelos de comparación
python modelo.py

# 4. Abrir la aplicación web
streamlit run app.py
```

## 🖥️ La aplicación

Hecha con Streamlit y Plotly. Incluye cuatro pestañas:

1. **Proyección:** RNM oficial frente a la predicción de los modelos, con la zona de prueba resaltada y un selector de año.
2. **Comparación de modelos:** tabla y gráfico de MAE, RMSE y MAPE.
3. **Datos:** tabla por año con descarga en CSV.
4. **Metodología:** resumen del enfoque y sus límites.

En la barra lateral se elige el área geográfica y los modelos a comparar.

## 🔁 Reproducibilidad

Las semillas aleatorias están fijadas (`SEED = 42`), por lo que los resultados deberían ser los mismos en cada ejecución con las mismas versiones de las librerías. Pequeñas diferencias entre versiones de TensorFlow son posibles.

## 🛠️ Tecnologías

Python · pandas · NumPy · TensorFlow/Keras · scikit-learn · statsmodels · Streamlit · Plotly

## 📚 Fuente de datos

DANE — Dirección Técnica de Censos y Demografía. *Proyecciones de población y estudios demográficos (PPED), serie nacional 2018-2070*. Actualizado el 18 de julio de 2025.

## 👤 Autor

Gabriel Rodríguez y Janer Gomez — Proyecto de la materia Ciencia de Datos.