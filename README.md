# Property Oracle — California Housing & Bank Churn

Proyecto de Machine Learning supervisado sobre dos dominios distintos: predicción de precios de vivienda en California y predicción de abandono de clientes (churn) en un banco europeo. El proyecto vive en una sola notebook autocontenida que sigue el "Ritual de los 7 pasos": EDA → feature engineering → selección de features → división train/test sin fuga → preprocesamiento en pipelines → entrenamiento → evaluación sobre test, con interpretación humana y conclusiones honestas.

El desafío plantea dos preguntas centrales: *¿cuánto vale una vivienda dado su contexto?* y *¿qué clientes están en riesgo real de abandonar el banco?* Para cada una entrenamos un modelo supervisado básico — regresión lineal y regresión logística — y lo evaluamos con las métricas correctas para su dominio, verificando además que generalice (train vs test) y que no memorice.

## Datasets utilizados

| Dataset | Fuente | Descripción |
|---|---|---|
| California Housing Prices | [Kaggle — camnugent](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | Precios de vivienda en California (1990), a nivel de bloque censal |
| Bank Customer Churn | [Kaggle — abbas829](https://www.kaggle.com/datasets/abbas829/bank-customer-churn) | 10.000 clientes de un banco europeo y su indicador de abandono (`Exited`) |

## Estructura del repositorio

```
├── data/
│   └── raw/                    # Datasets originales, sin modificar
│       ├── housing.csv         # Dataset de regresión (California)
│       └── Bank_Churn.csv      # Dataset de clasificación (churn bancario)
├── Property_Oracle.ipynb       # Notebook principal (35 celdas, ejecutado)
├── requirements.txt            # Dependencias del proyecto
└── README.md
```

## Notebook — Propuesta principal (Property_Oracle.ipynb)

### Sesión 1 — Regresión: California Housing

Predicción del precio medio de vivienda a partir de características del bloque censal.

**Proceso:** EDA (detección del techo artificial de $500,001 en el target y su eliminación) → feature engineering (ratios con sentido económico, interacción `age_x_income`, zonas geográficas con K-Means, one-hot de `ocean_proximity`) → selección de features → train/test split (80/20, semilla fija) → escalado dentro de un pipeline sin fuga de datos → regresión lineal (OLS) → evaluación → interpretación de coeficientes.

**Hallazgos principales:**
- `median_income` es el predictor dominante (+$43,817 por desviación estándar), consistente con la correlación alta observada en el EDA.
- Las coordenadas crudas (`latitude`/`longitude`) conservan peso (~±$40K–$50K) pese a las zonas K-Means: la señal espacial no se captura del todo con 8 clusters. Esta decisión de retenerlas quedó documentada tras una prueba controlada (quitarlas bajaba el R² de 0.628 a 0.596).
- `ocean_proximity` sigue la intuición: `INLAND` descuenta (~ -$25K) y `NEAR OCEAN` aumenta (+$24K). `ISLAND` muestra un coeficiente enorme (+$181K) pese a representar menos del 1% del dataset — señal de poca evidencia, no una regla de mercado.
- El diagnóstico de residuos revela que el modelo **subestima sistemáticamente las viviendas de alto valor** (heterocedasticidad visible en residuos vs predicción), consistente con las limitaciones de un modelo lineal frente al mercado de lujo.

**Resultado:** RMSE ≈ $60,883 · R² ≈ 0.6282 · verificación train vs test Δ < 5% → **generaliza** (no memoriza).

### Sesión 2 — Clasificación: Bank Churn

Predicción de abandono de clientes bancarios a partir de datos demográficos, de producto y de actividad.

**Proceso:** EDA (balance de clases, patrones por edad/geografía/actividad) → limpieza y encoding → train/test split estratificado (80/20) → escalado → regresión logística → evaluación con Accuracy/Precision/Recall y matriz de confusión → comparación con `class_weight='balanced'` → reflexión sobre el costo de los errores.

**Hallazgos principales:**
- `Age` es el predictor más fuerte: los clientes que se van tienen una edad media de **44.8 años** vs 37.4 de los que se quedan.
- **Alemania concentra el 32% de las bajas** (vs ~16% en Francia y España) → `Geography_Germany` es un predictor relevante.
- `IsActiveMember` tiene un **efecto protector**: los miembros activos abandonan menos.
- El target está desbalanceado (79.63% se queda / 20.37% se va): el **Recall es clave** porque Accuracy por sí sola es engañosa.
- El modelo base alcanza Accuracy 0.81 pero **Recall bajo (0.19)**: deja escapar al 81% de los que se van. El balanceado sacrifica Accuracy (0.71) para subir Recall a **0.70**.

**Resultado:** Accuracy 0.8080 · Precision 0.5891 · Recall 0.1867 (modelo base) — Accuracy 0.7135 · Precision 0.3872 · Recall 0.7002 (modelo balanceado). La elección entre ambos depende del **costo de cada error** para el negocio, no de un único número "mejor".

## Herramientas

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Instalación

```bash
git clone https://github.com/katherinevarela775/C2DS-Property_Oracle.git
cd C2DS-Property_Oracle
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Property_Oracle.ipynb
```

El notebook está autocontenido y con todas las celdas ya ejecutadas: basta abrirlo para ver el pipeline completo, o ejecutarlo con **Kernel → Restart & Run All** para reproducir los resultados.