# Property Oracle — California Housing & Bank Churn

Proyecto de Machine Learning supervisado sobre dos dominios distintos: predicción de precios de vivienda en California y predicción de abandono de clientes (churn) en un banco europeo. Sigue el "Ritual de los 7 pasos": EDA → feature engineering → selección de features → división train/test sin fuga → preprocesamiento en pipelines → entrenamiento → evaluación sobre test, con interpretación humana y conclusiones honestas. Los datos base ya vienen **curados**: California desde el proyecto C1DS (`housing_clean.csv`) y churn desde su notebook de EDA dedicado (`bank_churn_clean.csv`).

El desafío plantea dos preguntas centrales: *¿cuánto vale una vivienda dado su contexto?* y *¿qué clientes están en riesgo real de abandonar el banco?* Para cada una entrenamos un modelo supervisado básico — regresión lineal y regresión logística — y lo evaluamos con las métricas correctas para su dominio, verificando además que generalice (train vs test) y que no memorice.

## Datasets utilizados

| Dataset | Fuente | Descripción |
|---|---|---|
| California Housing Prices | [Kaggle — camnugent](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | Precios de vivienda en California (1990), a nivel de bloque censal |
| Bank Customer Churn | [Kaggle — abbas829](https://www.kaggle.com/datasets/abbas829/bank-customer-churn) | 10.000 clientes de un banco europeo y su indicador de abandono (`Exited`) |

## Estructura del repositorio

```
├── data/
│   ├── raw/                      # Datasets originales, sin modificar
│   │   ├── housing.csv           # Dataset de regresión (California)
│   │   └── Bank_Churn.csv        # Dataset de clasificación (churn bancario)
│   └── processed/                # Datasets curados (entradas del notebook principal)
│       ├── housing_clean.csv     # California curado — heredado del proyecto C1DS
│       └── bank_churn_clean.csv  # Churn curado — exportado por notebooks/churn_eda.ipynb
├── notebooks/
│   ├── Property_Oracle.ipynb    # Notebook principal (36 celdas, ejecutado)
│   └── churn_eda.ipynb          # EDA profundo de Bank Churn + exportación curada
├── requirements.txt              # Dependencias del proyecto
└── README.md
```

## Notebooks

### Notebook de EDA — Churn (`notebooks/churn_eda.ipynb`)

Análisis exploratorio del dataset crudo `Bank_Churn.csv` (10.000 clientes): calidad de datos, balance de clases, análisis bivariado (edad, geografía, actividad, productos) y correlaciones. Cierra exportando `data/processed/bank_churn_clean.csv` (descarta identificadores, codifica `Geography`/`Gender` con `drop_first`, sin nulos), que es lo que consume el notebook principal.

### Notebook principal — Propuesta (`notebooks/Property_Oracle.ipynb`)

### Sesión 1 — Regresión: California Housing

Predicción del precio medio de vivienda a partir de características del bloque censal, sobre el **dataset curado de C1DS** (19.087 bloques, sin techo de $500,001 ni outliers p99 en ratios).

**Proceso:** contexto del EDA previo (techos artificiales detectados y filtrados, ratios curados p99) → feature engineering (ratios curados, cross feature `age_x_income`, zonas geográficas con K-Means, one-hot de `ocean_proximity`) → selección de features → train/test split (80/20, semilla fija) → escalado dentro de un pipeline sin fuga de datos → regresión lineal (OLS) → evaluación → interpretación de coeficientes.

**Hallazgos principales:**
- `median_income` es el predictor dominante (+$47,046 por desviación estándar), aunque parte de su señal la absorbe la cross feature `age_x_income` (+$13,974): una casa vieja **y** rica vale más que la suma de ambas por separado.
- Las coordenadas crudas (`latitude`/`longitude`) conservan peso (~ -$45K / -$63K) pese a las zonas K-Means: la señal espacial no se captura del todo con los clusters. Esta decisión de retenerlas quedó documentada tras una prueba controlada (quitarlas bajaba el R² de forma sensible).
- Los **ratios curados** aportan señal de ocupación: `bedrooms_per_room` (+$24,416) y `rooms_per_household` (+$18,559) suben el precio; `population_per_household` (-$20,014) lo baja. Las zonas de K-Means (one-hot k-1, con `geo_zone_0` como referencia) las hacen interpretables (`geo_zone_1` +$7,276, `geo_zone_5` -$14,523).
- `ocean_proximity` sigue la intuición: `INLAND` descuenta (~ -$19K) y `NEAR OCEAN` aumenta (+$19K). `ISLAND` muestra un coeficiente enorme (+$123,829) pese a representar menos del 1% del dataset — señal de poca evidencia, no una regla de mercado.
- El diagnóstico de residuos revela que el modelo **subestima sistemáticamente las viviendas de alto valor** (heterocedasticidad visible en residuos vs predicción), consistente con las limitaciones de un modelo lineal frente al mercado de lujo.

**Resultado:** RMSE ≈ $55,207 · R² ≈ 0.6790 · verificación train vs test Δ < 5% → **generaliza** (no memoriza).

### Sesión 2 — Clasificación: Bank Churn

Predicción de abandono de clientes bancarios a partir de datos demográficos, de producto y de actividad.

**Proceso:** carga del dataset curado exportado por `notebooks/churn_eda.ipynb` (sin identificadores ni nulos, categóricas codificadas) → feature engineering con **cross features** (`Balance_x_NumOfProducts`, `Tenure_per_Age`, `Balance_per_Salary`) → train/test split estratificado (80/20) → escalado → regresión logística → evaluación con Accuracy/Precision/Recall y matriz de confusión → comparación con `class_weight='balanced'` → interpretación de coeficientes (California y Churn, en unidades de desviación estándar) → reflexión sobre el costo de los errores. El EDA profundo (balance de clases, edad, geografía, actividad, productos) vive en el notebook dedicado.

**Hallazgos principales:**
- `Age` es el predictor más fuerte: los clientes que se van tienen una edad media de **44.8 años** vs 37.4 de los que se quedan.
- **Alemania concentra el 32% de las bajas** (vs ~16% en Francia y España) → `Geography_Germany` es un predictor relevante.
- `IsActiveMember` tiene un **efecto protector**: los miembros activos abandonan menos.
- Las **cross features** (`Balance_x_NumOfProducts`, `Tenure_per_Age`, `Balance_per_Salary`) aportan señal con lectura económica; el modelo balanceado alcanza un **Recall de 0.73**, atrapando al 73% de los que se van.
- El target está desbalanceado (79.63% se queda / 20.37% se va): el **Recall es clave** porque Accuracy por sí sola es engañosa.
- El modelo base alcanza Accuracy 0.81 pero **Recall bajo (0.21)**: deja escapar al 79% de los que se van. El balanceado sacrifica Accuracy (0.72) para subir Recall a **0.73**.

**Resultado:** Accuracy 0.8090 · Precision 0.5839 · Recall 0.2138 (modelo base) — Accuracy 0.7150 · Precision 0.3929 · Recall 0.7346 (modelo balanceado). La elección entre ambos depende del **costo de cada error** para el negocio, no de un único número "mejor".

## Herramientas

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Instalación

```bash
git clone https://github.com/katherinevarela775/C2DS-Property_Oracle.git
cd C2DS-Property_Oracle
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Property_Oracle.ipynb
```

El notebook está autocontenido y con todas las celdas ya ejecutadas: basta abrirlo para ver el pipeline completo, o ejecutarlo con **Kernel → Restart & Run All** para reproducir los resultados. Bajo cada celda que genera un gráfico hay un bloque corto de **hallazgos** pensado para explicar el resultado en una presentación: qué revela cada subplot, qué implica para el modelo y qué decisión de negocio se desprende.