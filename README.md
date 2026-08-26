# Property Oracle — California Housing & Bank Churn

Proyecto de Machine Learning supervisado sobre dos dominios distintos: predicción de precios de vivienda en California y predicción de abandono de clientes (churn) en un banco europeo.

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
