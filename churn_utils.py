"""
Funciones de apoyo para el notebook modelado_metricas.ipynb.

Las dejé en un archivo aparte para poder probarlas con pytest y para que el
notebook quede más corto y fácil de leer.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA_CSV = Path(__file__).resolve().parent / "data" / "Telco-Customer-Churn.csv"
COLUMNA_OBJETIVO = "Churn"
COLUMNA_ID = "customerID"


def cargar_datos(ruta=RUTA_CSV):
    """Lee el CSV de clientes. Da error si el archivo no existe o no tiene filas."""
    df = pd.read_csv(ruta)
    if df.empty:
        raise ValueError("El archivo no tiene filas")
    return df


def limpiar_datos(df):
    """
    Deja el DataFrame listo para modelar:
    - TotalCharges viene como texto y tiene 11 espacios en blanco (clientes con
      tenure = 0 que todavía no recibieron ninguna factura). Se convierte a número
      y esos casos se completan con 0.
    - Churn pasa de "Yes"/"No" a 1/0.
    - Se elimina customerID porque es solo un identificador.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df[COLUMNA_OBJETIVO] = (df[COLUMNA_OBJETIVO] == "Yes").astype(int)
    return df.drop(columns=[COLUMNA_ID], errors="ignore")


def separar_columnas(X):
    """Devuelve las listas de columnas numéricas y categóricas de X."""
    numericas = X.select_dtypes(include="number").columns.tolist()
    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    return numericas, categoricas


def construir_preprocesador(numericas, categoricas):
    """Escala las columnas numéricas y codifica las categóricas con OneHotEncoder."""
    return ColumnTransformer([
        ("numericas", StandardScaler(), numericas),
        # handle_unknown="ignore": una categoría nueva no rompe el predict
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])


def construir_pipeline(numericas, categoricas, class_weight=None):
    """
    Pipeline con el preprocesamiento y una regresión logística.
    Todo queda dentro del Pipeline: el escalador y el codificador se ajustan
    solamente con los datos de entrenamiento (así no hay data leakage).
    """
    return Pipeline([
        ("preprocesador", construir_preprocesador(numericas, categoricas)),
        ("modelo", LogisticRegression(max_iter=1000, class_weight=class_weight)),
    ])
