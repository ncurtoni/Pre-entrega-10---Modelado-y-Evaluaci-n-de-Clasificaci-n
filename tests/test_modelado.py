import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

# permite importar churn_utils desde la carpeta principal
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from churn_utils import (RUTA_CSV, cargar_datos, construir_pipeline,  # noqa: E402
                         limpiar_datos, separar_columnas)


@pytest.fixture
def df_chico():
    """Dataset armado a mano con un TotalCharges vacío, como en el original."""
    return pd.DataFrame({
        "customerID": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "SeniorCitizen": [0, 1, 0, 0, 1, 0, 0, 1],
        "tenure": [1, 34, 0, 45, 2, 8, 22, 10],
        "MonthlyCharges": [30.0, 56.9, 20.0, 42.3, 70.7, 99.6, 89.1, 29.8],
        "TotalCharges": ["30.0", "1889.5", " ", "1840.7", "151.6", "820.5", "1949.4", "301.9"],
        "Contract": ["Month-to-month", "One year", "Two year", "One year",
                     "Month-to-month", "Month-to-month", "Month-to-month", "One year"],
        "Churn": ["Yes", "No", "No", "No", "Yes", "Yes", "No", "No"],
    })


# ---------- datos ----------

def test_dataset_versionado_existe():
    assert RUTA_CSV.exists()
    df = cargar_datos()
    assert df.shape == (7043, 21)
    assert {"customerID", "Churn", "tenure"} <= set(df.columns)


def test_limpiar_convierte_total_charges_y_churn(df_chico):
    limpio = limpiar_datos(df_chico)
    assert pd.api.types.is_numeric_dtype(limpio["TotalCharges"])
    assert limpio["TotalCharges"].isna().sum() == 0
    assert limpio.loc[2, "TotalCharges"] == 0          # el espacio en blanco pasó a 0
    assert set(limpio["Churn"].unique()) == {0, 1}
    assert "customerID" not in limpio.columns


def test_limpiar_dataframe_sin_columna_id(df_chico):
    limpio = limpiar_datos(df_chico.drop(columns=["customerID"]))
    assert len(limpio) == len(df_chico)


def test_cargar_csv_vacio_da_error(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text("customerID,Churn\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_datos(ruta)


def test_cargar_csv_inexistente_da_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_datos(tmp_path / "no_existe.csv")


# ---------- división train/test ----------

def test_division_80_20_mantiene_proporcion_de_churn():
    df = limpiar_datos(cargar_datos())
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    assert len(X_test) == pytest.approx(0.2 * len(df), abs=1)
    assert len(X_train) + len(X_test) == len(df)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


# ---------- pipeline ----------

def test_separar_columnas(df_chico):
    X = limpiar_datos(df_chico).drop(columns=["Churn"])
    numericas, categoricas = separar_columnas(X)
    assert "tenure" in numericas and "TotalCharges" in numericas
    assert categoricas == ["Contract"]


def test_pipeline_predice_con_categoria_no_vista(df_chico):
    limpio = limpiar_datos(df_chico)
    X, y = limpio.drop(columns=["Churn"]), limpio["Churn"]
    numericas, categoricas = separar_columnas(X)
    pipeline = construir_pipeline(numericas, categoricas).fit(X, y)
    nuevo = X.iloc[[0]].copy()
    nuevo["Contract"] = "Contrato nuevo"
    assert len(pipeline.predict(nuevo)) == 1


def test_pipeline_con_dataframe_vacio_da_error():
    pipeline = construir_pipeline(["a"], ["b"])
    vacio = pd.DataFrame({"a": [], "b": []})
    with pytest.raises(ValueError):
        pipeline.fit(vacio, pd.Series([], dtype=int))


def test_class_weight_balanced_sube_el_recall():
    """Con class_weight='balanced' el modelo detecta más bajas que sin él."""
    from sklearn.metrics import recall_score
    df = limpiar_datos(cargar_datos())
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    numericas, categoricas = separar_columnas(X)
    base = construir_pipeline(numericas, categoricas).fit(X_train, y_train)
    balanceado = construir_pipeline(numericas, categoricas, "balanced").fit(X_train, y_train)
    assert recall_score(y_test, balanceado.predict(X_test)) > recall_score(y_test, base.predict(X_test))


# ---------- notebook ----------

def test_notebook_existe_y_no_tiene_errores_guardados():
    ruta = RAIZ / "modelado_metricas.ipynb"
    assert ruta.exists()
    notebook = json.loads(ruta.read_text(encoding="utf-8"))
    codigo = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(codigo) > 0
    for celda in codigo:
        assert all(s["output_type"] != "error" for s in celda.get("outputs", []))
