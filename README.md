# Modelado y evaluación de clasificación: churn de clientes

Pre-entrega de Data Science II (Coderhouse, comisión 90545): Modelado y Evaluación de Clasificación.

## Descripción del problema

Una empresa de telecomunicaciones quiere saber qué clientes se van a dar de baja (*churn*) para poder ofrecerles una acción de retención antes de que se vayan. Es un problema de **clasificación binaria**: la variable a predecir es `Churn` (1 = se fue, 0 = se quedó).

En el notebook `modelado_metricas.ipynb` hago la carga de datos, la división 80/20 en entrenamiento y prueba, entreno una regresión logística y la evalúo con Accuracy, Precision, Recall, F1-Score y matriz de confusión. Al final justifico por qué el **Recall** es la métrica más importante para este problema.

## Dataset

Uso el dataset *Telco Customer Churn* de IBM (`data/Telco-Customer-Churn.csv`): 7043 clientes y 21 columnas con datos del contrato, los servicios contratados y la facturación. Aproximadamente el 26,5% de los clientes se fue (clases desbalanceadas).

Fuente: https://github.com/IBM/telco-customer-churn-on-icp4d (datos de ejemplo de IBM). El archivo está incluido en el repositorio, por lo que el notebook se ejecuta sin conexión a internet.

Limpieza: `TotalCharges` venía como texto y tenía 11 espacios en blanco (clientes nuevos con `tenure` = 0). Los pasé a número y los completé con 0. El detalle está en el notebook.

## Cómo ejecutar el notebook

```bash
# 1. Clonar el repositorio y entrar a la carpeta
git clone <url-del-repositorio>
cd <nombre-del-repositorio>

# 2. (Opcional) crear un entorno virtual
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Abrir el notebook y ejecutar todas las celdas en orden
```

El notebook se puede abrir en VS Code (con la extensión de Jupyter, eligiendo el kernel del entorno donde se instalaron las dependencias) o con Jupyter (`jupyter notebook`). También se puede leer directamente en GitHub: ya viene ejecutado, con los resultados guardados.

En Windows, si `python` o `pip` no se reconocen, usar `py -m venv .venv` y `py -m pip install -r requirements.txt`.

Para correr los tests: `pytest`.

## Resumen de resultados

Resultados sobre el conjunto de prueba (1409 clientes, 20% de los datos):

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Siempre predice "se queda" (referencia) | 0.735 | 0.000 | 0.000 | 0.000 |
| Regresión logística base | 0.806 | 0.657 | 0.559 | 0.604 |
| **Regresión logística balanceada (final)** | 0.738 | 0.504 | **0.783** | 0.614 |

Matriz de confusión del modelo final (filas = real, columnas = predicho):

|  | Predicho: se queda | Predicho: se va |
|---|---|---|
| Real: se quedó | 747 | 288 |
| Real: se fue | 81 | 293 |

- El modelo final (`class_weight="balanced"`) detecta cerca de 8 de cada 10 clientes que se van. El costo es una Precision de 0,50: la mitad de las alertas son falsas.
- Elegí el Recall como métrica principal porque un falso negativo (perder a un cliente sin avisar) cuesta bastante más que un falso positivo (ofrecer un descuento que no hacía falta). Con costos de ejemplo supuestos, el modelo balanceado tiene un costo total de errores menor que el base, aunque su Accuracy sea menor.
- La elección entre variantes se hizo con validación cruzada solo sobre el conjunto de entrenamiento; el conjunto de prueba se usó una única vez para la evaluación final.

Los valores pueden variar en la tercera cifra decimal según la versión de scikit-learn instalada.

## Estructura del repositorio

```
modelado_metricas.ipynb     # notebook principal (ya ejecutado)
churn_utils.py              # funciones de carga, limpieza y pipeline
data/
    Telco-Customer-Churn.csv
tests/
    test_modelado.py        # tests con pytest
requirements.txt
pytest.ini
.gitignore
README.md
```

## Tests

`pytest` corre tests que cubren: la carga del dataset versionado, la limpieza de `TotalCharges` y `Churn`, un CSV vacío o inexistente (da error), la proporción de churn en la división 80/20, una categoría no vista al predecir, un DataFrame vacío al entrenar, que `class_weight="balanced"` suba el Recall y que el notebook no tenga errores guardados.
