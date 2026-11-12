# 🔎 GlucoLens

**Predicción de diabetes mediante técnicas de minería de datos aplicadas a indicadores clínicos y demográficos.**

Aplicación web del proyecto de aula de **Minería de Datos** (Universidad de la Costa, CUC). Recibe ocho indicadores clínicos y demográficos de una paciente y estima la probabilidad de que presente diabetes con un modelo de clasificación entrenado previamente.

> ⚠️ **Prototipo académico.** GlucoLens no es un dispositivo médico, no diagnostica y no sustituye el criterio de un profesional de la salud.

- **Aplicación:** https://glucolens.streamlit.app
- **Integrante:** Isaac Mendoza Herrera
- **Docente:** José Escorcia-Gutierrez, Ph.D. — Departamento de Ciencias de la Computación y Electrónica
- **Actividad:** Actividad 1 del tercer corte — Integración y despliegue del modelo con Streamlit

## Qué incluye la aplicación

| Pestaña | Contenido |
|---|---|
| 🏠 Proyecto | Presentación, objetivo, integrante y fuente de los datos. |
| 📊 Exploración de datos | Tres visualizaciones del EDA: distribución del diagnóstico, distribución de cada variable según el diagnóstico (interactiva) y proporción de diabetes por rango de glucosa. |
| 🩺 Predicción | Formulario con las 8 variables del modelo (con unidades), validación de tipos y rangos, opción de "dato no disponible", predicción y probabilidad, y carga de registros conocidos del conjunto de prueba. |
| 📈 Modelo y limitaciones | Métricas finales, matriz de confusión, comparación antes/después de la revisión y limitaciones. |

## Datos

Conjunto de datos **Pima Indians Diabetes** ([Kaggle: amineipad/diabetes-dataset](https://www.kaggle.com/datasets/amineipad/diabetes-dataset)): 768 registros de mujeres de al menos 21 años de herencia indígena Pima, 8 variables predictoras numéricas y la variable objetivo `Outcome` (1 = con diabetes). No contiene datos personales identificables.

## Modelo

- **Pipeline guardado** (`modelo_glucolens.joblib`): imputación de faltantes con la mediana → random forest (300 árboles, `max_depth=8`, `min_samples_leaf=10`, `max_features=0.5`, pesos de clase balanceados) → umbral de decisión de **0.40**.
- La aplicación **carga** este archivo con `joblib`; no vuelve a entrenar el modelo.
- **Criterio de selección:** no se eligió por *accuracy*. El error más costoso es el falso negativo (una paciente con diabetes clasificada como sana), por lo que se priorizó el *recall* de la clase positiva.

### Revisión y ajuste (sección 10 del notebook)

Con la ficha de revisión de calidad del curso se identificó **sobreajuste** en el random forest del segundo corte. Se hicieron dos ajustes, usando solo el conjunto de entrenamiento con validación cruzada estratificada de 5 pliegues, y se evaluó una sola vez en prueba:

1. Regularización mediante búsqueda de hiperparámetros (48 combinaciones).
2. Ajuste del umbral de decisión de 0.50 a 0.40 (regla: el umbral más alto con *recall* ≥ 0.80 en validación cruzada).

| Métrica (prueba, n = 154) | Segundo corte | Modelo final |
|---|---|---|
| Accuracy | 0.727 | 0.727 |
| Recall (diabetes) | 0.741 | **0.852** |
| Precision (diabetes) | 0.588 | 0.575 |
| F1 (diabetes) | 0.656 | 0.687 |
| ROC-AUC | 0.819 | 0.830 |
| Falsos negativos | 14 | **8** |
| Falsos positivos | 28 | 34 |
| Brecha F1 entrenamiento–prueba | 0.193 | **0.081** |

Validación cruzada del modelo final: *recall* 0.846 ± 0.027, F1 0.690 ± 0.028, *accuracy* 0.735.

Matriz de confusión final: 66 verdaderos negativos, 34 falsos positivos, 8 falsos negativos y 46 verdaderos positivos.

### Limitaciones

- **Falsas alarmas frecuentes:** precision de 0.575 (aproximadamente 4 de cada 10 pacientes marcadas con riesgo no tienen diabetes). Es el costo de bajar el umbral.
- El modelo deja sin detectar 8 de las 54 pacientes con diabetes del conjunto de prueba.
- Frente a la señal de revisión del 70 %: el *accuracy* (0.727) y el *recall* (0.852) la superan; el F1 (0.687) y la precision (0.575) no.
- `Insulin` y `SkinThickness` tienen 48.7 % y 29.6 % de valores faltantes, imputados con la mediana.
- Dataset pequeño (768 registros) y de una población específica: los resultados no se pueden generalizar sin validación externa.
- El modelo identifica asociaciones estadísticas, no causas.

## Pruebas de funcionamiento

Las pruebas están automatizadas en `tests/test_app.py` (con `streamlit.testing`) y también se verificaron en la aplicación:

| Prueba | Entrada | Resultado |
|---|---|---|
| Caso normal | Valores por defecto del formulario (medianas del dataset) | Probabilidad de 51.1 % → clase 1 (riesgo), sin errores. |
| Entrada inválida | Glucosa = 500 mg/dL y edad = 15 años | La aplicación no predice y muestra qué datos están fuera del rango admitido. |
| Registros conocidos del conjunto de prueba | Registros 44, 672, 700, 630 y 81 (`casos_prueba.csv`) | La predicción y la probabilidad de la aplicación coinciden con las del notebook en los cinco (por ejemplo, registro 700: clase 0, 23.4 %). |

## Estructura del repositorio

```
app.py                             Aplicación en Streamlit
requirements.txt                   Dependencias (versiones fijas)
modelo_glucolens.joblib            Pipeline final serializado (preprocesamiento + modelo + umbral)
metadatos_modelo.json              Métricas, umbral, hiperparámetros y rangos de las variables
casos_prueba.csv                   Registros del conjunto de prueba con la predicción del notebook
diabetes.csv                       Conjunto de datos (para las visualizaciones del EDA)
GlucoLens_Proyecto_Diabetes.ipynb  Notebook actualizado (la sección 10 corresponde al tercer corte)
tests/test_app.py                  Pruebas automatizadas de la aplicación
.streamlit/config.toml             Tema visual
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se desarrolló con Python 3.13. El modelo se serializó con scikit-learn 1.9.1, por lo que debe cargarse con esa misma versión.

## Uso y evaluación de inteligencia artificial

| Elemento | Descripción |
|---|---|
| **Herramienta** | Claude (Anthropic), a través de Claude Code. |
| **Propósito** | Revisar el modelo del segundo corte con la ficha de calidad, proponer y programar los ajustes, escribir `app.py`, las pruebas, este README y apoyar el despliegue. |
| **Prompt principal** | «Mira la nueva actividad», enviado junto con el PDF del enunciado (*Actividad 1 – Tercer corte: Integración y despliegue del modelo con Streamlit*). La IA tenía como contexto el notebook y el informe del segundo corte, la ficha *Revisión de resultados* y el *Tutorial Streamlit* del curso. El trabajo se hizo por partes; en cada una la IA presentó un resultado y el estudiante lo aprobó o pidió aclaraciones, por ejemplo: «Explicame esa parte del costo del ajuste y la señal del 70%, haz como si no conociera nada del proyecto». |
| **Cómo se verificó** | (1) El notebook se ejecutó completo, en orden y sin errores. (2) Se comprobó con `assert` que el modelo cargado desde el archivo `.joblib` reproduce exactamente las predicciones y probabilidades del notebook. (3) Pruebas automatizadas de la aplicación: carga, caso normal, entrada inválida y cinco registros del conjunto de prueba, cuyas predicciones coinciden con las del notebook. (4) Revisión visual de cada pestaña. (5) Las cifras de la aplicación y de este README se leen de `metadatos_modelo.json`, generado por el notebook, y se contrastaron con sus salidas. |
| **Problemas encontrados** | El modelo del segundo corte, construido con apoyo de la IA, presentaba sobreajuste (brecha de F1 de 0.193). El ajuste del umbral propuesto por la IA mejora el *recall* pero aumenta las falsas alarmas y reduce la precision, algo que debía entenderse antes de aceptarlo. En la primera versión de la aplicación, una métrica del resumen aparecía cortada, los campos del formulario quedaban desalineados y el código usaba un parámetro obsoleto de Streamlit (`use_container_width`). El perfil por defecto del formulario (medianas) se clasifica como riesgo (51.1 %), lo que refleja la tendencia del modelo a generar falsas alarmas. |
| **Ajustes realizados** | Se regularizó el modelo y se documentó la comparación antes/después; el costo del umbral se declaró como limitación en el notebook, la aplicación y este README; se corrigieron la métrica cortada, la alineación del formulario y el parámetro obsoleto; se fijaron las versiones en `requirements.txt` para que el modelo serializado cargue igual en la nube. |

El integrante revisó el contenido generado con apoyo de IA, comprende el código entregado y asume la responsabilidad académica por el repositorio y la aplicación.
