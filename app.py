"""GlucoLens: prototipo académico para estimar el riesgo de diabetes.

Proyecto de aula de Minería de Datos (Universidad de la Costa).
La aplicación carga el pipeline entrenado en el notebook y NO vuelve a entrenar el modelo.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
COLOR_SIN, COLOR_CON = "#4C78A8", "#E45756"

st.set_page_config(page_title="GlucoLens · Riesgo de diabetes", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------------------
# Definición de las variables del formulario
#   plausible: límites fuera de los cuales el valor se rechaza (entrada inválida)
#   El rango observado en los datos se lee de metadatos_modelo.json y genera advertencias.
# --------------------------------------------------------------------------------------
VARIABLES = {
    "Pregnancies": dict(etiqueta="Número de embarazos", unidad="embarazos", entero=True, plausible=(0, 20),
                        puede_faltar=False, ayuda="Cantidad de embarazos que ha tenido la paciente."),
    "Glucose": dict(etiqueta="Glucosa en plasma", unidad="mg/dL", entero=False, plausible=(40, 300),
                    puede_faltar=True, ayuda="Concentración de glucosa a las 2 horas de una prueba de tolerancia oral."),
    "BloodPressure": dict(etiqueta="Presión arterial diastólica", unidad="mm Hg", entero=False, plausible=(20, 140),
                          puede_faltar=True, ayuda="Presión diastólica (el valor menor de la medición)."),
    "SkinThickness": dict(etiqueta="Grosor del pliegue cutáneo del tríceps", unidad="mm", entero=False,
                          plausible=(5, 100), puede_faltar=True, ayuda="Medida del pliegue de piel en el tríceps."),
    "Insulin": dict(etiqueta="Insulina sérica a las 2 horas", unidad="μU/ml", entero=False, plausible=(10, 900),
                    puede_faltar=True, ayuda="Nivel de insulina en sangre a las 2 horas."),
    "BMI": dict(etiqueta="Índice de masa corporal (IMC)", unidad="kg/m²", entero=False, plausible=(15, 70),
                puede_faltar=True, ayuda="Peso en kg dividido por la estatura en metros al cuadrado."),
    "DiabetesPedigreeFunction": dict(etiqueta="Función de antecedentes familiares", unidad="puntaje", entero=False,
                                     plausible=(0.05, 2.5), puede_faltar=False,
                                     ayuda="Puntaje del dataset que resume los antecedentes familiares de diabetes. "
                                           "Valores más altos indican más antecedentes."),
    "Age": dict(etiqueta="Edad", unidad="años", entero=True, plausible=(21, 90), puede_faltar=False,
                ayuda="El modelo se entrenó solo con mujeres de 21 años o más."),
}
ORDEN = list(VARIABLES)


# --------------------------------------------------------------------------------------
# Carga de recursos (una sola vez)
# --------------------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "modelo_glucolens.joblib")


@st.cache_data
def cargar_metadatos():
    return json.loads((BASE / "metadatos_modelo.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / "diabetes.csv")
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols] = df[cols].replace(0, np.nan)          # ceros imposibles = valores faltantes
    df["Diagnóstico"] = df["Outcome"].map({0: "Sin diabetes", 1: "Con diabetes"})
    return df


@st.cache_data
def cargar_casos():
    return pd.read_csv(BASE / "casos_prueba.csv")


def validar(valores: dict, rangos: dict):
    """Devuelve (errores, advertencias) para los valores ingresados."""
    errores, advertencias = [], []
    for var, cfg in VARIABLES.items():
        v = valores[var]
        nombre = cfg["etiqueta"]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if cfg["puede_faltar"]:
                continue
            errores.append(f"**{nombre}**: es un dato obligatorio.")
            continue
        if not isinstance(v, (int, float, np.integer, np.floating)):
            errores.append(f"**{nombre}**: debe ser un número.")
            continue
        if cfg["entero"] and float(v) != int(v):
            errores.append(f"**{nombre}**: debe ser un número entero.")
            continue
        lo, hi = cfg["plausible"]
        if v < lo or v > hi:
            errores.append(f"**{nombre}**: {v:g} {cfg['unidad']} está fuera del rango admitido "
                           f"({lo:g} a {hi:g} {cfg['unidad']}).")
            continue
        r = rangos[var]
        if v < r["min"] or v > r["max"]:
            advertencias.append(f"**{nombre}**: {v:g} {cfg['unidad']} está fuera del rango de los datos de "
                                f"entrenamiento ({r['min']:g} a {r['max']:g}); la estimación es menos confiable.")
    return errores, advertencias


def predecir(modelo, valores: dict):
    """Aplica el pipeline guardado a un registro. Devuelve (clase, probabilidad)."""
    fila = pd.DataFrame([[np.nan if valores[v] is None else valores[v] for v in ORDEN]], columns=ORDEN, dtype=float)
    return int(modelo.predict(fila)[0]), float(modelo.predict_proba(fila)[0, 1])


def cargar_caso_en_formulario(caso):
    for var, cfg in VARIABLES.items():
        v = caso[var]
        if pd.isna(v):
            st.session_state[f"nd_{var}"] = True
        else:
            st.session_state[f"nd_{var}"] = False
            st.session_state[f"in_{var}"] = int(v) if cfg["entero"] else float(v)
    st.session_state["caso_cargado"] = caso.to_dict()
    st.session_state.pop("resultado", None)


modelo = cargar_modelo()
meta = cargar_metadatos()
df = cargar_datos()
casos = cargar_casos()
RANGOS = meta["rangos_observados"]
UMBRAL = meta["umbral"]

# --------------------------------------------------------------------------------------
# Encabezado y barra lateral
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 GlucoLens")
    st.caption("Estimación del riesgo de diabetes con minería de datos")
    st.warning("**Prototipo académico.** No es un dispositivo médico ni sustituye el criterio de un "
               "profesional de la salud. No debe usarse para diagnosticar ni para tomar decisiones clínicas.")
    st.markdown("**Integrante**  \nIsaac Mendoza Herrera")
    st.markdown("**Asignatura**  \nMinería de Datos · Ingeniería de Sistemas  \nUniversidad de la Costa (CUC)  \n"
                "Prof. José Escorcia-Gutierrez, Ph.D.")
    st.markdown("**Datos**  \n[Diabetes Dataset (Pima Indians)](https://github.com/IsaacMendozaDev/glucolens/blob/main/diabetes.csv)  \n"
                "Fuente original: Kaggle (amineipad/diabetes-dataset)")

st.title("GlucoLens")
st.markdown("##### Predicción de diabetes mediante técnicas de minería de datos aplicadas a indicadores clínicos y demográficos")

tab_inicio, tab_eda, tab_pred, tab_modelo = st.tabs(
    ["🏠 Proyecto", "📊 Exploración de datos", "🩺 Predicción", "📈 Modelo y limitaciones"])

# --------------------------------------------------------------------------------------
# Pestaña 1: presentación
# --------------------------------------------------------------------------------------
with tab_inicio:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("¿Qué es GlucoLens?")
        st.markdown(
            "GlucoLens es la aplicación del proyecto de aula de Minería de Datos. Recibe ocho indicadores clínicos y "
            "demográficos de una paciente y estima la **probabilidad de que presente diabetes** con un modelo de "
            "clasificación entrenado previamente.")
        st.subheader("Objetivo")
        st.markdown(
            "Responder la pregunta del proyecto: **¿es posible predecir si una paciente presenta diabetes a partir de "
            "ocho indicadores clínicos y demográficos?** La aplicación se plantea como apoyo a un *tamizaje "
            "orientativo*: señalar casos que convendría confirmar con pruebas, dando prioridad a no dejar pasar "
            "pacientes con diabetes (falsos negativos).")
        st.subheader("Fuente de los datos")
        st.markdown(
            "Conjunto de datos **Pima Indians Diabetes**, descargado de Kaggle (publicación *Diabetes Dataset* del "
            "usuario amineipad). El archivo utilizado está disponible en el "
            "[repositorio del proyecto](https://github.com/IsaacMendozaDev/glucolens/blob/main/diabetes.csv). Contiene 768 registros de mujeres "
            "de al menos 21 años de herencia indígena Pima, con 8 variables predictoras y el diagnóstico (`Outcome`). "
            "Los datos fueron recolectados originalmente por el Instituto Nacional de Diabetes y Enfermedades "
            "Digestivas y Renales de Estados Unidos.")
    with c2:
        st.subheader("Resumen")
        a, b = st.columns(2)
        a.metric("Registros", f"{len(df)}")
        b.metric("Variables predictoras", "8")
        a.metric("Con diabetes", f"{df['Outcome'].mean() * 100:.1f} %")
        b.metric("Recall del modelo", f"{meta['metricas_prueba']['Recall (1)']:.3f}",
                 help="De las pacientes con diabetes del conjunto de prueba, proporción que el modelo detecta.")
        st.info("**Cómo usar la aplicación**\n\n"
                "1. *Exploración de datos*: gráficas principales del análisis exploratorio.\n"
                "2. *Predicción*: ingrese los datos de una paciente y consulte la estimación.\n"
                "3. *Modelo y limitaciones*: métricas, matriz de confusión y alcance del modelo.")
        st.markdown("**Integrante:** Isaac Mendoza Herrera")

# --------------------------------------------------------------------------------------
# Pestaña 2: EDA
# --------------------------------------------------------------------------------------
with tab_eda:
    st.subheader("Hallazgos principales del análisis exploratorio")
    st.caption("Las gráficas usan los valores observados: los ceros imposibles se tratan como datos faltantes. "
               "Describen asociaciones, no relaciones causales.")
    escala_dx = alt.Scale(domain=["Sin diabetes", "Con diabetes"], range=[COLOR_SIN, COLOR_CON])

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("**1. Distribución del diagnóstico**")
        conteo = df["Diagnóstico"].value_counts().rename_axis("Diagnóstico").reset_index(name="Pacientes")
        conteo["Porcentaje"] = (conteo["Pacientes"] / conteo["Pacientes"].sum() * 100).round(1)
        conteo["Etiqueta"] = conteo.apply(lambda r: f"{r['Pacientes']} ({r['Porcentaje']} %)", axis=1)
        barras = alt.Chart(conteo).mark_bar().encode(
            x=alt.X("Diagnóstico:N", sort=["Sin diabetes", "Con diabetes"], title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Pacientes:Q", title="Número de pacientes"),
            color=alt.Color("Diagnóstico:N", scale=escala_dx, legend=None),
            tooltip=["Diagnóstico", "Pacientes", "Porcentaje"])
        texto = barras.mark_text(dy=-8, color="#1F2937").encode(text="Etiqueta:N")
        st.altair_chart((barras + texto).properties(height=320), width="stretch")
        st.markdown("Hay 500 pacientes sin diabetes (65.1 %) y 268 con diabetes (34.9 %). Por este desbalance, "
                    "el *accuracy* por sí solo no basta para evaluar el modelo.")
    with col2:
        st.markdown("**2. Distribución de cada variable según el diagnóstico**")
        var_sel = st.selectbox("Variable", ORDEN, index=1, format_func=lambda v: f"{VARIABLES[v]['etiqueta']} ({VARIABLES[v]['unidad']})")
        datos_var = df[[var_sel, "Diagnóstico"]].dropna()
        hist = alt.Chart(datos_var).mark_bar(opacity=0.6).encode(
            x=alt.X(f"{var_sel}:Q", bin=alt.Bin(maxbins=30), title=f"{VARIABLES[var_sel]['etiqueta']} ({VARIABLES[var_sel]['unidad']})"),
            y=alt.Y("count():Q", stack=None, title="Número de pacientes"),
            color=alt.Color("Diagnóstico:N", scale=escala_dx, legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("count():Q", title="Pacientes"), "Diagnóstico"])
        st.altair_chart(hist.properties(height=320), width="stretch")
        med = datos_var.groupby("Diagnóstico")[var_sel].median()
        st.markdown(f"Mediana de **{VARIABLES[var_sel]['etiqueta'].lower()}**: {med['Sin diabetes']:g} sin diabetes "
                    f"frente a {med['Con diabetes']:g} con diabetes ({len(datos_var)} valores observados). "
                    "La glucosa es la variable que mejor separa los grupos, aunque las distribuciones se superponen.")

    st.markdown("**3. Proporción de pacientes con diabetes según el rango de glucosa**")
    rangos_g = pd.cut(df["Glucose"], bins=[0, 100, 125, 150, 175, 200], labels=["≤100", "101–125", "126–150", "151–175", "176–200"])
    tasa = (df.assign(Rango=rangos_g).dropna(subset=["Rango"]).groupby("Rango", observed=True)["Outcome"]
            .agg(Pacientes="size", Porcentaje=lambda s: round(s.mean() * 100, 1)).reset_index())
    tasa["Rango"] = tasa["Rango"].astype(str)
    tasa["Etiqueta"] = tasa["Porcentaje"].map(lambda p: f"{p} %")
    b2 = alt.Chart(tasa).mark_bar(color=COLOR_CON).encode(
        x=alt.X("Rango:N", sort=list(tasa["Rango"]), title="Glucosa (mg/dL)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Porcentaje:Q", title="Pacientes con diabetes (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["Rango", "Pacientes", "Porcentaje"])
    t2 = b2.mark_text(dy=-8, color="#1F2937").encode(text="Etiqueta:N")
    regla = alt.Chart(pd.DataFrame({"y": [df["Outcome"].mean() * 100]})).mark_rule(strokeDash=[6, 4], color="gray").encode(y="y:Q")
    st.altair_chart((b2 + t2 + regla).properties(height=320), width="stretch")
    st.markdown("La proporción de diabetes aumenta de forma sostenida con la glucosa: de 8.6 % con valores de hasta "
                "100 mg/dL a 83.6 % entre 176 y 200 mg/dL. La línea discontinua marca el promedio general (34.9 %).")

# --------------------------------------------------------------------------------------
# Pestaña 3: formulario y predicción
# --------------------------------------------------------------------------------------
with tab_pred:
    st.subheader("Estimación del riesgo para una paciente")
    st.warning("Prototipo académico: el resultado es una estimación estadística y **no sustituye una evaluación "
               "médica ni una prueba diagnóstica**. El modelo se entrenó con mujeres de 21 años o más de una "
               "población específica.")

    with st.expander("Cargar un registro conocido del conjunto de prueba (para verificar la aplicación)"):
        opciones = {f"Registro {int(r['indice_original'])} · diagnóstico real: "
                    f"{'con diabetes' if r['Outcome_real'] == 1 else 'sin diabetes'}": i for i, r in casos.iterrows()}
        elegido = st.selectbox("Registro de prueba", list(opciones))
        st.button("Cargar en el formulario", on_click=cargar_caso_en_formulario, args=(casos.loc[opciones[elegido]],))
        st.caption("Estos registros no se usaron para entrenar el modelo. La predicción de la aplicación debe "
                   "coincidir con la obtenida en el notebook.")

    with st.form("formulario"):
        entradas, no_disponible = {}, {}
        for inicio in range(0, len(ORDEN), 2):                 # dos variables por fila, alineadas
            for col, var in zip(st.columns(2), ORDEN[inicio:inicio + 2]):
                cfg, r = VARIABLES[var], RANGOS[var]
                with col:
                    defecto = int(r["mediana"]) if cfg["entero"] else float(round(r["mediana"], 3))
                    st.session_state.setdefault(f"in_{var}", defecto)
                    entradas[var] = st.number_input(
                        f"{cfg['etiqueta']} ({cfg['unidad']})", key=f"in_{var}",
                        step=1 if cfg["entero"] else (0.01 if var == "DiabetesPedigreeFunction" else 0.1),
                        format="%d" if cfg["entero"] else ("%.3f" if var == "DiabetesPedigreeFunction" else "%.1f"),
                        help=f"{cfg['ayuda']} Rango admitido: {cfg['plausible'][0]:g} a {cfg['plausible'][1]:g} {cfg['unidad']}.")
                    if cfg["puede_faltar"]:
                        no_disponible[var] = st.checkbox(
                            "Dato no disponible", key=f"nd_{var}",
                            help="Si no se cuenta con la medición, el modelo usa la mediana de los datos de entrenamiento.")
        enviado = st.form_submit_button("Estimar riesgo", type="primary")

    if enviado:
        valores = {v: (None if no_disponible.get(v) else entradas[v]) for v in ORDEN}
        errores, advertencias = validar(valores, RANGOS)
        if errores:
            st.session_state["resultado"] = {"errores": errores}
        else:
            clase, prob = predecir(modelo, valores)
            st.session_state["resultado"] = {"clase": clase, "prob": prob, "advertencias": advertencias, "valores": valores}

    res = st.session_state.get("resultado")
    if res and res.get("errores"):
        st.error("No se puede calcular la predicción. Corrija los siguientes datos:\n\n" + "\n".join(f"- {e}" for e in res["errores"]))
    elif res:
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad estimada de diabetes", f"{res['prob'] * 100:.1f} %")
            st.progress(min(max(res["prob"], 0.0), 1.0))
            st.caption(f"Umbral de decisión del modelo: {UMBRAL * 100:.0f} %")
        with c2:
            if res["clase"] == 1:
                st.error(f"**Resultado: riesgo de diabetes (clase 1).** La probabilidad estimada es igual o mayor "
                         f"que el umbral de {UMBRAL * 100:.0f} %. Se sugiere confirmación con un profesional de la salud.")
            else:
                st.success(f"**Resultado: sin indicios de diabetes (clase 0).** La probabilidad estimada es menor "
                           f"que el umbral de {UMBRAL * 100:.0f} %. Esto no descarta la enfermedad.")
            faltantes = [VARIABLES[v]["etiqueta"] for v in ORDEN if res["valores"][v] is None]
            if faltantes:
                st.info("Datos no disponibles, reemplazados por la mediana de entrenamiento: " + ", ".join(faltantes) + ".")
            for a in res["advertencias"]:
                st.warning(a)
        caso = st.session_state.get("caso_cargado")
        if caso is not None:
            mismo = all((pd.isna(caso[v]) and res["valores"][v] is None) or
                        (not pd.isna(caso[v]) and res["valores"][v] is not None and np.isclose(caso[v], res["valores"][v]))
                        for v in ORDEN)
            if mismo:
                coincide = (int(caso["prediccion_notebook"]) == res["clase"]
                            and abs(caso["probabilidad_notebook"] - res["prob"]) < 1e-3)
                st.markdown(f"**Verificación con el notebook (registro {int(caso['indice_original'])}):** "
                            f"predicción del notebook = clase {int(caso['prediccion_notebook'])} "
                            f"(probabilidad {caso['probabilidad_notebook'] * 100:.1f} %), diagnóstico real = "
                            f"clase {int(caso['Outcome_real'])}. "
                            + ("✅ La predicción de la aplicación coincide con la del notebook." if coincide
                               else "⚠️ La predicción no coincide con la del notebook."))

# --------------------------------------------------------------------------------------
# Pestaña 4: modelo, métricas y limitaciones
# --------------------------------------------------------------------------------------
with tab_modelo:
    mp, mc, cvm, antes = meta["metricas_prueba"], meta["matriz_confusion"], meta["validacion_cruzada"], meta["metricas_antes"]
    hp = meta["hiperparametros"]
    st.subheader("Modelo final")
    st.markdown(
        f"**Random forest** de 300 árboles con pesos de clase balanceados (`max_depth={hp['max_depth']}`, "
        f"`min_samples_leaf={hp['min_samples_leaf']}`, `max_features={hp['max_features']}`), precedido de la "
        f"imputación de valores faltantes con la mediana y con un **umbral de decisión de {UMBRAL:.2f}**. "
        f"Se entrenó con {meta['n_entrenamiento']} registros y se evaluó con {meta['n_prueba']} que no participaron "
        "en el entrenamiento. La aplicación carga el pipeline guardado; no vuelve a entrenarlo.")

    st.markdown("**Métricas finales en el conjunto de prueba**")
    cols = st.columns(6)
    for col, (nombre, clave, ayuda) in zip(cols, [
            ("Accuracy", "Accuracy", "Proporción total de aciertos."),
            ("Recall (diabetes)", "Recall (1)", "De las pacientes con diabetes, proporción que el modelo detecta. Es la métrica prioritaria."),
            ("Precision (diabetes)", "Precision (1)", "De las pacientes marcadas con riesgo, proporción que realmente tiene diabetes."),
            ("F1 (diabetes)", "F1 (1)", "Media armónica de precision y recall."),
            ("Especificidad", "Especificidad", "De las pacientes sin diabetes, proporción clasificada correctamente."),
            ("ROC-AUC", "ROC-AUC", "Capacidad de ordenar a las pacientes por riesgo (0.5 = azar, 1 = perfecto).")]):
        col.metric(nombre, f"{mp[clave]:.3f}", help=ayuda)
    st.caption(f"Validación cruzada de 5 pliegues (entrenamiento): recall {cvm['recall (media)']:.3f} ± "
               f"{cvm['recall (desv.)']:.3f}, F1 {cvm['f1 (media)']:.3f} ± {cvm['f1 (desv.)']:.3f}, "
               f"accuracy {cvm['accuracy (media)']:.3f}.")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Matriz de confusión (prueba, n = 154)**")
        mat = pd.DataFrame([
            {"Real": "Sin diabetes", "Predicha": "Sin diabetes", "n": mc["VN"], "tipo": "Verdadero negativo"},
            {"Real": "Sin diabetes", "Predicha": "Con diabetes", "n": mc["FP"], "tipo": "Falso positivo (falsa alarma)"},
            {"Real": "Con diabetes", "Predicha": "Sin diabetes", "n": mc["FN"], "tipo": "Falso negativo (no detectada)"},
            {"Real": "Con diabetes", "Predicha": "Con diabetes", "n": mc["VP"], "tipo": "Verdadero positivo"}])
        base = alt.Chart(mat).encode(
            x=alt.X("Predicha:N", sort=["Sin diabetes", "Con diabetes"], title="Clase predicha", axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("Real:N", sort=["Sin diabetes", "Con diabetes"], title="Clase real"))
        calor = base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="teals"), legend=None),
                                        tooltip=["tipo", alt.Tooltip("n:Q", title="Pacientes")])
        numeros = base.mark_text(fontSize=22, fontWeight="bold").encode(
            text="n:Q", color=alt.condition(alt.datum.n > 40, alt.value("white"), alt.value("#1F2937")))
        st.altair_chart((calor + numeros).properties(height=300), width="stretch")
        st.markdown(f"De 54 pacientes con diabetes, el modelo detecta **{mc['VP']}** y deja sin detectar "
                    f"**{mc['FN']}** (falsos negativos). De 100 pacientes sin diabetes, clasifica bien "
                    f"**{mc['VN']}** y genera **{mc['FP']}** falsas alarmas.")
    with c2:
        st.markdown("**Antes y después de la revisión del modelo (prueba)**")
        comp = pd.DataFrame({
            "Métrica": ["Accuracy", "Recall (diabetes)", "Precision (diabetes)", "F1 (diabetes)", "ROC-AUC",
                        "Falsos negativos", "Falsos positivos", "Brecha F1 entrenamiento–prueba"],
            "Segundo avance": [antes["Accuracy"], antes["Recall (1)"], antes["Precision (1)"], antes["F1 (1)"],
                               antes["ROC-AUC"], antes["FN"], antes["FP"], antes["Brecha F1"]],
            "Modelo final": [mp["Accuracy"], mp["Recall (1)"], mp["Precision (1)"], mp["F1 (1)"], mp["ROC-AUC"],
                             mc["FN"], mc["FP"], meta["brecha_f1_final"]]})
        st.dataframe(comp.style.format({"Segundo avance": "{:g}", "Modelo final": "{:g}"}), hide_index=True, width="stretch")
        st.markdown(
            "En el tercer corte se regularizó el random forest (búsqueda de hiperparámetros con validación cruzada) "
            "para reducir el sobreajuste y se bajó el umbral de decisión de 0.50 a 0.40 para detectar más casos de "
            "diabetes. **El costo:** más falsas alarmas y una precision algo menor. Se aceptó porque en este "
            "problema no detectar un caso es más grave que pedir una prueba confirmatoria.")

    st.subheader("Limitaciones")
    st.markdown(
        f"- **Falsas alarmas frecuentes:** la precision es de {mp['Precision (1)']:.3f}; aproximadamente 4 de cada 10 "
        "pacientes marcadas con riesgo no tienen diabetes.\n"
        f"- **Casos no detectados:** el modelo deja pasar {mc['FN']} de 54 pacientes con diabetes en la prueba, sobre "
        "todo pacientes jóvenes con glucosa intermedia.\n"
        f"- **Señal del 70 %:** el accuracy ({mp['Accuracy']:.3f}) y el recall ({mp['Recall (1)']:.3f}) superan 0.70, "
        f"pero el F1 ({mp['F1 (1)']:.3f}) y la precision no.\n"
        "- **Datos imputados:** en el dataset falta el 48.7 % de los valores de insulina y el 29.6 % del pliegue "
        "cutáneo; se reemplazan por la mediana, lo que limita la información que aportan.\n"
        "- **Población específica y muestra pequeña:** 768 mujeres de 21 años o más de herencia Pima. Los resultados "
        "no se pueden generalizar a otras poblaciones (hombres, menores, otros grupos) sin validación externa.\n"
        "- **Asociación, no causalidad:** el modelo identifica patrones estadísticos; no explica causas.\n"
        "- **Uso:** prototipo académico. No es un dispositivo médico y no sustituye la decisión de un profesional.")

st.divider()
st.caption("GlucoLens · Proyecto de aula de Minería de Datos · Universidad de la Costa · Isaac Mendoza Herrera · "
           "Prototipo académico: no sustituye una decisión profesional.")
