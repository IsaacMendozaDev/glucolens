import sys, pandas as pd, numpy as np
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("1) La app carga sin errores. Pestañas:", [t.label for t in at.tabs])
def boton(label): return [b for b in at.button if b.label == label][0]
# 2) Caso normal (valores por defecto = medianas)
boton("Estimar riesgo").click().run()
assert not at.exception, at.exception
print("2) Caso normal ->", at.metric[[m.label for m in at.metric].index("Probabilidad estimada de diabetes")].value,
      "|", (at.success or at.error)[0].value[:60])
# 3) Entrada inválida: glucosa 500 y edad 15
at.number_input(key="in_Glucose").set_value(500.0); at.number_input(key="in_Age").set_value(15)
boton("Estimar riesgo").click().run()
errs = [e.value for e in at.error]
assert any("No se puede calcular" in e for e in errs), errs
assert "Probabilidad estimada de diabetes" not in [m.label for m in at.metric]
print("3) Entrada inválida ->", errs[0].replace("\n", " ")[:230])
# 4) Registros conocidos del conjunto de prueba
casos = pd.read_csv("casos_prueba.csv")
for i, r in casos.iterrows():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.selectbox[1].select_index(i).run()
    [b for b in at.button if b.label == "Cargar en el formulario"][0].click().run()
    [b for b in at.button if b.label == "Estimar riesgo"][0].click().run()
    assert not at.exception, at.exception
    prob = at.metric[[m.label for m in at.metric].index("Probabilidad estimada de diabetes")].value
    ver = [m.value for m in at.markdown if "Verificación con el notebook" in m.value][0]
    assert "✅" in ver, ver
    print(f"4) Registro {int(r['indice_original'])}: app {prob} | notebook clase {int(r['prediccion_notebook'])} prob {r['probabilidad_notebook']*100:.1f} % | faltantes: {int(r[list(casos.columns[1:9])].isna().sum())} -> coincide ✅")
