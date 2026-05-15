import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("modelo_random_forest.pkl")
encoder = joblib.load("encoder_risco.pkl")

st.title("Sistema AIoT para Detecção de Risco de Contaminação")

st.write("Classificação inteligente de risco usando sensores IoT e Machine Learning.")

temperatura = st.number_input("Temperatura (°C)", min_value=0.0, max_value=50.0, value=25.0)
umidade = st.number_input("Umidade (%)", min_value=0.0, max_value=100.0, value=50.0)
voc = st.number_input("VOC", min_value=0.0, max_value=1000.0, value=100.0)
co = st.number_input("CO", min_value=0.0, max_value=20.0, value=2.0)
nox = st.number_input("NOx", min_value=0.0, max_value=500.0, value=100.0)
no2 = st.number_input("NO2", min_value=0.0, max_value=300.0, value=50.0)
particulas = st.number_input("Partículas", min_value=0.0, max_value=300.0, value=30.0)
tempo_uso = st.number_input("Tempo de uso da roupa/EPI (horas)", min_value=0.0, max_value=24.0, value=2.0)
area_critica = st.selectbox("Área crítica?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

entrada = pd.DataFrame([{
    "temperatura": temperatura,
    "umidade": umidade,
    "voc": voc,
    "co": co,
    "nox": nox,
    "no2": no2,
    "particulas": particulas,
    "tempo_uso": tempo_uso,
    "area_critica": area_critica
}])

if st.button("Classificar risco"):
    predicao = modelo.predict(entrada)
    risco = encoder.inverse_transform(predicao)[0]

    st.subheader("Resultado da classificação:")

    if risco == "baixo":
        st.success("Risco BAIXO")
    elif risco == "medio":
        st.warning("Risco MÉDIO")
    else:
        st.error("Risco ALTO")

    st.write("Dados analisados:")
    st.dataframe(entrada)