import streamlit as st
import pickle
import re

st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="💬",
    layout="centered"
)

# =========================
# Cargar modelo y vectorizador
# =========================
modelo = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizador = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# =========================
# Limpieza de texto (MISMA lógica que en el entrenamiento)
# =========================
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# =========================
# Interfaz
# =========================
st.markdown(
    "<h1 style='text-align:center;'>💬 Análisis de Sentimientos</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Ingrese una reseña de película y conozca su sentimiento</p>",
    unsafe_allow_html=True
)

texto = st.text_area(
    "Texto a analizar",
    height=150,
    placeholder="Escriba aquí su texto..."
)

# =========================
# Predicción
# =========================
if st.button("Analizar sentimiento"):

    if texto.strip() == "":
        st.warning("Por favor, ingrese un texto.")
    else:
        texto_limpio = limpiar_texto(texto)
        vector = vectorizador.transform([texto_limpio])

        # Obtener probabilidades
        proba = modelo.predict_proba(vector)[0]
        prob_neg = proba[0]
        prob_pos = proba[1]

        # Decisión basada en la MAYOR probabilidad
        if prob_pos > prob_neg:
            st.success("😊 Sentimiento POSITIVO")
        else:
            st.error("😞 Sentimiento NEGATIVO")

        # Mostrar probabilidades
        st.markdown("### Probabilidades")
        st.write(f"🔴 Negativo: **{prob_neg:.2f}**")
        st.write(f"🟢 Positivo: **{prob_pos:.2f}**")
