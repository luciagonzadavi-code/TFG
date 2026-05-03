import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Simulador Airbnb", layout="wide")

st.title("Simulador de Precio Óptimo - Airbnb Madrid")
st.markdown("Herramienta de apoyo a la decisión basada en modelos econométricos.")

# ========================
# CARGAR MODELO
# ========================
with open("model_occ_entire_log.pkl", "rb") as f:
    model_occ_entire_log = pickle.load(f)

# ========================
# SIDEBAR (INPUTS)
# ========================
st.sidebar.header("Características del alojamiento")

accommodates = st.sidebar.slider("Capacidad", 1, 10, 4)
bedrooms = st.sidebar.slider("Habitaciones", 0, 6, 2)
amenities_count = st.sidebar.slider("Amenities", 0, 80, 25)
has_wifi = st.sidebar.selectbox("WiFi", [1, 0], format_func=lambda x: "Sí" if x==1 else "No")
review_scores_rating = st.sidebar.slider("Rating", 0.0, 5.0, 4.7, 0.1)
host_is_superhost = st.sidebar.selectbox("Superhost", [1, 0], format_func=lambda x: "Sí" if x==1 else "No")

neighbourhood = st.sidebar.selectbox("Barrio", [
    "Sol","Palacio","Embajadores","Justicia","Universidad","Cortes"
])

price_min, price_max = st.sidebar.slider("Rango de precios", 40, 600, (40, 600))

# ========================
# SIMULACIÓN
# ========================
price_range = np.linspace(price_min, price_max, 100)
resultados = []

for p in price_range:
    fila = pd.DataFrame({
        "price": [p],
        "log_price": [np.log(p)],
        "accommodates": [accommodates],
        "bedrooms": [bedrooms],
        "amenities_count": [amenities_count],
        "has_wifi": [has_wifi],
        "bathrooms_missing": [0],
        "review_scores_rating": [review_scores_rating],
        "host_is_superhost": [host_is_superhost],
        "neighbourhood_cleansed": [neighbourhood]
    })

    occ = model_occ_entire_log.predict(fila).iloc[0]
    occ = max(0, min(1, occ))
    revpar = p * occ

    resultados.append({
        "Precio": p,
        "Ocupación": occ,
        "RevPAR": revpar
    })

resultados = pd.DataFrame(resultados)
optimo = resultados.loc[resultados["RevPAR"].idxmax()]

# ========================
# RESULTADOS
# ========================
col1, col2, col3 = st.columns(3)

col1.metric("Precio óptimo", f"{optimo['Precio']:.2f} €")
col2.metric("Ocupación", f"{optimo['Ocupación']:.1%}")
col3.metric("RevPAR", f"{optimo['RevPAR']:.2f} €")

# ========================
# GRÁFICO
# ========================
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(resultados["Precio"], resultados["RevPAR"])
ax.axvline(optimo["Precio"], linestyle="--")
ax.scatter(optimo["Precio"], optimo["RevPAR"], s=100)

ax.set_xlabel("Precio (€)")
ax.set_ylabel("RevPAR")
ax.set_title("Precio óptimo")
ax.grid(alpha=0.3)

st.pyplot(fig)

# ========================
# INTERPRETACIÓN
# ========================
st.subheader("Interpretación de negocio")

if optimo["Precio"] > 300:
    st.success("Estrategia premium: maximizar margen")
elif optimo["Precio"] > 150:
    st.info("Estrategia equilibrada")
else:
    st.warning("Estrategia de volumen")