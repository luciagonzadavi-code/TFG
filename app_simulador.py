#0.Carga de paquetes necesarios:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests


#1.Texto de la aplicación:
st.set_page_config(page_title="Simulador Airbnb", layout="wide")
st.title("Simulador de Precio Óptimo - Airbnb Madrid")
st.markdown("Herramienta de apoyo a la decisión basada en modelos econométricos.")

#2.Carga del modelo:
@st.cache_resource
def load_model ():
    url = "https://drive.google.com/uc?id=1xG_AsfpgGYUV977EDECXyDq37kN-ZY6U"
    response = requests.get(url)
    return pickle.loads(response.content)
model_occ_entire_log = load_model()

#3.Definición de filtros:
st.sidebar.header("Características del alojamiento")

accommodates = st.sidebar.slider("Capacidad", 1, 10, 4)
bedrooms = st.sidebar.slider("Habitaciones", 0, 6, 2)
amenities_count = st.sidebar.slider("Amenities", 0, 80, 25)
has_wifi = st.sidebar.selectbox("WiFi", [1, 0], format_func=lambda x: "Sí" if x==0 else "No")
review_scores_rating = st.sidebar.slider("Rating", 0.0, 5.0, 4.7, 0.1)
host_is_superhost = st.sidebar.selectbox("Superhost", [1, 0], format_func=lambda x: "Sí" if x==1 else "No")
neighbourhood = st.sidebar.selectbox("Barrio", ["Sol", "Universidad", "Embajadores", "Cortes", "Chopera", "Valdefuentes",
"Palacio", "Justicia", "Niño Jesús", "Concepción", "Recoletos",
"Palos de Moguer", "San Diego", "Simancas", "Timón", "Marroquina", "Goya",
"Almenara", "San Andrés", "Almagro", "Nueva España", "Rios Rosas", "Valverde",
"Lista", "Fontarrón", "Berruguete", "Trafalgar", "Canillas", "Argüelles",
"Ibiza", "Guindalera", "Castilla", "Castellana", "Jerónimos",
"San Juan Bautista", "Aravaca", "Ciudad Jardín", "Cármenes", "Arapiles",
"Hispanoamérica", "Imperial", "Casa de Campo", "Quintana", "Lucero",
"Castillejos", "Puerta del Angel", "Acacias", "Delicias", "El Viso",
"Campamento", "Piovera", "Costillares", "Ventas", "Pueblo Nuevo",
"Fuente del Berro", "Adelfas", "Los Angeles", "Almendrales", "Pacífico",
"Gaztambide", "Prosperidad", "Valdeacederas", "Rejas", "San Isidro",
"Cuatro Caminos", "San Cristobal", "Moscardó", "Santa Eugenia", "Comillas",
"Hellín", "Pilar", "Bellas Vistas", "Atocha", "Casco Histórico de Vallecas",
"Numancia", "Peñagrande", "Cuatro Vientos", "Legazpi", "Vallehermoso",
"Pinar del Rey", "Orcasur", "Corralejos", "Mirasierra", "Colina", "Opañel",
"Salvador", "Portazgo", "Casco Histórico de Barajas", "Estrella", "Aguilas",
"Rosas", "Vista Alegre", "Los Rosales", "Casco Histórico de Vicálvaro",
"Buenavista", "Canillejas", "Palomeras Bajas", "La Paz", "Puerta Bonita",
"Apostol Santiago", "San Fermín", "Vinateros", "San Pascual", "Aluche",
"El Goloso", "Valdezarza", "Pradolongo", "Pavones", "Entrevías", "Arcos",
"Zofío", "Ciudad Universitaria", "Abrantes", "Amposta", "Palomeras Sureste",
"Orcasitas", "Fuentelareina", "Alameda de Osuna", "Palomas", "Media Legua",
"Butarque", "Ambroz", "Valdemarín", "Aeropuerto", "El Plantío"])

price_min, price_max = st.sidebar.slider("Rango de precios", 40, 600, (40, 600))
precio_actual = st.sidebar.number_input("inserte el precio actual del alojamiento", min_value = 20, value = 120)

#4.Simulación
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
diferencia = optimo["Precio"] - precio_actual
porcentaje = (diferencia / precio_actual) * 100

#5. Obtención de resultados:
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio óptimo", f"{optimo['Precio']:.2f} €")
col2.metric("Precio actual", f"{precio_actual:.2f} €")
col3.metric("Ocupación estimada", f"{optimo['Ocupación']:.1%}")
col4.metric("RevPAR máximo", f"{optimo['RevPAR']:.2f} €")


#6. Graficación
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(resultados["Precio"], resultados["RevPAR"], label = "RevPAR estimado")
ax.axvline(optimo["Precio"], linestyle="--", label = "Precio óptimo")
ax.axvline(precio_actual, linestyle=":", label="Precio actual")
ax.scatter(optimo["Precio"], optimo["RevPAR"], s=100)
ax.set_xlabel("Precio (€)")
ax.set_ylabel("RevPAR estimado")
ax.set_title("Precio - RevPAR estimado")
ax.grid(alpha=0.3)
st.pyplot(fig)


#7. Interpretación de los resultados:
#7.1 Recomendaciones de negocio generales:
st.subheader("Recomendación de negocio")
if optimo["Precio"] > 300:
    st.success("""Estrategia a seguir: Priorizar el margen sobre volumen.
               El alojamiento presenta características que permiten rangos de precios elevados sin resultar en una caida significativa en la ocupación, pero siempre reforzando la propuesta de valor a través de la calidad, la experiencia del cliente y la diferenciación frente a la competencia.
               """)
elif optimo["Precio"] > 150:
    st.info("""Estrategia a seguir: Equilibrio entre precio y opcupación.
            El alojamiento se sitúa en un rango intermedio donde el rendimiento se maximiza el combinar un nivel de precio competitivo con una ocupación estable, para conseguir esta rentabilidad, es recomendable mantener una estrategia flexible, ajustando el precio en función de la demanda para optimizar el RevPAR sin sacrificar el volumen de reservas.
            """)
else:
    st.warning("""Estrategia a seguir: Maximización de la opcupación
               El rendimiento del alojamiento depende en mayor medida del volumen de reservas que del margen por noche, por ello es recomendable mantener precios competitivos para asegurar una alta ocupación, especialmente si todavía se encuentra en proceso de integración o hay un alto grado de competencia.
               """)
#7.2 Recomendaciones personalizadas:
st.subheader("Recomendación estratégica personalizada")
if diferencia > 10:
    st.success(f"""
El precio actual ({precio_actual:.2f} €) se encuentra por debajo del nivel óptimo estimado.
Se recomienda aumentar el precio aproximadamente un {porcentaje:.1f}% hasta situarlo en torno a {optimo['Precio']:.2f} €.
Este ajuste permitiría incrementar el RevPAR sin una reducción significativa de la ocupación.
""")

elif abs(diferencia) <= 10:
    st.info(f"""
El precio actual ({precio_actual:.2f} €) se encuentra próximo al nivel óptimo.
Se recomienda mantener la estrategia actual, realizando pequeños ajustes en función de la demanda para optimizar el rendimiento.
El alojamiento cuneta con un equilibrio adecuado entre precio y ocupación.
""")

else:
    st.warning(f"""
El precio actual ({precio_actual:.2f} €) se encuentra por encima del nivel óptimo estimado.
Se recomienda reducir el precio aproximadamente un {abs(porcentaje):.1f}% hasta situarlo en torno a {optimo['Precio']:.2f} €.
Este ajuste permitiría mejorar la ocupación estimada y aumentar el rendimiento económico total.
""")

st.markdown("""
**Nota:** Esta recomendación se basa en estimaciones del modelo estimado y debe interpretarse como una guía orientativa para la toma de decisiones.
""")
