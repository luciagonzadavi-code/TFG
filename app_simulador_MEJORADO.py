"""
Simulador de Precio Óptimo — Airbnb Madrid
TFG Business Analytics · Lucía González Dávila · UFV 2025/2026
"""

# ── 0. PAQUETES ────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

# ── 1. CONFIGURACIÓN ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador Airbnb Madrid",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colores corporativos
C_RED   = "#E84855"
C_BLUE  = "#2E75B6"
C_GREEN = "#2D6A4F"
C_GRAY  = "#6B7280"

# ── 2. CABECERA ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, {C_BLUE}, #1a4a7a);
     padding: 24px 32px; border-radius: 12px; margin-bottom: 24px;">
  <h1 style="color:white; margin:0; font-size:28px;">🏠 Simulador de Precio Óptimo — Airbnb Madrid</h1>
  <p style="color:#c8dcf0; margin:6px 0 0 0; font-size:14px;">
    Herramienta de revenue management basada en modelos econométricos y datos reales de Inside Airbnb (sept. 2025)
  </p>
</div>
""", unsafe_allow_html=True)

# ── 3. CARGA DEL MODELO ────────────────────────────────────────────────────
@st.cache_resource
def load_model ():
    url = "https://drive.google.com/uc?id=1xG_AsfpgGYUV977EDECXyDq37kN-ZY6U"
    response = requests.get(url)
    return pickle.loads(response.content)
model_occ_entire_log = load_model()

# ── 4. DATOS DE REFERENCIA DEL MERCADO (de listings_enriched) ─────────────
# Estadísticos reales por barrio calculados en el notebook
BARRIO_STATS = {
    "Sol":           {"revpar_med": 54.28, "precio_med": 150, "n": 893},
    "Recoletos":     {"revpar_med": 62.38, "precio_med": 179, "n": 178},
    "Cortes":        {"revpar_med": 58.14, "precio_med": 142, "n": 692},
    "Justicia":      {"revpar_med": 44.78, "precio_med": 138, "n": 720},
    "Almagro":       {"revpar_med": 45.76, "precio_med": 148, "n": 66},
    "Embajadores":   {"revpar_med": 38.50, "precio_med": 120, "n": 850},
    "Universidad":   {"revpar_med": 36.20, "precio_med": 115, "n": 620},
    "Palacio":       {"revpar_med": 40.10, "precio_med": 130, "n": 540},
    "Niño Jesús":    {"revpar_med": 59.18, "precio_med": 139, "n": 44},
    "Palos de Moguer":{"revpar_med": 43.67, "precio_med": 109, "n": 248},
}
REVPAR_MADRID_MED = 31.44   # mediana global del mercado
PRECIO_MADRID_MED = 111.0   # mediana global del mercado
OCUP_MADRID_MED   = 0.437   # ocupación real media (calendar.csv)

# Segmentos K-means (centroides del análisis)
SEGMENTOS = {
    "C1 — Premium":                {"precio": 294, "revpar": 127.2, "ocup": 0.417, "capacidad": 6.9, "color": C_RED},
    "C2 — Calidad media-alta":     {"precio": 115, "revpar": 76.2,  "ocup": 0.507, "capacidad": 3.4, "color": C_BLUE},
    "C3 — Económico alta ocup.":   {"precio": 95,  "revpar": 20.8,  "ocup": 0.783, "capacidad": 2.9, "color": C_GREEN},
    "C4 — Bajo rendimiento":       {"precio": 103, "revpar": 16.6,  "ocup": 0.177, "capacidad": 3.0, "color": C_GRAY},
}

TODOS_BARRIOS = [
    "Sol","Universidad","Embajadores","Cortes","Chopera","Valdefuentes","Palacio","Justicia",
    "Niño Jesús","Concepción","Recoletos","Palos de Moguer","San Diego","Simancas","Timón",
    "Marroquina","Goya","Almenara","San Andrés","Almagro","Nueva España","Rios Rosas","Valverde",
    "Lista","Fontarrón","Berruguete","Trafalgar","Canillas","Argüelles","Ibiza","Guindalera",
    "Castilla","Castellana","Jerónimos","San Juan Bautista","Aravaca","Ciudad Jardín","Cármenes",
    "Arapiles","Hispanoamérica","Imperial","Casa de Campo","Quintana","Lucero","Castillejos",
    "Puerta del Angel","Acacias","Delicias","El Viso","Campamento","Piovera","Costillares",
    "Ventas","Pueblo Nuevo","Fuente del Berro","Adelfas","Los Angeles","Almendrales","Pacífico",
    "Gaztambide","Prosperidad","Valdeacederas","Rejas","San Isidro","Cuatro Caminos",
    "San Cristobal","Moscardó","Santa Eugenia","Comillas","Hellín","Pilar","Bellas Vistas",
    "Atocha","Casco Histórico de Vallecas","Numancia","Peñagrande","Cuatro Vientos","Legazpi",
    "Vallehermoso","Pinar del Rey","Orcasur","Corralejos","Mirasierra","Colina","Opañel",
    "Salvador","Portazgo","Casco Histórico de Barajas","Estrella","Aguilas","Rosas",
    "Vista Alegre","Los Rosales","Casco Histórico de Vicálvaro","Buenavista","Canillejas",
    "Palomeras Bajas","La Paz","Puerta Bonita","Apostol Santiago","San Fermín","Vinateros",
    "San Pascual","Aluche","El Goloso","Valdezarza","Pradolongo","Pavones","Entrevías","Arcos",
    "Zofío","Ciudad Universitaria","Abrantes","Amposta","Palomeras Sureste","Orcasitas",
    "Fuentelareina","Alameda de Osuna","Palomas","Media Legua","Butarque","Ambroz",
    "Valdemarín","Aeropuerto","El Plantío"
]

# ── 5. SIDEBAR — INPUTS ────────────────────────────────────────────────────
st.sidebar.markdown("## 🏠 Características del alojamiento")
st.sidebar.markdown("---")

neighbourhood        = st.sidebar.selectbox("📍 Barrio", TODOS_BARRIOS)
accommodates         = st.sidebar.slider("👥 Capacidad (personas)", 1, 10, 4)
bedrooms             = st.sidebar.slider("🛏️ Habitaciones", 0, 6, 2)
amenities_count      = st.sidebar.slider("🔢 Número de cualidades", 0, 80, 25)
has_wifi             = st.sidebar.selectbox("🛜 WiFi", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
review_scores_rating = st.sidebar.slider("⭐ Puntuación media", 1.0, 5.0, 4.7, 0.1)
host_is_superhost    = st.sidebar.selectbox("🏅 Superhost", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
reviews_last_12m     = st.sidebar.slider("📝 Reseñas últimos 12 meses", 0, 100, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("## 💰 Precio")
precio_actual   = st.sidebar.number_input("Precio actual (€/noche)", min_value=20, max_value=600, value=120)
price_min, price_max = st.sidebar.slider("Rango de simulación (€)", 40, 600, (40, 400))

# ── 6. SIMULACIÓN ─────────────────────────────────────────────────────────
@st.cache_data
def simular(neighbourhood, accommodates, bedrooms, amenities_count, has_wifi,
            review_scores_rating, host_is_superhost, reviews_last_12m,
            price_min, price_max):
    price_range = np.linspace(price_min, price_max, 200)
    resultados = []
    for p in price_range:
        fila = pd.DataFrame({
            "price":                 [p],
            "log_price":             [np.log(p)],
            "accommodates":          [accommodates],
            "bedrooms":              [bedrooms],
            "amenities_count":       [amenities_count],
            "has_wifi":              [has_wifi],
            "bathrooms_missing":     [0],
            "review_scores_rating":  [review_scores_rating],
            "host_is_superhost":     [host_is_superhost],
            "neighbourhood_cleansed":[neighbourhood],
            "reviews_last_12m":      [reviews_last_12m],
        })
        occ = float(model_occ_entire_log.predict(fila).iloc[0])
        occ = max(0.0, min(1.0, occ))
        resultados.append({"Precio": p, "Ocupación": occ, "RevPAR": p * occ})
    return pd.DataFrame(resultados)

df_sim  = simular(neighbourhood, accommodates, bedrooms, amenities_count, has_wifi,
                  review_scores_rating, host_is_superhost, reviews_last_12m,
                  price_min, price_max)
optimo  = df_sim.loc[df_sim["RevPAR"].idxmax()]
diferencia  = optimo["Precio"] - precio_actual
porcentaje  = (diferencia / precio_actual) * 100

# ── 7. CLASIFICACIÓN EN SEGMENTO K-MEANS ──────────────────────────────────
def clasificar_segmento(precio, ocupacion, accommodates):
    """Clasifica el perfil del usuario en el segmento más cercano."""
    mejor, dist_min = None, float("inf")
    for nombre, c in SEGMENTOS.items():
        # Distancia euclidiana normalizada en precio, ocupación y capacidad
        d = ((precio - c["precio"]) / 300)**2 + \
            ((ocupacion - c["ocup"])  / 0.5)**2 + \
            ((accommodates - c["capacidad"]) / 5)**2
        if d < dist_min:
            dist_min, mejor = d, nombre
    return mejor

segmento_usuario = clasificar_segmento(
    optimo["Precio"], optimo["Ocupación"], accommodates)

# ── 8. MÉTRICAS PRINCIPALES ────────────────────────────────────────────────
st.markdown("### 📊 Resultados de la simulación")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🎯 Precio óptimo",     f"{optimo['Precio']:.0f} €",
          delta=f"{diferencia:+.0f} € vs actual")
c2.metric("💵 Precio actual",     f"{precio_actual:.0f} €")
c3.metric("🏨 Ocupación estimada",f"{optimo['Ocupación']:.1%}")
c4.metric("💰 RevPAR máximo",     f"{optimo['RevPAR']:.2f} €")
revpar_barrio = BARRIO_STATS.get(neighbourhood, {}).get("revpar_med", REVPAR_MADRID_MED)
delta_mercado = optimo["RevPAR"] - revpar_barrio
c5.metric("📍 vs mercado barrio", f"{delta_mercado:+.2f} €",
          delta=f"mediana barrio: {revpar_barrio:.1f} €",
          delta_color="normal")

st.markdown("---")

# ── 9. GRÁFICOS ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### Relación Precio – RevPAR y Ocupación")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.patch.set_facecolor("white")

    # RevPAR
    ax1.plot(df_sim["Precio"], df_sim["RevPAR"], color=C_BLUE, linewidth=2.5, label="RevPAR estimado")
    ax1.axvline(optimo["Precio"],  color=C_RED,   linestyle="--", linewidth=1.5, label=f"Precio óptimo ({optimo['Precio']:.0f} €)")
    ax1.axvline(precio_actual,     color=C_GRAY,  linestyle=":",  linewidth=1.5, label=f"Precio actual ({precio_actual:.0f} €)")
    ax1.axhline(revpar_barrio,     color=C_GREEN, linestyle="-.", linewidth=1.2, alpha=0.8, label=f"Mediana barrio ({revpar_barrio:.1f} €)")
    ax1.scatter(optimo["Precio"],  optimo["RevPAR"], s=100, color=C_RED, zorder=5)
    ax1.set_ylabel("RevPAR (€)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Ocupación
    ax2.plot(df_sim["Precio"], df_sim["Ocupación"] * 100, color=C_GREEN, linewidth=2.5)
    ax2.axvline(optimo["Precio"], color=C_RED,  linestyle="--", linewidth=1.5)
    ax2.axvline(precio_actual,    color=C_GRAY, linestyle=":",  linewidth=1.5)
    ax2.axhline(OCUP_MADRID_MED * 100, color=C_GRAY, linestyle="-.", linewidth=1.2,
                alpha=0.8, label=f"Ocup. media Madrid ({OCUP_MADRID_MED:.1%})")
    ax2.set_ylabel("Ocupación real (%)")
    ax2.set_xlabel("Precio (€/noche)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Fuente: elaboración propia. Modelo OLS sobre occupancy_rate_cal (calendar.csv, Inside Airbnb sept. 2025)")

with col_right:
    # Posicionamiento vs mercado
    st.markdown("#### Posicionamiento vs mercado")
    barrio_info = BARRIO_STATS.get(neighbourhood, None)

    precio_med_ref   = barrio_info["precio_med"] if barrio_info else PRECIO_MADRID_MED
    revpar_med_ref   = barrio_info["revpar_med"] if barrio_info else REVPAR_MADRID_MED
    ref_label        = neighbourhood if barrio_info else "Madrid (global)"

    fig2, ax = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("white")

    # Scatter: precio actual del usuario
    ax.scatter(precio_actual, optimo["Ocupación"] * 100,
               s=200, color=C_RED, zorder=5, label="Tu alojamiento (precio actual)")
    ax.scatter(optimo["Precio"], optimo["Ocupación"] * 100,
               s=200, color=C_BLUE, zorder=5, marker="*", label="Precio óptimo")
    ax.scatter(precio_med_ref, OCUP_MADRID_MED * 100,
               s=120, color=C_GRAY, zorder=4, marker="D", label=f"Mediana {ref_label}")

    # Cuadrantes
    ax.axvline(precio_med_ref,     color="lightgray", linestyle="--", linewidth=1)
    ax.axhline(OCUP_MADRID_MED*100, color="lightgray", linestyle="--", linewidth=1)

    # Etiquetas cuadrantes
    ax.text(0.02, 0.97, "Alta ocup.\nPrecio bajo",  transform=ax.transAxes, fontsize=7,
            va="top", color=C_GRAY)
    ax.text(0.98, 0.97, "Alta ocup.\nPrecio alto",  transform=ax.transAxes, fontsize=7,
            va="top", ha="right", color=C_GREEN)
    ax.text(0.02, 0.03, "Bajo rend.\nPrecio bajo",  transform=ax.transAxes, fontsize=7,
            va="bottom", color=C_GRAY)
    ax.text(0.98, 0.03, "Precio alto\nOcup. baja",  transform=ax.transAxes, fontsize=7,
            va="bottom", ha="right", color=C_RED)

    ax.set_xlabel("Precio (€/noche)")
    ax.set_ylabel("Ocupación real estimada (%)")
    ax.set_title(f"Posición en el mercado\n({ref_label})", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

    # Segmento K-means
    st.markdown("#### Tu segmento de mercado")
    seg_info = SEGMENTOS[segmento_usuario]
    color_seg = seg_info["color"]
    st.markdown(f"""
    <div style="background:{color_seg}18; border-left: 4px solid {color_seg};
         padding: 12px 16px; border-radius: 6px;">
      <b style="color:{color_seg}; font-size:15px;">{segmento_usuario}</b><br>
      <small style="color:#444;">
        Precio mediano del segmento: <b>{seg_info['precio']} €</b> &nbsp;·&nbsp;
        RevPAR mediano: <b>{seg_info['revpar']} €</b><br>
        Ocupación media: <b>{seg_info['ocup']:.1%}</b> &nbsp;·&nbsp;
        Capacidad media: <b>{seg_info['capacidad']:.1f} personas</b>
      </small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── 10. TABLA DE ESCENARIOS ────────────────────────────────────────────────
st.markdown("### 📋 Comparativa de escenarios de precio")

# Calcular tres escenarios
precio_bajo  = price_min + (price_max - price_min) * 0.15
precio_opt   = optimo["Precio"]
precio_alto  = price_min + (price_max - price_min) * 0.85

def predecir_fila(p):
    fila = pd.DataFrame({
        "price": [p], "log_price": [np.log(p)],
        "accommodates": [accommodates], "bedrooms": [bedrooms],
        "amenities_count": [amenities_count], "has_wifi": [has_wifi],
        "bathrooms_missing": [0], "review_scores_rating": [review_scores_rating],
        "host_is_superhost": [host_is_superhost],
        "neighbourhood_cleansed": [neighbourhood],
        "reviews_last_12m": [reviews_last_12m],
    })
    occ = float(model_occ_entire_log.predict(fila).iloc[0])
    return max(0.0, min(1.0, occ))

escenarios = []
for label, precio, emoji in [
    ("Precio bajo (máx. ocupación)", precio_bajo,  "📉"),
    ("Precio óptimo (máx. RevPAR)",  precio_opt,   "🎯"),
    ("Precio alto (margen premium)",  precio_alto,  "💎"),
    ("Precio actual",                precio_actual,"📌"),
]:
    occ   = predecir_fila(precio)
    revpar = precio * occ
    ingresos_mes = revpar * 30
    escenarios.append({
        "Escenario":            f"{emoji} {label}",
        "Precio (€/noche)":    f"{precio:.0f} €",
        "Ocupación estimada":  f"{occ:.1%}",
        "RevPAR (€)":          f"{revpar:.2f} €",
        "Ingreso est. mensual":f"{ingresos_mes:.0f} €",
    })

df_esc = pd.DataFrame(escenarios)

def highlight_optimo(row):
    if "óptimo" in row["Escenario"]:
        return [f"background-color: {C_BLUE}20; font-weight: bold"] * len(row)
    if "actual" in row["Escenario"]:
        return [f"background-color: {C_GRAY}15"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_esc.style.apply(highlight_optimo, axis=1),
    use_container_width=True, hide_index=True
)
st.caption("El ingreso mensual estimado asume 30 días disponibles. Es una estimación orientativa.")

st.markdown("---")

# ── 11. RECOMENDACIÓN DE NEGOCIO ──────────────────────────────────────────
st.markdown("### 💡 Recomendación estratégica")

col_rec1, col_rec2 = st.columns([1, 1])

with col_rec1:
    # Recomendación general según precio óptimo
    st.markdown("#### Estrategia general")
    if optimo["Precio"] > 300:
        st.success(f"""
**Estrategia premium: priorizar margen sobre volumen**

Su alojamiento ({neighbourhood}, {accommodates} personas) tiene características
que soportan precios elevados sin una caída significativa en la ocupación.

Precio óptimo estimado: **{optimo['Precio']:.0f} €/noche** | RevPAR: **{optimo['RevPAR']:.2f} €**

Recomendaciones:
- Refuerce la propuesta de valor con fotografías profesionales y descripción detallada
- Considere el programa Smart Pricing de Airbnb para ajuste dinámico
- Monitorice la competencia en {neighbourhood} (mediana del barrio: {revpar_barrio:.1f} € RevPAR)
""")
    elif optimo["Precio"] > 150:
        st.info(f"""
**Estrategia equilibrada: precio competitivo con ocupación estable**

Su alojamiento se sitúa en el segmento medio del mercado, donde el RevPAR
se maximiza combinando precio moderado con alta ocupación.

Precio óptimo estimado: **{optimo['Precio']:.0f} €/noche** | RevPAR: **{optimo['RevPAR']:.2f} €**

Recomendaciones:
- Ajuste el precio según la temporada (pico en julio-septiembre en Madrid)
- Aumente el número de servicios (amenities_count actual: {amenities_count}) para justificar subidas de precio
- El segmento *{segmento_usuario}* tiene un RevPAR mediano de {seg_info['revpar']} € — tiene margen de mejora
""")
    else:
        st.warning(f"""
**Estrategia de volumen: maximizar ocupación**

El rendimiento depende principalmente del volumen de reservas.
El alojamiento tiene potencial de mejora en precio sin sacrificar ocupación.

Precio óptimo estimado: **{optimo['Precio']:.0f} €/noche** | RevPAR: **{optimo['RevPAR']:.2f} €**

Recomendaciones:
- Mejore la puntuación (rating actual: {review_scores_rating}) para acceder al badge Superhost
- Aumente el número de servicios básicos (WiFi, aire acondicionado, lavadora)
- Considere subir el precio gradualmente (+10%) y observar el impacto en la ocupación
""")

with col_rec2:
    # Recomendación personalizada precio actual vs óptimo
    st.markdown("#### Ajuste de precio recomendado")
    if diferencia > 10:
        st.success(f"""
**Su precio actual está por debajo del óptimo**

- Precio actual: **{precio_actual:.0f} €**
- Precio óptimo: **{optimo['Precio']:.0f} €**
- Ajuste recomendado: **+{diferencia:.0f} € (+{porcentaje:.1f}%)**

Subir el precio hasta el nivel óptimo podría aumentar
su RevPAR en aproximadamente **{optimo['RevPAR'] - precio_actual * predecir_fila(precio_actual):.2f} €/día**
sin reducir significativamente la ocupación.
""")
    elif abs(diferencia) <= 10:
        st.info(f"""
**Su precio está próximo al nivel óptimo**

- Precio actual: **{precio_actual:.0f} €**
- Precio óptimo: **{optimo['Precio']:.0f} €**
- Diferencia: **{diferencia:+.0f} €**

El alojamiento tiene un equilibrio adecuado entre precio
y ocupación. Mantenga la estrategia actual y realice
pequeños ajustes estacionales para optimizar el RevPAR.
""")
    else:
        st.warning(f"""
**Su precio está por encima del nivel óptimo**

- Precio actual: **{precio_actual:.0f} €**
- Precio óptimo: **{optimo['Precio']:.0f} €**
- Ajuste recomendado: **{diferencia:.0f} € ({porcentaje:.1f}%)**

Reducir el precio podría mejorar significativamente
la ocupación y aumentar el RevPAR total estimado.
""")

    # Comparativa vs mercado del barrio
    st.markdown("#### Posición vs mercado")
    if delta_mercado > 0:
        st.success(f"✅ Su RevPAR óptimo ({optimo['RevPAR']:.1f} €) supera la mediana del barrio {neighbourhood} ({revpar_barrio:.1f} €) en **+{delta_mercado:.1f} €**.")
    elif delta_mercado > -10:
        st.info(f"📊 Su RevPAR óptimo ({optimo['RevPAR']:.1f} €) está próximo a la mediana del barrio {neighbourhood} ({revpar_barrio:.1f} €).")
    else:
        st.warning(f"⚠️ Su RevPAR óptimo ({optimo['RevPAR']:.1f} €) está por debajo de la mediana del barrio {neighbourhood} ({revpar_barrio:.1f} €) en {delta_mercado:.1f} €. Revise las características del alojamiento.")

# ── 12. PIE ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:{C_GRAY}; font-size:12px; padding: 8px;">
    TFG Business Analytics · Lucía González Dávila · UFV 2025/2026 &nbsp;|&nbsp;
    Modelo basado en Inside Airbnb Madrid (sept. 2025) &nbsp;|&nbsp;
    Las estimaciones son orientativas y no constituyen asesoramiento financiero.
</div>
""", unsafe_allow_html=True)
