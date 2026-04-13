#Importación de los paquetes necesarios:
import pandas as pd
import ast
import numpy as np
#Lectura del CSV:
listings = pd.read_csv('/Users/luciagonzalezdavila/Documents/4º/TFG/BBDD airbnb/listings.csv')

#Exploraci�n inicial: 
#Filas y columnas: 
print("Filas y columnas iniciales:", listings.shape)

#Tipos de variables: 
print(listings.dtypes)

#Exploraci�n de las variables num�ricas: 
listings.describe()

#Nulos:
print(listings.isnull().sum().sort_values(ascending=False).head(20))

#Visualizacion de las variables y valores nulos:
null_counts = listings.isnull().sum().sort_values(ascending=False)
plt.figure(figsize=(12,18))
bars = plt.barh(null_counts.index[::-1], null_counts.values[::-1])
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + max(null_counts) * 0.01,         
        bar.get_y() + bar.get_height() / 2,
        f"{int(width)}",
        va="center",
        fontsize=8
    )
plt.xlabel("N�mero de valores nulos")
plt.ylabel("Variables")
plt.title("Variables iniciales y valores nulos")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

#Carga de las columnas:
cols = [
    "id",
    "host_id",
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "price",
    "minimum_nights",
    "maximum_nights",
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "reviews_per_month",
    "host_since",
    "host_is_superhost",
    "instant_bookable",
    "amenities",
    "calculated_host_listings_count",
    "availability_365",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d"
]


print(listings.columns.tolist())
print(len(listings.columns))

#REORGANIZACIÓN: 
#Filtrar columnas para que solo se quede con las que he mencionado: 
listings = listings[cols]
listings.describe 
#Renombrar columnas: 
listings = listings.rename(columns={"id": "listing_id"})

#Limpieza en precio: 
listings["price"] = listings["price"].replace("[\$,]", "", regex=True).astype(float)

#Limpieza en baños: 
#a)Convertir a nuemrico y si es raro en missing:
listings["bathrooms"] = pd.to_numeric(listings["bathrooms"], errors="coerce")
#b)Ver si faltaban datos:
listings["bathrooms_missing"] = listings["bathrooms"].isna().astype(int)
#c)Rellenar los valores faltanntes con 0 y usar la mediana (evita outliers, no distorsiona)
listings["bathrooms"] = listings["bathrooms"].fillna(listings["bathrooms"].median())

#Limpieza en anfitrión (binaria): 
listings["host_is_superhost"] = listings["host_is_superhost"].map({"t":1, "f":0}).fillna(0)

#Limpieza en booking(binaria): 
listings["instant_bookable"] = listings["instant_bookable"].map({"t":1, "f":0}).fillna(0)

#Limpieza en amenities (de semiestructutrado a variables cuantitativas): 
#a)Creación de una función que devuelve una lista vacia si no hay datos y convierte las string en listas reales sin errores:
def parse_amenities(x):
    if pd.isna(x) or x == "":
        return []
    try:
        val = ast.literal_eval(x)
        return val if isinstance(val, list) else []
    except:
        return []
#b)Aplicamos la función [conivierte la columna en listas realies]:
listings["amenities_list"] = listings["amenities"].apply(parse_amenities)
#c)Cuenta correctamente el número real:
listings["amenities_count"] = listings["amenities_list"].apply(len)
#d)Ver si tiene wifi:
listings["has_wifi"] = listings["amenities_list"].apply(lambda x: int("Wifi" in x))


#Limpieza en fecha alojamiento: 
listings["host_since"] = pd.to_datetime(listings["host_since"], errors="coerce")
#Fecha fija para que sea reproducible:
reference_date = pd.Timestamp("2025-12-01")
listings["host_tenure_days"] =  (reference_date - listings["host_since"]).dt.days


#CREACIÓN DE NUEVAS COLUMNAS:
#RevPAR
listings["revpar"] = listings["estimated_revenue_l365d"] / 365
#Ocupación:
listings["occupancy_rate"] = listings["estimated_occupancy_l365d"] / 365

#ADR (APX, solo chequeo):
listings["adr_check"] = listings["estimated_revenue_l365d"] / listings["estimated_occupancy_l365d"].replace(0, np.nan)
#Otras variables (derivativas): 
# Precio por persona (evitando división entre 0):
listings["accommodates"] = listings["accommodates"].replace(0, 1)
listings["price_per_person"] = listings["price"] / listings["accommodates"]
# Host profesional (considero profesional si tiene más de un alojamiento):
listings["professional_host"] = (listings["calculated_host_listings_count"] > 1).astype(int)

#LIMPIEZA:
#eliminacion de outliers de precio (el 0,5% mñas extremo, eliminando el subjetivismo de lo considerado "caro" y "barato"):
upper_price = listings["price"].quantile(0.995)
listings = listings[listings["price"] <= upper_price]

# Eliminación de casos sin beneficio o sin ocupación (así me quedo solo con los activos):
listings = listings[
    (listings["estimated_occupancy_l365d"] > 0) &
    (listings["estimated_revenue_l365d"] > 0)
]

#Grafico de outliers:
#Carga del paquete de visualización:
import matplotlib.pyplot as plt
#Guardar una copia antes del filtrado: 
listings_pre_filtrado = listings.copy()
#Umbral percentil 99.5:
upper_price = listings_pre_filtrado["price"].quantile(0.995)
#Aplicar filtro:
listings_post_filtrado = listings_pre_filtrado[listings_pre_filtrado["price"] <= upper_price].copy()
#Visualización:
plt.figure(figsize=(10,6))
plt.hist(listings_pre_filtrado["price"].dropna(), bins=60, alpha=0.5, label="Antes del filtrado")
plt.hist(listings_post_filtrado["price"].dropna(), bins=60, alpha=0.7, label="Después del filtrado")
plt.axvline(upper_price, linestyle="--", label=f"Percentil 99.5 = {upper_price:.2f} €")
plt.title("Distribución del precio antes y después del tratamiento de outliers")
plt.xlabel("Precio por noche (€)")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

#Gr�fico de nulos tras limpieza: 
#1�comporbar nulos: 
null_counts_final = listings.isnull().sum()
null_counts_final = null_counts_final[null_counts_final > 0].sort_values(ascending=False)
print("Valores nulos tras la limpieza:")
print(null_counts_final)

#2�Visualizaci�n: 
if len(null_counts_final) > 0:
    plt.figure(figsize=(10,6))
    bars = plt.barh(null_counts_final.index[::-1], null_counts_final.values[::-1])
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + max(null_counts_final) * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{int(width)}",
            va="center",
            fontsize=9
        )
    plt.xlabel("N�mero de valores nulos")
    plt.ylabel("Variables")
    plt.title("Valores nulos tras la limpieza")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No quedan valores nulos en el dataset final.")
    

#COMPROBACIONES: 
#Estadísticos de las variables:   
print(listings[["estimated_occupancy_l365d",
                "occupancy_rate",
                "estimated_revenue_l365d",
                "adr_check",
                "revpar"]].describe())

#Calculo de RevPAR para vericar que sea coherente (RevPAR = ADR * Occupancy Rate):
check = listings["adr_check"] * listings["occupancy_rate"]

#Tabla de comparación (RevPAR calculado vs RevPAR "recontruido"):
comparison = pd.DataFrame({
    "revpar": pd.to_numeric(listings["revpar"], errors="coerce"),
    "adr_x_occ": pd.to_numeric(check, errors="coerce")
})
print(comparison.head(10))

#Ver la diferencia entre ambos cálculos:
diff = comparison["revpar"] - comparison["adr_x_occ"]
diff = pd.to_numeric(diff, errors="coerce")
print(diff.describe())
print("Máxima diferencia absoluta:", diff.abs().max())
print(comparison.head(10))
print((comparison["revpar"] - comparison["adr_x_occ"]).describe())

#Visualización: 
plt.figure(figsize=(6,6))
plt.scatter(comparison["adr_x_occ"], comparison["revpar"], alpha=0.3)
plt.xlabel("ADR × Occupancy")
plt.ylabel("RevPAR")
plt.title("Validación de RevPAR")
plt.plot([0, comparison["revpar"].max()],
         [0, comparison["revpar"].max()],
         linestyle="--")
plt.show()


#OUTPUT: 
listings.to_csv('/Users/luciagonzalezdavila/Documents/4º/TFG/Nuevas BBDD/listings_clean.csv', index=False)


#VISUALIZACIÓN:
#DISTRIBUCIÓN DE PRECIOS:
plt.figure(figsize=(8,5))
plt.hist(listings["price"].dropna(), bins=50)
plt.title("Distribución del precio por noche")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")
plt.show()

#PRECIO VS OCUPACIÓN: 
plt.figure(figsize=(8,5))
plt.scatter(listings["price"], listings["estimated_occupancy_l365d"], alpha=0.3)
plt.title("Precio vs Ocupación")
plt.xlabel("Precio")
plt.ylabel("Noches ocupadas estimadas")
plt.show()

#PRECIO VS REVPAR:
plt.figure(figsize=(8,5))
plt.scatter(listings["price"], listings["revpar"], alpha=0.3)
plt.title("Precio vs RevPAR")
plt.xlabel("Precio")
plt.ylabel("RevPAR")
plt.show()

#BOXPLOT POR TIPO DE HABITACIÓN: 
room_data = []
labels = []

for room_type in listings["room_type"].dropna().unique():
    room_data.append(listings[listings["room_type"] == room_type]["revpar"].dropna())
    labels.append(room_type)

plt.figure(figsize=(10,6))
plt.boxplot(room_data, labels=labels)
plt.title("RevPAR por tipo de habitación")
plt.ylabel("RevPAR")
plt.xticks(rotation=45)
plt.show()


#REVPAR MEDIO POR BARRIO: 
revpar_barrio = listings.groupby("neighbourhood_cleansed")["revpar"].median().sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
plt.bar(revpar_barrio.index, revpar_barrio.values)
plt.title("Top 15 barrios por RevPAR (mediana)")
plt.ylabel("RevPAR medio")
plt.xticks(rotation=75)
plt.show()

#DISTRIBUCIÓN TASA DE OCUPACIÓN:
plt.figure(figsize=(8,5))
plt.hist(listings["occupancy_rate"].dropna(), bins=50)
plt.title("Distribución de la tasa de ocupación")
plt.xlabel("Occupancy rate")
plt.ylabel("Frecuencia")
plt.show()


#SUPERHOST: 
listings.groupby("host_is_superhost")["revpar"].mean()

# plot SUPERHOST
print("Precio medio:")
print(listings.groupby("host_is_superhost")["price"].mean())

print("Ocupación media:")
print(listings.groupby("host_is_superhost")["estimated_occupancy_l365d"].mean())


#PROFESIONALIZACIÓN: 
print("RevPAR medio:")
print(listings.groupby("professional_host")["revpar"].mean())

print("Precio medio:")
print(listings.groupby("professional_host")["price"].mean())

print("Ocupación media:")
print(listings.groupby("professional_host")["estimated_occupancy_l365d"].mean())


#CORRELACIÓN: 
corr = listings[[
    "price",
    "accommodates",
    "bedrooms",
    "review_scores_rating",
    "estimated_occupancy_l365d",
    "revpar"
]].corr()

print(corr)


#de forma visual: 
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Matriz de correlación")
plt.show()





#MODELADO:
#preparaciones previas: 
# Copia del dataset:
listings = listings.copy()
#Revisar nulos: 
print(listings[[
    "price", "accommodates", "bedrooms", "amenities_count", "has_wifi",
    "room_type", "neighbourhood_cleansed", "review_scores_rating", "host_is_superhost"
]].isna().sum())

#Eliminar los nulos: 
model_data_price = listings[[
    "price", "accommodates", "bedrooms", "amenities_count", "has_wifi",
    "room_type", "neighbourhood_cleansed", "review_scores_rating", "host_is_superhost"
]].dropna().copy()




#MODELO 1 — PRECIO (HEDONIC PRICING)
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

model_price = smf.ols(
    formula="""
    price ~ accommodates + bedrooms + amenities_count + has_wifi + C(room_type) + 
            C(neighbourhood_cleansed) + 
            review_scores_rating + host_is_superhost
    """,
    data=listings
).fit()

print(model_data_price.summary())

#Métricas:
y_true_price = model_data_price["price"]
y_pred_price = model_price.fittedvalues

rmse_price = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
mae_price = mean_absolute_error(y_true_price, y_pred_price)

print("RMSE precio:", rmse_price)
print("MAE precio:", mae_price)

#Visualización gráfica del modelo:
#Gráfico real vs predicho:
pred = model_price.fittedvalues
real = listings.loc[pred.index, "price"]

plt.figure(figsize=(6,6))
plt.scatter(pred, real, alpha=0.3)
plt.xlabel("Precio predicho")
plt.ylabel("Precio real")
plt.title("Real vs Predicho (Modelo de Precio)")
plt.show()

#Gráfico de residuos:
residuals = y_true_price - y_pred_price
plt.figure(figsize=(8,5))
plt.scatter(y_pred_price, residuals, alpha=0.3)
plt.axhline(0)
plt.xlabel("Precio predicho")
plt.ylabel("Residuos")
plt.title("Residuos vs Predicción")
plt.show()

#Distribución de residuos: 
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=50)
plt.title("Distribución de residuos (Modelo de precio)")
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.show()

# Importancia aproximada de variables:
coefs = model_price.params.copy()
# Filtrado de variables principales:
variables_clave = [
    "accommodates",
    "bedrooms",
    "amenities_count",
    "has_wifi",
    "review_scores_rating",
    "host_is_superhost",
    "C(room_type)[T.Private room]",
    "C(room_type)[T.Shared room]",
    "C(room_type)[T.Hotel room]"
]
coefs_filtrados = coefs[coefs.index.isin(variables_clave)].sort_values()
plt.figure(figsize=(9, 5))
bars = plt.barh(coefs_filtrados.index, coefs_filtrados.values)
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + (1 if width >= 0 else -6),    
        bar.get_y() + bar.get_height()/2,
        f"{width:.1f}",
        va="center"
    )
plt.xlabel("Impacto estimado sobre el precio (�)")
plt.title("Variables m�s influyentes en el precio")
plt.grid(axis="x", alpha=0.3)
plt.show()

#MODELO 2 — OCUPACIÓN (LOG PRECIO): 
#Evitar problemas con log(0) y hacer una copia:
occ_data = listings[listings["price"] > 0].copy()

#Creación de variable para poder observar las subidas porcentuales: 
listings["log_price"] = np.log(listings["price"])

#Revisar los nulos en la ocupación: 
print(occ_data[[
    "occupancy_rate", "log_price", "accommodates", "bedrooms",
    "amenities_count", "has_wifi", "bathrooms_missing",
    "room_type", "neighbourhood_cleansed",
    "review_scores_rating", "host_is_superhost"
]].isna().sum())

#Crear base específica para el modelo de ocupación:
model_data_occ = occ_data[[
    "occupancy_rate", "log_price", "accommodates", "bedrooms",
    "amenities_count", "has_wifi", "bathrooms_missing",
    "room_type", "neighbourhood_cleansed",
    "review_scores_rating", "host_is_superhost"
]].dropna().copy()

#Modelo:
model_occ = smf.ols(
    formula="""
    occupancy_rate ~ log_price + accommodates + bedrooms + amenities_count + has_wifi + bathrooms_missing +
                     C(room_type) + C(neighbourhood_cleansed) +
                     review_scores_rating + host_is_superhost
    """,
    data= model_data_occ
).fit()

print(model_occ.summary())
#Métricas:
y_true_occ = model_data_occ["occupancy_rate"]
y_pred_occ = model_occ.fittedvalues

rmse_occ = np.sqrt(mean_squared_error(y_true_occ, y_pred_occ))
mae_occ = mean_absolute_error(y_true_occ, y_pred_occ)

print("RMSE ocupación:", rmse_occ)
print("MAE ocupación:", mae_occ)

#Visualización gráfica del modelo:
#Gráfico real vs predicho:
plt.figure(figsize=(6,6))
plt.scatter(y_pred_occ, y_true_occ, alpha=0.3)
plt.xlabel("Ocupación predicha")
plt.ylabel("Ocupación real")
plt.title("Real vs Predicho (Modelo de ocupación)")
plt.show()

#Residuos y predicción: 
residuals_occ = y_true_occ - y_pred_occ
plt.figure(figsize=(8,5))
plt.scatter(y_pred_occ, residuals_occ, alpha=0.3)
plt.axhline(0)
plt.xlabel("Ocupación predicha")
plt.ylabel("Residuos")
plt.title("Residuos vs Predicción (Modelo de ocupación)")
plt.show()

#Distribución de residuos: 
plt.figure(figsize=(8,5))
plt.hist(residuals_occ, bins=50)
plt.title("Distribución de residuos (Modelo de ocupación)")
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.show()


#SIMULACIÓN DE PRECIO ÓPTIMO (segmentado por entire home/apt):
#Filtrar solo viviendas completas:
entire = listings[listings["room_type"] == "Entire home/apt"].copy()

# Asegurarme de que hay occupancy_rate:
if "occupancy_rate" not in entire.columns:
    entire["occupancy_rate"] = entire["estimated_occupancy_l365d"] / 365
    
#Evitar problemas con log(0):
entire = entire[entire["price"] > 0].copy()

#Creación variable logarítmica del precio:
entire["log_price"] = np.log(entire["price"])

#Revisar nulos en variables del modelo segmentado:
print(entire[[
    "occupancy_rate", "log_price", "accommodates", "bedrooms",
    "amenities_count", "has_wifi", "bathrooms_missing",
    "neighbourhood_cleansed", "review_scores_rating", "host_is_superhost"
]].isna().sum())

#Crear base específica del modelo segmentado:
entire_model_data = entire[[
    "occupancy_rate", "log_price", "price", "accommodates", "bedrooms",
    "amenities_count", "has_wifi", "bathrooms_missing",
    "neighbourhood_cleansed", "review_scores_rating", "host_is_superhost"
]].dropna().copy()

#A)MODELO DE OCUPACIÓN
model_occ_entire_log = smf.ols(
    formula="""
    occupancy_rate ~ log_price +
                     accommodates + bedrooms +
                     amenities_count + has_wifi +
                     bathrooms_missing +
                     C(neighbourhood_cleansed) +
                     review_scores_rating + host_is_superhost
    """,
    data= entire_model_data
).fit()

#Métricas del modelo de ocupación segmentado:
y_true_entire = entire_model_data["occupancy_rate"]
y_pred_entire = model_occ_entire_log.fittedvalues

rmse_entire = np.sqrt(mean_squared_error(y_true_entire, y_pred_entire))
mae_entire = mean_absolute_error(y_true_entire, y_pred_entire)

print("RMSE ocupación entire home/apt:", rmse_entire)
print("MAE ocupación entire home/apt:", mae_entire)
print(model_occ_entire_log.summary())

#B)SIMULACIÓN DE RANGOS DE PRECIOS:
price_range = np.linspace(40, 400, 80)


# C)CREACIÓN DE UN AIRBNB REPRESENTATIVO COMO BASE:
mean_values = entire_model_data.mean(numeric_only=True)
base = entire_model_data.iloc[0:1].copy()

#Sustituir las variables numéricas por la media:
for col in mean_values.index:
    if col in base.columns:
        base[col] = mean_values[col]

#Fijar por valores categóricos válidos:
base["room_type"] = "Entire home/apt"
base["neighbourhood_cleansed"] = entire["neighbourhood_cleansed"].mode()[0]


#D)SIMULACIÓN:
results = []
for p in price_range:
    base["price"] = p
    base["log_price"] = np.log(p)

    pred_occ = model_occ_entire_log.predict(base).iloc[0]

#Limitar al rango lógico [0,1]
    pred_occ = max(0, min(1, pred_occ))

    revpar = p * pred_occ

    results.append((p, pred_occ, revpar))

#Convertir a dataframe:
results_df_entire_log = pd.DataFrame(results, columns=["price", "occupancy", "revpar"])


#E)ENCONTRAR PRECIO ÓPTIMO:
optimal_entire_log = results_df_entire_log.loc[results_df_entire_log["revpar"].idxmax()]

print("RESULTADO FINAL | ENTIRE HOME/APT | MODELO LOG")
print(f"Precio óptimo: {optimal_entire_log['price']:.2f} €")
print(f"Ocupación estimada: {optimal_entire_log['occupancy']:.3f}")
print(f"RevPAR máximo: {optimal_entire_log['revpar']:.2f} €")


#F)VISUALIZACIÓN DE RESULTADOS:
#RevPAR vs Precio:
plt.figure(figsize=(8,5))
plt.plot(results_df_entire_log["price"], results_df_entire_log["revpar"])
plt.axvline(optimal_entire_log["price"], linestyle="--")
plt.scatter(optimal_entire_log["price"], optimal_entire_log["revpar"])
plt.xlabel("Precio (€)")
plt.ylabel("RevPAR (€)")
plt.title("Relación Precio - RevPAR | Entire home/apt")
plt.grid()
plt.show()

#Ocupación vs Precio:
plt.figure(figsize=(8,5))
plt.plot(results_df_entire_log["price"], results_df_entire_log["occupancy"])
plt.axvline(optimal_entire_log["price"], linestyle="--")
plt.scatter(optimal_entire_log["price"], optimal_entire_log["occupancy"])
plt.xlabel("Precio (€)")
plt.ylabel("Ocupación estimada")
plt.title("Relación Precio - Ocupación | Entire home/apt")
plt.grid()
plt.show()

#G)OUTPUT DEL MODELO:
results_df_entire_log.to_csv(
    '/Users/luciagonzalezdavila/Documents/4º/TFG/Nuevas BBDD/price_optimization_entire_log.csv',
    index=False
)


