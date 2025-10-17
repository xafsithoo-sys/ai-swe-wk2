
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pydeck as pdk
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Load model and scaler
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("kenya_water.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()
labels = ['Sunny', 'Rainy']

# ---------------------------------------------------------
# County rainfall baselines (approximate averages)
# ---------------------------------------------------------
county_baseline = {
    "Turkana": 15.0, "Marsabit": 30.0, "Garissa": 25.0, "Wajir": 35.0,
    "Mandera": 40.0, "Machakos": 80.0, "Kitui": 60.0, "Nakuru": 100.0,
    "Nyeri": 120.0, "Kisumu": 150.0, "Nairobi": 110.0, "Mombasa": 160.0,
    "Kwale": 140.0, "Narok": 90.0, "Baringo": 75.0,
}

# County coordinates
county_coords = {
    "Turkana": [3.121, 35.596], "Marsabit": [2.34, 37.99], "Garissa": [0.456, 39.646],
    "Wajir": [1.747, 40.057], "Mandera": [3.937, 41.856], "Machakos": [-1.517, 37.263],
    "Kitui": [-1.367, 38.010], "Nakuru": [-0.303, 36.080], "Nyeri": [-0.420, 36.950],
    "Kisumu": [-0.0917, 34.768], "Nairobi": [-1.286, 36.817], "Mombasa": [-4.043, 39.668],
    "Kwale": [-4.173, 39.452], "Narok": [-1.085, 35.871], "Baringo": [0.678, 35.967],
}

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Kenya Water AI", page_icon="💧", layout="wide")

st.title("💧 AI for Tackling Water Scarcity in Kenya")
st.markdown("""
This app uses an **AI Neural Network** trained on Kenyan regional data 
to predict **Weather Conditions** (Sunny/Rainy) based on rainfall and climate variables.

🔹 Supports **SDG 6: Clean Water and Sanitation**  
🔹 Supports **SDG 13: Climate Action**
""")

# ---------------------------------------------------------
# User Inputs
# ---------------------------------------------------------
st.header("📍 Predict Regional Weather Condition")

col1, col2 = st.columns(2)
with col1:
    county = st.selectbox("Select County", sorted(county_baseline.keys()))
with col2:
    rainfall_mm = st.number_input("Monthly Rainfall (mm)", min_value=0.0, max_value=1000.0, value=float(county_baseline[county]))

col3, col4, col5 = st.columns(3)
with col3:
    year = st.number_input("Year", min_value=1980, max_value=2035, value=2025)
with col4:
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=10)
with col5:
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=5.0)

temperature = st.slider("🌡️ Average Temperature (°C)", 10.0, 40.0, 25.0)
humidity = st.slider("💨 Humidity (%)", 10.0, 100.0, 60.0)

# ---------------------------------------------------------
# Predict selected county's weather
# ---------------------------------------------------------
input_data = np.array([[year, month, temperature, humidity, wind_speed, rainfall_mm]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
predicted_class = np.argmax(prediction)
availability = labels[predicted_class]

st.success(f"🌍 County: **{county}** → 💧 Predicted Weather: **{availability}**")

st.bar_chart(pd.DataFrame(prediction, columns=labels))

if availability == "Sunny":
    st.info("☀️ Expected weather is **Sunny**. Water scarcity may increase — consider conservation measures.")
else:
    st.warning("🌧️ Expected weather is **Rainy**. Ensure water harvesting and flood management systems are ready.")
    # ---------------------------------------------------------
# 🌍 Dynamic Kenya Weather Map with Auto-Zoom + Summary Card
# ---------------------------------------------------------

st.header("🗺️ Interactive Kenya Weather Prediction Map")

# Dropdown for user to focus on a specific county
selected_county = st.selectbox(
    "🔎 Choose a county to focus on:",
    options=["All Counties"] + list(county_coords.keys()),
    index=0
)

# --- Function to generate predictions ---
def generate_map_data(year, month, temperature, humidity, wind_speed, rainfall):
    data = []
    for cty, (lat, lon) in county_coords.items():
        baseline_rainfall = county_baseline[cty]
        rainfall_input = rainfall or baseline_rainfall

        x = np.array([[year, month, temperature, humidity, wind_speed, rainfall_input]])
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)
        pred_class = np.argmax(pred)
        condition = labels[pred_class]

        color = [255, 210, 0] if condition == "Sunny" else [0, 140, 255]

        data.append({
            "County": cty,
            "Latitude": lat,
            "Longitude": lon,
            "Condition": condition,
            "Rainfall (mm)": round(rainfall_input, 1),
            "Temperature (°C)": temperature,
            "color": color
        })
    return pd.DataFrame(data)

# Generate prediction data
df_map = generate_map_data(year, month, temperature, humidity, wind_speed, rainfall_mm)

# Determine zoom + center dynamically
if selected_county != "All Counties":
    focus_lat, focus_lon = county_coords[selected_county]
    zoom_level = 8.5
else:
    focus_lat, focus_lon = 0.2, 37.8
    zoom_level = 6.5

# --- Layout for Map and Summary Card ---
col_map, col_info = st.columns([3, 1])

# 🗺️ Display Interactive Map
with col_map:
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(
                latitude=focus_lat,
                longitude=focus_lon,
                zoom=zoom_level,
                pitch=50,
                bearing=20
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position='[Longitude, Latitude]',
                    get_fill_color='color',
                    get_radius=70000,
                    pickable=True,
                    opacity=0.9
                ),
                pdk.Layer(
                    "TextLayer",
                    data=df_map,
                    get_position='[Longitude, Latitude]',
                    get_text="County",
                    get_color=[255, 255, 255],
                    get_size=16,
                    get_alignment_baseline="'bottom'"
                ),
            ],
            tooltip={"text": "{County}\nCondition: {Condition}\nRainfall: {Rainfall (mm)} mm"}
        )
    )


# 🧾 Display Summary Card
with col_info:
    st.subheader("📋 County Summary")
    if selected_county == "All Counties":
        st.info("Select a county to view detailed predictions.")
    else:
        selected_data = df_map[df_map["County"] == selected_county].iloc[0]
        condition = selected_data["Condition"]
        rain = selected_data["Rainfall (mm)"]
        temp = selected_data["Temperature (°C)"]

        st.metric("🌍 County", selected_county)
        st.metric("🌤️ Condition", condition)
        st.metric("🌧️ Rainfall (mm)", f"{rain}")
        st.metric("🌡️ Temperature (°C)", f"{temp}")
        if condition == "Rainy":
            st.warning("☔ Expect wet conditions — possible heavy showers.")
        else:
            st.success("☀️ Sunny and clear — minimal rainfall expected.")

st.caption(
    "💡 Select a county to zoom in and see AI-predicted conditions. "
    "The summary updates instantly with temperature and rainfall."
)

# ---------------------------------------------------------
# 📊 Explore County Rainfall & Temperature Patterns
# ---------------------------------------------------------
st.header("📊 Explore County Rainfall & Weather Patterns")

weather_data = {
    "Turkana":  {"rainfall": [10, 12, 15, 14, 18, 10, 8, 5, 7, 9, 11, 13],
                 "temperature": [34, 35, 36, 37, 38, 39, 38, 37, 36, 35, 34, 33]},
    "Nairobi":  {"rainfall": [100, 80, 60, 40, 50, 20, 30, 70, 90, 120, 110, 95],
                 "temperature": [27, 26, 25, 23, 22, 21, 22, 23, 25, 26, 27, 28]},
    "Mombasa":  {"rainfall": [160, 140, 150, 130, 120, 100, 90, 110, 130, 150, 170, 180],
                 "temperature": [30, 31, 31, 30, 29, 28, 27, 28, 29, 30, 31, 32]},
    "Kisumu":   {"rainfall": [120, 100, 80, 60, 70, 90, 110, 130, 140, 160, 170, 150],
                 "temperature": [29, 28, 27, 26, 25, 26, 27, 28, 29, 30, 30, 29]},
}

col1, col2 = st.columns(2)
with col1:
    county_a = st.selectbox("Select County A", weather_data.keys(), index=0)
with col2:
    county_b = st.selectbox("Select County B", weather_data.keys(), index=1)

df_a = pd.DataFrame({
    "Month": list(range(1, 13)),
    "Rainfall (mm)": weather_data[county_a]["rainfall"],
    "Temperature (°C)": weather_data[county_a]["temperature"],
    "County": county_a
})
df_b = pd.DataFrame({
    "Month": list(range(1, 13)),
    "Rainfall (mm)": weather_data[county_b]["rainfall"],
    "Temperature (°C)": weather_data[county_b]["temperature"],
    "County": county_b
})
combined_df = pd.concat([df_a, df_b])

st.subheader("🌧️ Monthly Rainfall Comparison")
st.line_chart(combined_df.pivot(index="Month", columns="County", values="Rainfall (mm)"))

st.subheader("🔥 Monthly Temperature Comparison")
st.line_chart(combined_df.pivot(index="Month", columns="County", values="Temperature (°C)"))

month_selected = st.slider("Select Month", 1, 12, 6)
st.write(f"📅 **Month {month_selected}** Climate Snapshot:")

rainfall_a = weather_data[county_a]["rainfall"][month_selected - 1]
rainfall_b = weather_data[county_b]["rainfall"][month_selected - 1]
temp_a = weather_data[county_a]["temperature"][month_selected - 1]
temp_b = weather_data[county_b]["temperature"][month_selected - 1]

col1, col2 = st.columns(2)
with col1:
    st.metric(f"{county_a} Rainfall (mm)", rainfall_a)
    st.metric(f"{county_a} Temp (°C)", temp_a)
with col2:
    st.metric(f"{county_b} Rainfall (mm)", rainfall_b)
    st.metric(f"{county_b} Temp (°C)", temp_b)

st.caption("💡 Compare rainfall and temperature patterns across counties to identify climate risks and plan water management.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.write("👩‍💻 Developed as part of an AI for Sustainability project — addressing **SDG 6** & **SDG 13**.")




