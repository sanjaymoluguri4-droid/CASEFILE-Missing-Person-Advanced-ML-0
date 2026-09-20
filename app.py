import streamlit as st
import pandas as pd
import numpy as np
import folium

from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CASEFILE - Missing Person Investigation",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔎 CASEFILE")
st.subheader(
    "AI-Powered Missing Person Investigation and "
    "Probable Location Prediction System"
)

st.info(
    "Academic simulation only. All case information shown in "
    "this application is fictional/synthetic."
)


# ============================================================
# SYNTHETIC CASE DATA
# ============================================================

case_data = {
    "Case ID": "MP-2026-017",
    "Person ID": "P-017",
    "Age Group": "18–25",
    "Gender": "Female",
    "Last Known Location": "Area A",
    "Last Latitude": 23.2599,
    "Last Longitude": 77.4126,
    "Last Seen Time": "18:45",
    "Day": "Friday",
    "Weather": "Rain",
    "Average Distance": 9.5,
    "Average Speed": 32.0,
    "Time Since Last Seen": 4.5
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Case Selection")

case_id = st.sidebar.text_input(
    "Case ID",
    value=case_data["Case ID"]
)

st.sidebar.success("Synthetic Case Loaded")


# ============================================================
# CASE INFORMATION
# ============================================================

st.header("📋 Case Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Case ID", case_data["Case ID"])
col2.metric("Age Group", case_data["Age Group"])
col3.metric("Last Seen", case_data["Last Seen Time"])
col4.metric("Weather", case_data["Weather"])

st.write("### Case Details")

case_df = pd.DataFrame({
    "Attribute": [
        "Person ID",
        "Gender",
        "Last Known Location",
        "Last Latitude",
        "Last Longitude",
        "Day",
        "Average Distance",
        "Average Speed",
        "Time Since Last Seen"
    ],
    "Value": [
        case_data["Person ID"],
        case_data["Gender"],
        case_data["Last Known Location"],
        case_data["Last Latitude"],
        case_data["Last Longitude"],
        case_data["Day"],
        f'{case_data["Average Distance"]} km',
        f'{case_data["Average Speed"]} km/h',
        f'{case_data["Time Since Last Seen"]} hours'
    ]
})

st.dataframe(
    case_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SAMPLE HISTORICAL MOVEMENT DATA
# ============================================================

movement_data = pd.DataFrame({
    "Latitude": [
        23.2599, 23.2605, 23.2612, 23.2620,
        23.2630, 23.2640, 23.2650, 23.2660,
        23.2670, 23.2680
    ],
    "Longitude": [
        77.4126, 77.4135, 77.4145, 77.4155,
        77.4165, 77.4175, 77.4185, 77.4195,
        77.4205, 77.4215
    ],
    "Speed": [
        25, 28, 30, 32, 35,
        31, 29, 40, 65, 70
    ],
    "Distance": [
        0.5, 0.7, 0.8, 0.9, 1.0,
        0.8, 0.7, 1.2, 2.0, 2.5
    ]
})


# ============================================================
# FEATURE ENGINEERING
# ============================================================

features = movement_data[
    ["Speed", "Distance"]
].copy()

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)


# ============================================================
# MOVEMENT CLUSTERING
# ============================================================

st.header("📍 Movement Pattern Clustering")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

movement_data["Cluster"] = kmeans.fit_predict(
    scaled_features
)

cluster_count = movement_data["Cluster"].value_counts()

st.write("Detected movement patterns:")

st.dataframe(
    cluster_count.rename("Number of Points"),
    use_container_width=True
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.header("⚠️ Anomaly Detection")

isolation_forest = IsolationForest(
    contamination=0.20,
    random_state=42
)

movement_data["Anomaly"] = isolation_forest.fit_predict(
    scaled_features
)

movement_data["Anomaly Status"] = movement_data[
    "Anomaly"
].map({
    1: "Normal",
    -1: "Anomaly"
})

anomalies = movement_data[
    movement_data["Anomaly"] == -1
]

st.metric(
    "Detected Anomalies",
    len(anomalies)
)

st.dataframe(
    movement_data[
        [
            "Latitude",
            "Longitude",
            "Speed",
            "Distance",
            "Anomaly Status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.warning(
    "An anomaly represents unusual movement compared with "
    "the analyzed historical pattern. It does not prove "
    "suspicious or criminal behavior."
)


# ============================================================
# PROBABLE LOCATION PREDICTION
# ============================================================

st.header("🎯 Probable Location Prediction")

areas = pd.DataFrame({
    "Area": [
        "Area A",
        "Area B",
        "Area C",
        "Area D",
        "Area E"
    ],
    "Probability": [
        0.34,
        0.27,
        0.18,
        0.13,
        0.08
    ],
    "Latitude": [
        23.2599,
        23.2660,
        23.2720,
        23.2780,
        23.2840
    ],
    "Longitude": [
        77.4126,
        77.4195,
        77.4260,
        77.4330,
        77.4400
    ],
    "Visit_Frequency": [
        0.90,
        0.75,
        0.55,
        0.40,
        0.25
    ],
    "Route_Similarity": [
        0.85,
        0.75,
        0.60,
        0.45,
        0.30
    ],
    "Distance_Relevance": [
        0.90,
        0.80,
        0.65,
        0.50,
        0.35
    ],
    "Time_Relevance": [
        0.85,
        0.75,
        0.65,
        0.50,
        0.40
    ],
    "Anomaly_Evidence": [
        0.70,
        0.60,
        0.45,
        0.30,
        0.20
    ]
})


# ============================================================
# SEARCH PRIORITY SCORE
# ============================================================

areas["Search_Priority_Score"] = (
    areas["Probability"] * 30
    + areas["Visit_Frequency"] * 20
    + areas["Route_Similarity"] * 15
    + areas["Distance_Relevance"] * 15
    + areas["Time_Relevance"] * 10
    + areas["Anomaly_Evidence"] * 10
)

areas = areas.sort_values(
    "Search_Priority_Score",
    ascending=False
).reset_index(drop=True)


def priority_label(score):

    if score <= 30:
        return "Low"

    elif score <= 60:
        return "Medium"

    elif score <= 80:
        return "High"

    else:
        return "Very High"


areas["Priority"] = areas[
    "Search_Priority_Score"
].apply(priority_label)


areas["Probability"] = (
    areas["Probability"] * 100
).round(2)


areas["Search_Priority_Score"] = (
    areas["Search_Priority_Score"]
).round(2)


st.write("### Ranked Probable Areas")

display_df = areas[
    [
        "Area",
        "Probability",
        "Search_Priority_Score",
        "Priority"
    ]
].copy()

display_df["Probability"] = (
    display_df["Probability"].astype(str) + "%"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ROUTE PREDICTION
# ============================================================

st.header("🛣️ Probable Route Prediction")

route = [
    "Last Known Location",
    "Area A",
    "Area B",
    "Area C",
    "Area D"
]

route_df = pd.DataFrame({
    "Step": range(1, len(route) + 1),
    "Location": route
})

st.dataframe(
    route_df,
    use_container_width=True,
    hide_index=True
)

st.success(
    " → ".join(route)
)


# ============================================================
# EXPLAINABLE AI
# ============================================================

st.header("💡 Explainable AI")

top_area = areas.iloc[0]

explanations = [
    (
        "ML Prediction Probability",
        f'{top_area["Probability"]}%'
    ),
    (
        "Historical Visit Frequency",
        f'{top_area["Visit_Frequency"]:.2f}'
    ),
    (
        "Route Similarity",
        f'{top_area["Route_Similarity"]:.2f}'
    ),
    (
        "Distance Relevance",
        f'{top_area["Distance_Relevance"]:.2f}'
    ),
    (
        "Time Relevance",
        f'{top_area["Time_Relevance"]:.2f}'
    ),
    (
        "Anomaly Evidence",
        f'{top_area["Anomaly_Evidence"]:.2f}'
    )
]

explanation_df = pd.DataFrame(
    explanations,
    columns=["Feature", "Value"]
)

st.write(
    f'### Why {top_area["Area"]} received a high score'
)

st.dataframe(
    explanation_df,
    use_container_width=True,
    hide_index=True
)

st.info(
    "The score is based on the defined project factors. "
    "It represents a probabilistic investigation-support "
    "output and does not establish the person's actual location."
)


# ============================================================
# INTERACTIVE MAP
# ============================================================

st.header("🗺️ Interactive Investigation Map")

map_object = folium.Map(
    location=[
        case_data["Last Latitude"],
        case_data["Last Longitude"]
    ],
    zoom_start=13
)


# Last known location

folium.Marker(
    [
        case_data["Last Latitude"],
        case_data["Last Longitude"]
    ],
    popup="Last Known Location",
    tooltip="Last Known Location",
    icon=folium.Icon(
        color="red",
        icon="info-sign"
    )
).add_to(map_object)


# Historical movement

movement_points = movement_data[
    ["Latitude", "Longitude"]
].values.tolist()

folium.PolyLine(
    movement_points,
    tooltip="Historical Movement",
    weight=4
).add_to(map_object)


# Predicted areas

for _, row in areas.iterrows():

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=10,
        popup=(
            f'Area: {row["Area"]}<br>'
            f'Probability: {row["Probability"]}%<br>'
            f'Priority: {row["Priority"]}'
        ),
        tooltip=row["Area"],
        fill=True
    ).add_to(map_object)


# Predicted route

route_coordinates = [
    [
        case_data["Last Latitude"],
        case_data["Last Longitude"]
    ]
]

for _, row in areas.head(4).iterrows():

    route_coordinates.append([
        row["Latitude"],
        row["Longitude"]
    ])


folium.PolyLine(
    route_coordinates,
    tooltip="Predicted Route",
    weight=5,
    dash_array="10"
).add_to(map_object)


st_folium(
    map_object,
    width=1200,
    height=600
)


# ============================================================
# FINAL SUMMARY
# ============================================================

st.header("📊 Investigation Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

summary_col1.metric(
    "Top Probable Area",
    top_area["Area"]
)

summary_col2.metric(
    "Prediction Probability",
    f'{top_area["Probability"]}%'
)

summary_col3.metric(
    "Search Priority",
    top_area["Priority"]
)

st.warning(
    "This application is an academic simulation only. "
    "Predictions are probabilistic and should not be interpreted "
    "as proof of a person's location."
)

st.caption(
    "CASEFILE – Advanced Machine Learning Individual Project"
)
