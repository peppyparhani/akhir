import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Prediksi Stunting Balita - Random Forest",
    layout="wide"
)

st.title("🌱 Prediksi Stunting Balita (Random Forest)")
st.markdown(
    """
    Model **Random Forest** digunakan untuk memprediksi
    persentase stunting balita berdasarkan tren historis 3 tahun sebelumnya.
    """
)

uploaded_file = st.file_uploader(
    "📂 Upload dataset stunting (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai.")
    st.stop()

# ================= LOAD & CLEAN DATA =================
df = pd.read_csv(uploaded_file)
df.columns = [c.strip() for c in df.columns]

df['persentase_balita_stunting'] = pd.to_numeric(
    df['persentase_balita_stunting'], errors='coerce'
)

df = df.dropna(subset=[
    'persentase_balita_stunting',
    'nama_kabupaten_kota',
    'tahun'
])

df['tahun'] = df['tahun'].astype(int)

# ================= FEATURE ENGINEERING =================
rows = []

for (prov, kode, kab), g in df.groupby(
    ['nama_provinsi','kode_kabupaten_kota','nama_kabupaten_kota']
):
    g = g.sort_values('tahun')
    years = g['tahun'].values
    vals = g['persentase_balita_stunting'].values

    for i in range(1, len(vals)):
        v1 = vals[i-1]
        v2 = vals[i-2] if i-2 >= 0 else np.nan
        v3 = vals[i-3] if i-3 >= 0 else np.nan

        mean_prev = np.nanmean([v1, v2, v3])

        slope = 0
        if i >= 2:
            xi = years[max(0, i-3):i]
            yi = vals[max(0, i-3):i]
            if len(xi) >= 2:
                slope = np.polyfit(xi, yi, 1)[0]

        rows.append([v1, v2, v3, mean_prev, slope, vals[i]])

data = pd.DataFrame(rows, columns=[
    'lag1','lag2','lag3','mean_prev','slope_prev','target'
])

features = ['lag1','lag2','lag3','mean_prev','slope_prev']
X = data[features].fillna(data[features].mean())
y = data['target']

# ================= TRAIN MODEL (RANDOM FOREST) =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred_test = model.predict(X_test)

# ================= INPUT MANUAL =================
st.subheader("Input Manual Prediksi Tahun Berikutnya")

col1, col2, col3 = st.columns(3)

with col1:
    lag1 = st.number_input(
        "Lag-1 (Tahun Terakhir)",
        min_value=0.0, max_value=100.0, value=20.0
    )

with col2:
    lag2 = st.number_input(
        "Lag-2 (2 Tahun Lalu)",
        min_value=0.0, max_value=100.0, value=21.0
    )

with col3:
    lag3 = st.number_input(
        "Lag-3 (3 Tahun Lalu)",
        min_value=0.0, max_value=100.0, value=22.0
    )

mean_prev = np.mean([lag1, lag2, lag3])
slope_prev = np.polyfit([1, 2, 3], [lag3, lag2, lag1], 1)[0]

st.write(f"📊 **Rata-rata 3 tahun:** {mean_prev:.2f}")
st.write(f"📈 **Tren (Slope):** {slope_prev:.3f}")

# ================= PREDICTION =================
if st.button("🔮 Prediksi Stunting"):
    user_feat = np.array([
        lag1, lag2, lag3, mean_prev, slope_prev
    ]).reshape(1, -1)

    pred_value = model.predict(user_feat)[0]

    st.success(
        f"📌 **Prediksi Persentase Stunting Tahun Berikutnya:** "
        f"**{pred_value:.2f}%**"
    )

    if pred_value < 10:
        st.info("🟢 Prioritas Rendah")
    elif pred_value <= 20:
        st.warning("🟡 Prioritas Sedang")
    else:
        st.error("🔴 Prioritas Tinggi")

