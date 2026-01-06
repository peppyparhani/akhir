import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# ====================================================
# CONFIG
# ====================================================
st.set_page_config(
    page_title="Prediksi Stunting Balita - Random Forest",
    layout="wide"
)

st.title("🌱 Prediksi Stunting Balita (Random Forest)")
st.markdown("""
Aplikasi ini memprediksi **persentase stunting balita tahun berikutnya**
menggunakan **Random Forest Regression** berdasarkan data historis.
""")

# ====================================================
# UPLOAD DATA
# ====================================================
uploaded_file = st.file_uploader(
    "📂 Upload dataset stunting (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai.")
    st.stop()

# ====================================================
# LOAD & CLEAN DATA
# ====================================================
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

st.subheader("📄 Preview Data")
st.dataframe(df.head())

# ====================================================
# FEATURE ENGINEERING (LAG, MEAN, SLOPE)
# ====================================================
rows = []

for (_, _, _), g in df.groupby(
    ['nama_provinsi', 'kode_kabupaten_kota', 'nama_kabupaten_kota']
):
    g = g.sort_values('tahun')
    years = g['tahun'].values
    vals = g['persentase_balita_stunting'].values

    for i in range(1, len(vals)):
        lag1 = vals[i - 1]
        lag2 = vals[i - 2] if i - 2 >= 0 else np.nan
        lag3 = vals[i - 3] if i - 3 >= 0 else np.nan

        mean_prev = np.nanmean([lag1, lag2, lag3])

        slope_prev = 0
        if i >= 2:
            x = years[max(0, i - 3):i]
            y = vals[max(0, i - 3):i]
            if len(x) >= 2:
                slope_prev = np.polyfit(x, y, 1)[0]

        rows.append([
            lag1, lag2, lag3,
            mean_prev, slope_prev,
            vals[i]
        ])

data = pd.DataFrame(rows, columns=[
    'lag1', 'lag2', 'lag3',
    'mean_prev', 'slope_prev',
    'target'
])

features = ['lag1', 'lag2', 'lag3', 'mean_prev', 'slope_prev']
X = data[features].fillna(data[features].mean())
y = data['target']

# ====================================================
# TRAIN RANDOM FOREST
# ====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred_test = model.predict(X_test)

# ====================================================
# EVALUATION
# ====================================================
st.subheader("📈 Evaluasi Model")

mae = mean_absolute_error(y_test, pred_test)
rmse = sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R²", f"{r2:.3f}")

# ====================================================
# USER INPUT (LAG SAJA)
# ====================================================
st.subheader("🧮 Input Data Historis")

col1, col2, col3 = st.columns(3)

with col1:
    lag1 = st.number_input(
        "Lag-1 (Satu Tahun Terakhir)",
        min_value=0.0, max_value=100.0, value=20.0
    )

with col2:
    lag2 = st.number_input(
        "Lag-2 (Dua Tahun Terakhir)",
        min_value=0.0, max_value=100.0, value=21.0
    )

with col3:
    lag3 = st.number_input(
        "Lag-3 (Tiga Tahun Terakhir)",
        min_value=0.0, max_value=100.0, value=22.0
    )

# ====================================================
# AUTO FEATURE (MEAN & SLOPE)
# ====================================================
x = np.array([-2, -1, 0])     # 3 tahun lalu → tahun terakhir
y = np.array([lag3, lag2, lag1])

mean_prev = y.mean()
slope_prev = np.polyfit(x, y, 1)[0]

st.markdown("### 📊 Fitur Otomatis (Dihitung Sistem)")

c4, c5 = st.columns(2)
c4.metric("Rata-rata 3 Tahun", f"{mean_prev:.2f}")
c5.metric("Nilai Tren (Slope)", f"{slope_prev:.4f}")

# ====================================================
# PREDICTION
# ====================================================
if st.button("🔮 Prediksi Stunting Tahun Berikutnya"):
    user_feat = np.array([
        lag1, lag2, lag3, mean_prev, slope_prev
    ]).reshape(1, -1)

    pred_value = model.predict(user_feat)[0]

    st.success(
        f"📌 **Prediksi Persentase Stunting:** "
        f"**{pred_value:.2f}%**"
    )

    if pred_value < 10:
        st.info("🟢 Prioritas Rendah")
    elif pred_value <= 20:
        st.warning("🟡 Prioritas Sedang")
    else:
        st.error("🔴 Prioritas Tinggi")
