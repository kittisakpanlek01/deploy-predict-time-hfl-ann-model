
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# ---------------- Load Models & Preprocessors ----------------
MODEL_PATHS = {
    "SHL": ("nn_time_model_shl.keras", "preprocessor_shl.pkl"),
    "SK1": ("nn_time_model_sk1.keras", "preprocessor_sk1.pkl"),
    "SK2": ("nn_time_model_sk2.keras", "preprocessor_sk2.pkl"),
    "SK3": ("nn_time_model_sk3.keras", "preprocessor_sk3.pkl"),
}

@st.cache_resource
def load_model_and_preprocessor(model_file, pre_file):
    model = tf.keras.models.load_model(model_file, compile=False)
    preprocessor = joblib.load(pre_file)
    return model, preprocessor


# ---------------- Prediction Function ----------------
def predict_cycle_time(process, input_df):
    model_file, pre_file = MODEL_PATHS[process]
    model, preprocessor = load_model_and_preprocessor(model_file, pre_file)

    X_processed = preprocessor.transform(input_df)
    preds = model.predict(X_processed, verbose=0)

    total_p50, total_p90, station_p50 = preds
    total_p50 = total_p50.flatten()
    total_p90 = total_p90.flatten()
    station_p50 = station_p50

    # Approx P90 per station
    std_per_station = np.std(station_p50, axis=0) * 0.1
    k = 1.28  # 90th quantile
    station_p90 = station_p50 + k * std_per_station

    # Cumulative
    cum_p50 = np.cumsum(station_p50, axis=1)
    cum_p90 = np.cumsum(station_p90, axis=1)

    station_cols_p50 = [f"T{i}_P50" for i in range(1, 10+1)]
    station_cols_p90 = [f"T{i}_P90" for i in range(1, 10+1)]
    cum_cols_p50 = [f"CUM_T{i}_P50" for i in range(1, 10+1)]
    cum_cols_p90 = [f"CUM_T{i}_P90" for i in range(1, 10+1)]
    total_cols = ["TOTAL_P50", "TOTAL_P90"]

    all_cols = station_cols_p50 + station_cols_p90 + cum_cols_p50 + cum_cols_p90 + total_cols

    pred_array = np.hstack([
        station_p50, station_p90, cum_p50, cum_p90,
        total_p50.reshape(-1, 1),
        total_p90.reshape(-1, 1)
    ])

    df_pred = pd.DataFrame(pred_array, columns=all_cols)

    # Convert sec → min (float)
    df_pred = df_pred.astype(float) / 60.0

    return df_pred


# ---------------- Streamlit UI ----------------
st.title("⏱️ Neural Network Cycle Time Prediction")
st.write("เลือกกระบวนการผลิต และใส่ข้อมูล input เพื่อคำนวณเวลา")

# 1) เลือก process
process = st.selectbox("เลือกกระบวนการ", ["SHL", "SK1", "SK2", "SK3"])

# 2) Input Mode
input_mode = st.radio("เลือกโหมด Input", ["Upload File", "Manual Form"])

if input_mode == "Upload File":
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel/CSV", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)
        st.success("✅ โหลดข้อมูลสำเร็จ")
        st.dataframe(input_df.head())
    else:
        st.warning("กรุณาอัปโหลดไฟล์ก่อนทำการพยากรณ์")
        input_df = None

else:
    st.subheader("📥 กรอกค่าพารามิเตอร์ (Manual Form)")

    # Numeric Inputs
    Thk = st.number_input("Thk (mm)", value=1.8)
    Width = st.number_input("Width (mm)", value=1000)
    Weight_In = st.number_input("Weight_In (kg)", value=20000)
    Length_In = st.number_input("Length_In (mm)", value=2000)

    LEN_T2 = st.number_input("LEN_T2", value=0)
    LEN_T4 = st.number_input("LEN_T4", value=10)
    LEN_T5 = st.number_input("LEN_T5", value=20)
    LEN_T6 = st.number_input("LEN_T6", value=1500)
    LEN_T8 = st.number_input("LEN_T8", value=0)
    LEN_T9 = st.number_input("LEN_T9", value=0)
    LEN_T10 = st.number_input("LEN_T10", value=40)

    CUT_T2 = st.number_input("CUT_T2", value=5)
    CUT_T7 = st.number_input("CUT_T7", value=6)
    CUT_T10 = st.number_input("CUT_T10", value=10)

    SPEED_T6 = st.number_input("SPEED_T6", value=120)
    SPEED_T8 = st.number_input("SPEED_T8", value=0)

    ENT_TENS_T6 = st.number_input("ENT_TENS_T6", value=4000)
    ENT_TENS_T8 = st.number_input("ENT_TENS_T8", value=0)

    # Extra fields for SK1 / SK2 / SK3
    if process in ["SK1", "SK2", "SK3"]:
        EXT_TENS_T6 = st.number_input("EXT_TENS_T6", value=12000)
        EXT_TENS_T8 = st.number_input("EXT_TENS_T8", value=0)
        FORCE_T6 = st.number_input("FORCE_T6", value=150)
        FORCE_T8 = st.number_input("FORCE_T8", value=0)
    else:
        EXT_TENS_T6 = EXT_TENS_T8 = FORCE_T6 = FORCE_T8 = None

    # Categorical Inputs
    if process in ["SK1"]:
        # no categorical inputs
        CycleCode_In = SteelGrade = None
    else:
        # CycleCode_In = st.text_input("CycleCode_In", value="A01")
        # SteelGrade = st.text_input("SteelGrade", value="SG1")
        CycleCode_In = st.selectbox("CycleCode_In", ["0", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"])
        SteelGrade = st.selectbox("SteelGrade", ["HR1", "SPHC", "A46", "A48", "SS400", "SS490", "J SAPH400", "J SAPH440", "J SAPH490", "J SAPH590", "SSI HY370"])

    # Build dict
    input_dict = {
        "Thk": Thk, "Width": Width, "Weight_In": Weight_In, "Length_In": Length_In,
        "LEN_T2": LEN_T2, "LEN_T4": LEN_T4, "LEN_T5": LEN_T5, "LEN_T6": LEN_T6,
        "LEN_T8": LEN_T8, "LEN_T9": LEN_T9, "LEN_T10": LEN_T10,
        "CUT_T2": CUT_T2, "CUT_T7": CUT_T7, "CUT_T10": CUT_T10,
        "SPEED_T6": SPEED_T6, "SPEED_T8": SPEED_T8,
        "ENT_TENS_T6": ENT_TENS_T6, "ENT_TENS_T8": ENT_TENS_T8,
        "CycleCode_In": CycleCode_In, "SteelGrade": SteelGrade
    }
    if process in ["SK2", "SK3"]:
        input_dict.update({
            "EXT_TENS_T6": EXT_TENS_T6,
            "EXT_TENS_T8": EXT_TENS_T8,
            "FORCE_T6": FORCE_T6,
            "FORCE_T8": FORCE_T8
        })

    input_df = pd.DataFrame([input_dict])

    st.success("✅ รับค่าจากฟอร์มเรียบร้อยแล้ว")
    st.dataframe(input_df)

    print("Process:", process)
    print("Input DF cols:", list(input_df.columns))
    print("X_processed shape:", X_processed.shape)
    print("Model input shape:", model.input_shape)


# 3) Run prediction
if input_df is not None and st.button("🔮 Predict"):
    df_pred = predict_cycle_time(process, input_df)

    st.subheader("📊 Prediction Results (minutes)")
    st.dataframe(df_pred)

    # กราฟ cumulative
    st.subheader("📈 Cumulative Time (P50)")
    st.line_chart(df_pred[[f"CUM_T{i}_P50" for i in range(1, 11)]])
