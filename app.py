import streamlit as st
import pandas as pd
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Fake Medicine Detector",
    page_icon="💊",
    layout="centered"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #1a1f2b, #0b0f17);
    color: white;
}
.main {
    display: flex;
    justify-content: center;
}
.glass-card {
    max-width: 520px;
    margin: auto;
    padding: 35px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #4da3ff;
}
.subtitle {
    text-align: center;
    color: #9aa4b2;
    font-size: 15px;
    margin-bottom: 28px;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    background: linear-gradient(90deg, #4da3ff, #7b5cff);
    color: white;
    border: none;
}
input {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD FILES =================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
data = pd.read_csv("medicine_data.csv")

# ================= UI =================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="title">💊 Fake Medicine Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check whether a medicine is REAL or FAKE</div>', unsafe_allow_html=True)

medicine = st.text_input("Medicine Name", placeholder="e.g. Paracetamol")
manufacturer = st.text_input("Manufacturer Name (optional)", placeholder="e.g. Cipla")

# ================= MAIN LOGIC =================
if st.button("🔍 Check Medicine"):

    # -------- Basic Validation --------
    if medicine.strip() == "":
        st.warning("⚠️ Please enter medicine name")
        st.stop()

    med_lower = medicine.lower().strip()

    # -------- STEP 1: DATASET DIRECT MATCH (HIGHEST PRIORITY) --------
    match = data[data["medicine_name"].str.lower() == med_lower]

    if not match.empty:
        label = int(match.iloc[0]["label"])

        if label == 1:
            st.success("✅ Medicine is REAL (Verified from database)")
            st.progress(100)
            st.markdown("**Confidence:** `100%`")
            st.subheader("📌 Medicine Uses")
            st.info(match.iloc[0]["uses"])
            st.stop()
        else:
            st.error("❌ Medicine is FAKE (Listed as fake in database)")
            st.progress(0)
            st.markdown("**Confidence:** `0%`")
            st.stop()

    # -------- STEP 2: RULE-BASED MANUFACTURER CHECK --------
    fake_keywords = ["fake", "unknown", "unverified", "no name", "local"]

    if manufacturer.strip() != "":
        manu_lower = manufacturer.lower()
        if any(word in manu_lower for word in fake_keywords):
            st.error("❌ Medicine is FAKE (Untrusted Manufacturer)")
            st.progress(0)
            st.markdown("**Confidence:** `0%`")
            st.stop()

    # -------- STEP 3: MACHINE LEARNING (FALLBACK) --------
    # Prepare text correctly (FIXED BUG)
    if manufacturer.strip() != "":
        text = medicine + " " + manufacturer
    else:
        text = medicine

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max() * 100

    if pred == 1 and confidence >= 75:
        st.success("✅ Medicine is REAL (ML Prediction)")
        st.progress(int(confidence))
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")
    else:
        st.error("❌ Medicine is FAKE (ML Prediction)")
        st.progress(0)
        st.markdown("**Confidence:** `0%`")

st.markdown('</div>', unsafe_allow_html=True)