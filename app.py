import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pywt
import tensorflow as tf
from tensorflow.keras.models import load_model
from openai import OpenAI
import os

from rag_module import DR_RAG

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetic Retinopathy AI System",
    layout="wide",
    page_icon="🧠"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.main {background-color: #0e1117;}

.title {
    font-size: 32px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #9aa4b2;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 Intelligent Diagnosis System for Diabetic Retinopathy</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>CNN + Wavelet + Grad-CAM + SHAP + RAG + LLM</div>", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_llm():
    return OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

@st.cache_resource
def load_all():
    model = load_model("./results_final/best_model.h5", compile=False)
    rag = DR_RAG()
    return model, rag

llm = load_llm()
model, rag = load_all()

classes = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
IMG_SIZE = 224

# =========================================================
# PREPROCESS
# =========================================================
def preprocess(img):
    img = np.array(img.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(3.0,(8,8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

    rgb = img.astype(np.float32)/255.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    LL,(LH,HL,HH) = pywt.dwt2(gray.astype(np.float32),'haar')

    waves = []
    for w in [LL,LH,HL,HH]:
        w = cv2.resize(w,(IMG_SIZE,IMG_SIZE))
        w = (w - w.min())/(w.max()-w.min()+1e-6)
        waves.append(w)

    wave = np.stack(waves,axis=-1)

    return np.concatenate([rgb,wave],axis=-1)

# =========================================================
# GRAD-CAM
# =========================================================
def gradcam(img_array):
    last_conv = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, pred = grad_model(img_array)
        loss = tf.reduce_max(pred)

    grads = tape.gradient(loss, conv)
    weights = tf.reduce_mean(grads, axis=(0,1,2))

    cam = tf.reduce_sum(weights * conv[0], axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    return cam.numpy()

# =========================================================
# LLM REPORT
# =========================================================
def generate_report(disease, conf, guideline):

    prompt = f"""
Disease: {disease}
Confidence: {conf}

Guideline:
{guideline[:2000]}

Return:
- Summary
- Risk
- Recommendation
"""

    try:
        res = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role":"system","content":"You are a medical assistant."},
                {"role":"user","content":prompt}
            ]
        )

        content = res.choices[0].message.content

        if content is None or content.strip() == "":
            return "⚠️ AI returned empty response."

        return content

    except Exception as e:
        return f"❌ API Error: {str(e)}"

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.subheader("📤 Upload Fundus Image")
    file = st.file_uploader("Choose image", type=["jpg","png","jpeg"])


if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "final_summary" not in st.session_state:
    st.session_state.final_summary = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None


if file != st.session_state.last_file:
    st.session_state.ai_report = None
    st.session_state.final_summary = None
    st.session_state.last_file = file

# =========================================================
# MAIN TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Diagnostic System",
    "📊 Performance Metrics",
    "🧠 SHAP Explainability"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    if file:

        img = Image.open(file).convert("RGB")

        x = preprocess(img)
        x_input = np.expand_dims(x,0)

        pred = model.predict(x_input)[0]
        cls = np.argmax(pred)
        conf = float(np.max(pred))

        rag_res = rag.generate_report(cls, conf)

        cam = gradcam(x_input)
        cam = cv2.resize(cam, (224,224))
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        base = np.array(img.resize((224,224)))
        overlay = cv2.addWeighted(base,0.6,heatmap,0.4,0)

        col1, col2 = st.columns([1.2,1])

        # ================= LEFT =================
        with col1:
            st.markdown("### 🖼 Feature Visualization")
            st.image(img)
            st.image(overlay)

            st.success(f"Disease: {classes[cls]}")
            st.info(f"Confidence: {conf:.2%}")

        # ================= RIGHT =================
        with col2:

            st.markdown("### 📌 Decision Support System")

            t1, t2, t3 = st.tabs([
                "📚 Guidelines",
                "🤖 AI Report",
                "💡 Recommendation"
            ])

            # -------- GUIDELINES --------
            with t1:
                st.text_area("Clinical Evidence", rag_res["template_report"], height=250)

            # -------- AI REPORT --------
            with t2:

                if st.button("Generate AI Expert Report"):

                    with st.spinner("Generating report..."):

                        st.session_state.ai_report = generate_report(
                            classes[cls],
                            conf,
                            rag_res["template_report"]
                        )

                if st.session_state.ai_report is not None:
                    st.success("AI Report Generated")
                    st.write(st.session_state.ai_report)

            # -------- RECOMMENDATION --------
            with t3:

                st.write(rag_res["template_report"])

                if st.button("📊 Aggregate Recommendation"):

                    with st.spinner("Generating summary..."):

                        summary_prompt = f"""
Local:
{rag_res["template_report"]}

Prediction:
{classes[cls]} ({conf:.2%})
"""

                        res = llm.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role":"user","content":summary_prompt}]
                        )

                        st.session_state.final_summary = res.choices[0].message.content

                if st.session_state.final_summary:
                    st.success("Final Recommendation")
                    st.write(st.session_state.final_summary)

# =========================================================
# TAB 2
# =========================================================
with tab2:

    st.markdown("### 📊 Model Performance")

    for f in [
        "results_final/annotated_learning_curves.png",
        "results_final/confusion_matrix.png",
        "results_final/annotated_roc_auc.png"
    ]:
        if os.path.exists(f):
            st.image(f)

# =========================================================
# TAB 3（SHAP）
# =========================================================
with tab3:

    st.markdown("### 🧠 SHAP Explainability")

    shap_dir = "shap/"

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(shap_dir + "Bar.png"):
            st.image(shap_dir + "Bar.png")

        if os.path.exists(shap_dir + "Dependence.png"):
            st.image(shap_dir + "Dependence.png")

    with col2:
        if os.path.exists(shap_dir + "Waterfall.png"):
            st.image(shap_dir + "Waterfall.png")

        if os.path.exists(shap_dir + "Force.png"):
            st.image(shap_dir + "Force.png")