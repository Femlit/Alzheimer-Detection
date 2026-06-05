import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AlzheimerAI — MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────
from utils.model_loader import load_model
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam

# ── Constants ─────────────────────────────────────────────────
CLASSES = ["MildDementia", "ModerateDementia", "NonDemented", "VeryMildDementia"]
CLASS_LABELS = {
    "MildDementia":     "Mild Dementia",
    "ModerateDementia": "Moderate Dementia",
    "NonDemented":      "Non Demented",
    "VeryMildDementia": "Very Mild Dementia",
}
CLASS_COLOURS = {
    "MildDementia":     "#E65100",
    "ModerateDementia": "#B71C1C",
    "NonDemented":      "#1B5E20",
    "VeryMildDementia": "#F57F17",
}
IMG_SIZE = 224

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.image("assets/oou.jpeg",
                 width=60)

st.sidebar.title("🧠 AlzheimerAI")
st.sidebar.markdown("*MRI Classification System*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Diagnose", "Model Performance", "About"],
    index=0
)

st.sidebar.divider()
st.sidebar.caption("EfficientNetB4 + ResNet50V2")
st.sidebar.caption("OASIS Dataset | 98.99% Accuracy")
st.sidebar.caption("Olabisi Onabanjo University, 2026")


# ═══════════════════════════════════════════════════════════════
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), np.array(img)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — DIAGNOSE
# ═══════════════════════════════════════════════════════════════
if page == "Diagnose":
    st.title("🧠 Alzheimer's Disease MRI Classifier")
    st.markdown(
        "Upload a T1-weighted brain MRI scan to receive an instant classification "
        "of Alzheimer's disease severity, along with a Grad-CAM heatmap showing "
        "which brain regions influenced the prediction."
    )
    st.info(
        "This tool is for research and educational purposes only. "
        "It is not a substitute for clinical diagnosis by a qualified neurologist.",
        icon="ℹ️"
    )
    st.divider()

    uploaded = st.file_uploader(
        "Upload a Brain MRI Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a 2D T1-weighted structural MRI scan."
    )

    if uploaded is not None:
        img_pil = Image.open(uploaded)
        model   = load_model()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("Uploaded MRI Scan")
            st.image(img_pil, width="stretch", caption="Original MRI")

        # Preprocess and predict
        # Preprocess and predict
        img_input, img_display = preprocess_image(img_pil)
        if hasattr(model, 'get_input_details'):
            # TFLite interpreter
            input_details  = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], img_input.astype('float32'))
            model.invoke()
            predictions = model.get_tensor(output_details[0]['index'])[0]
        else:
            # Keras model fallback
            predictions = model.predict(img_input, verbose=0)[0]
        
        pred_idx   = int(np.argmax(predictions))
        pred_class = CLASSES[pred_idx]
        confidence = float(predictions[pred_idx]) * 100
        colour      = CLASS_COLOURS[pred_class]

        with col2:
            st.subheader("Prediction Result")
            st.markdown(
                f"<div style='background-color:{colour}20; border-left: 5px solid {colour};"
                f"padding: 20px; border-radius: 8px;'>"
                f"<h2 style='color:{colour}; margin:0;'>{CLASS_LABELS[pred_class]}</h2>"
                f"<p style='font-size:18px; margin:8px 0 0;'>"
                f"Confidence: <strong>{confidence:.2f}%</strong></p>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("#### Confidence Scores")
            for i, cls in enumerate(CLASSES):
                score = float(predictions[i]) * 100
                bar_c = CLASS_COLOURS[cls]
                st.markdown(f"**{CLASS_LABELS[cls]}**")
                st.progress(score / 100,
                            text=f"{score:.2f}%")

        with col3:
            st.subheader("Grad-CAM Explanation")
            try:
                heatmap, _ = make_gradcam_heatmap(img_input, model)
                overlay, coloured = overlay_gradcam(img_display, heatmap)
                st.image(overlay, width="stretch",
                         caption="Heatmap overlay — warm regions drove the prediction")
                st.caption(
                    "Red/warm areas indicate brain regions that most strongly "
                    "influenced the prediction. Blue/cool areas had low influence."
                )
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated: {e}")

        st.divider()
        st.markdown("#### What does this mean?")
        descriptions = {
            "NonDemented":
                "No detectable signs of Alzheimer's disease were found. "
                "The brain structure appears consistent with a cognitively healthy individual (CDR = 0).",
            "VeryMildDementia":
                "Very early signs of cognitive decline are present. "
                "This corresponds to CDR = 0.5, where memory lapses may be minimal and could be "
                "consistent with normal ageing. Clinical follow-up is recommended.",
            "MildDementia":
                "Moderate cognitive decline is detected, affecting memory, language, and problem-solving. "
                "This corresponds to CDR = 1 and typically requires clinical assessment and management.",
            "ModerateDementia":
                "Advanced cognitive impairment is present, consistent with CDR = 2. "
                "Significant assistance with daily activities is usually required at this stage.",
        }
        st.info(descriptions[pred_class])


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        "Results from evaluating the trained EfficientNetB4 + ResNet50V2 "
        "attention fusion model on the held-out OASIS test set of 12,968 MRI images."
    )
    st.divider()

    # Load metrics
    metrics_path = "data/final_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy",   f"{m['test_accuracy']*100:.2f}%")
        col2.metric("AUC",             f"{m['test_auc']:.4f}")
        col3.metric("Weighted F1",     f"{m['weighted_f1']*100:.2f}%")
        col4.metric("Test Precision",  f"{m['test_precision']*100:.2f}%")

        st.divider()

    # Per-class table
    st.subheader("Per-Class Results")
    per_class_data = {
        "Class":     ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very Mild Dementia"],
        "Precision": ["0.96", "0.97", "1.00", "0.96"],
        "Recall":    ["1.00", "1.00", "0.99", "0.99"],
        "F1-Score":  ["0.98", "0.99", "0.99", "0.98"],
        "Support":   ["751",  "74",   "10,084", "2,059"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(per_class_data), width="stretch", hide_index=True)

    st.divider()

    # Plots
    tabs = st.tabs(["Training History", "Confusion Matrix", "ROC Curves", "Per-Class Metrics"])

    plot_files = {
        "Training History":   "assets/training_history.png",
        "Confusion Matrix":   "assets/confusion_matrix.png",
        "ROC Curves":         "assets/roc_auc_curves.png",
        "Per-Class Metrics":  "assets/per_class_metrics.png",
    }

    for tab, (title, path) in zip(tabs, plot_files.items()):
        with tab:
            if os.path.exists(path):
                st.image(path, width="stretch", caption=title)
            else:
                st.warning(f"{path} not found. Copy your plot files to the assets/ folder.")

    st.divider()
    st.subheader("Grad-CAM Examples from Test Set")
    if os.path.exists("assets/gradcam_grid.png"):
        st.image("assets/gradcam_grid.png", width="stretch",
                 caption="Grad-CAM heatmaps across all four dementia classes on the OASIS test set")

    st.divider()
    st.subheader("Benchmark Comparison")
    benchmark = {
        "Study":        ["Chandrasekaran et al. (2025)", "Abd El-Latif et al. (2023)",
                         "Asaduzzaman et al. (2025)",  "This Work (2026)"],
        "Dataset":      ["Kaggle (same)", "Kaggle (same)", "Kaggle (same)", "OASIS clean"],
        "Accuracy":     ["94.20%", "95.93%", "97.31%", "98.99%"],
    }
    df_bench = pd.DataFrame(benchmark)
    st.dataframe(df_bench, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About This Project")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Alzheimer's Disease Classification from Brain MRI Using Deep Learning

        This system classifies brain MRI scans into four Alzheimer's disease severity
        stages using a dual-branch deep learning architecture that combines
        **EfficientNetB4** and **ResNet50V2** through an attention-based feature
        fusion mechanism.

        **Dataset:** OASIS MRI Dataset (86,437 images, 4 classes)

        **Classes:**
        - Non Demented (CDR = 0)
        - Very Mild Dementia (CDR = 0.5)
        - Mild Dementia (CDR = 1)
        - Moderate Dementia (CDR = 2)

        **Training:**
        - Clean 70/15/15 patient-level data split
        - Zero data leakage between training, validation, and test sets
        - 30 epochs on Kaggle T4 x2 GPU
        - Class-weighted loss to handle class imbalance

        **Key Results:**
        - Test Accuracy: **98.99%**
        - AUC: **0.9998**
        - Perfect recall on Mild and Moderate Dementia classes
        - Grad-CAM explainability confirms neurologically meaningful activations
        """)

    with col2:
        st.markdown("""
        **Project Details**

        Institution:
        Olabisi Onabanjo University

        Department:
        Computer Engineering

        Supervisor:
        Dr. Oyedeji

        Year: 2026

        **Architecture**

        Branch A: EfficientNetB4

        Branch B: ResNet50V2

        Fusion: Attention gate + Concatenation

        Head: Dense 512 → Dense 256 → Softmax(4)
        """)

    st.divider()
    st.markdown("""
    **Disclaimer**

    This application is developed for academic research and educational purposes
    as a final year project. It is not a certified medical device and should not
    be used to make clinical diagnostic decisions. Any MRI classification output
    from this system must be reviewed by a qualified neurologist or radiologist
    before any clinical action is taken.
    """)
