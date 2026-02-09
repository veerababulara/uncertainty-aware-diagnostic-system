import streamlit as st
import time
from inference import (
    load_model,
    preprocess_image,
    extract_evidence,
    compute_epistemic_state,
    make_decision,
    detect_blur,
    generate_explanation,
    generate_quality_explanation,
    explain_abstention_cause,
    HYPOTHESES
)

st.set_page_config(
    page_title="Uncertainty-Aware Diagnostic System",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@300;500;700&display=swap');

html, body, .stApp {
    background-color: #EAF2FB;
    color: #1F2937;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Merriweather', serif;
    color: #1F2937;
}

.metric-box {
    padding: 1.3rem;
    border-radius: 14px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}

.decision-commit {
    border-left: 6px solid #2B6CB0;
    background: #EBF4FF;
}

.decision-abstain {
    border-left: 6px solid #718096;
    background: #F7FAFC;
}

.ignorance-box {
    border-left: 6px solid #DD6B20;
    background: #FFFAF0;
}

div[role="progressbar"] > div {
    background-color: #2B6CB0;
}

/* ===============================
   FIX: Uploaded filename visibility
   =============================== */

/* Uploaded file row (filename + size) */
[data-testid="stFileUploader"] + div {
    color: #1F2937 !important;
}

/* Text inside uploaded file row */
[data-testid="stFileUploader"] + div span,
[data-testid="stFileUploader"] + div p,
[data-testid="stFileUploader"] + div small {
    color: #1F2937 !important;
    font-weight: 500;
}

/* File icon */
[data-testid="stFileUploader"] + div svg {
    fill: #2B6CB0 !important;
}


</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    return load_model("model.pt")

model = load_system()


st.title("Entropy-Driven Uncertainty-Aware Diagnostic System")
st.caption(
    "Clinical decision-support with explicit epistemic ignorance. "
    "For research and assistive use only."
)

st.divider()


st.header("🩺 Evidence Acquisition")



uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.read())

    st.markdown(
        f"""
        <div class="metric-box" style="
            background: #EBF4FF;
            border-left: 6px solid #2B6CB0;
            margin-top: 0.5rem;
        "> 
            <b>📁 Uploaded File</b><br><br>
            <span style="
                background: #2B6CB0;
                color: white;
                padding: 0.25rem 0.6rem;
                border-radius: 6px;
                font-size: 0.95rem;
                font-weight: 500;
            ">
                <span style="
    background: #E3EEF9;
    color: #1F2937;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 500;
    border: 1px solid #B6D0EE;
">
    {uploaded_file.name}
</span>

    
        """,
        unsafe_allow_html=True
    )


    is_blurry, blur_score = detect_blur("temp_image.png")

    st.subheader("📷 Image Quality Assessment")
    st.markdown(
        f"""
        <div class="metric-box">
            <b>Optical Sharpness (Laplacian Variance)</b><br><br>
            <span style="font-size: 2.2rem; font-weight: 700; color: #1F2937;">
                {blur_score:.1f}
            </span><br>
            <span style="color: #4A5568;">
                Higher values indicate stronger edge definition and preserved detail
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if is_blurry:
        st.warning("⚠️ LOW IMAGE QUALITY DETECTED")
    else:
        st.success("✅ IMAGE QUALITY ACCEPTABLE")

    st.divider()


    st.header("🫁 Epistemic Preparation")

    st.markdown(
        """
        <div class="metric-box">
            <b>Preprocessing Objective</b><br><br>
            This stage standardizes the input representation so that evidential
            reasoning is driven by anatomical structure rather than acquisition
            variability or scanner-dependent artifacts.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Preparing evidential representation…"):
        progress = st.progress(0.0)

        steps = [
            "Intensity normalization to align with training conditions",
            "Spatial resizing to 224 × 224 for consistent feature extraction",
            "Tensor encoding and statistical standardization"
        ]

        for i, step in enumerate(steps):
            time.sleep(0.7)
            progress.progress((i + 1) / len(steps))
            st.caption(f"✔ {step}")

        image_tensor = preprocess_image("temp_image.png")
        z = extract_evidence(model, image_tensor)
        epistemic_state = compute_epistemic_state(z)
        decision, idx = make_decision(epistemic_state)

    st.divider()

    
    st.header("⚕️ Uncertainty-Aware Diagnostic Reasoning")

    if decision == "COMMIT":
        st.subheader("Hypothesis Support")
    else:
        st.subheader("Evidential Tendencies (Non-diagnostic)")
        
    for i, name in enumerate(HYPOTHESES):
        st.write(f"**{name}**")
        st.progress(float(epistemic_state["belief"][i]))
        st.caption(
            f"Belief: {epistemic_state['belief'][i]:.2f} | "
            f"Plausibility: {epistemic_state['plausibility'][i]:.2f}"
        )

    st.subheader("Explicit Ignorance")
    st.markdown(
        f"""
        <div class="metric-box ignorance-box">
            <b>Ignorance Mass:</b> {epistemic_state['ignorance']:.2f}<br>
            Represents uncommitted belief due to limited or ambiguous evidence.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Predictive Entropy")
    st.markdown(
        f"""
        <div class="metric-box">
            <b>Predictive Entropy (Uncertainty Strength)</b><br><br>
            <span style="font-size: 2.2rem; font-weight: 700; color: #1F2937;">
                {epistemic_state['entropy']:.2f}
            </span><br>
            <span style="color: #4A5568;">
                Higher values indicate increased evidential uncertainty
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Final Decision")

    if decision == "COMMIT":
        st.markdown(
            f"""
            <div class="metric-box decision-commit">
                ✅ <b>COMMIT</b><br>
                Diagnosed Condition: <b>{HYPOTHESES[idx]}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="metric-box decision-abstain">
                ⛔ <b>ABSTAIN</b><br>
                No diagnostic label is issued due to insufficient or ambiguous evidence.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("🧾 Epistemic Explanation")
    st.write(generate_explanation(epistemic_state, decision, idx))

    st.subheader("📷 Image Quality Context")
    st.write(generate_quality_explanation(is_blurry, blur_score))


st.divider()
st.caption(
    "⚠️ This system is a decision-support tool. "
    "Abstention reflects epistemic responsibility and patient safety."
)
