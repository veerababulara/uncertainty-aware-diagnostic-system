# ============================================================
# Entropy-Driven Evidential Reasoning System
# Professional Reference Implementation
# ============================================================

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np




# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HYPOTHESES = [
    "Pneumonia",
    "Lung Opacity",
    "Pleural Effusion"
]

BELIEF_THRESHOLD = 0.5          # τ in the paper
UNCERTAINTY_EPS = 1e-8

# ============================================================
# MODEL LOADING (Evidence Generator)
# ============================================================

def load_model(model_path: str) -> nn.Module:
    """
    Loads DenseNet-121 as an evidence generator.
    """
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
    )
    model.classifier = nn.Linear(model.classifier.in_features, len(HYPOTHESES))

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

# ============================================================
# PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor.to(DEVICE)

# ============================================================
# EVIDENTIAL & EPISTEMIC REASONING
# ============================================================

def extract_evidence(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Produces raw evidential activations z_i.
    """
    with torch.no_grad():
        z = model(image_tensor).squeeze(0)
    return z

def compute_epistemic_state(z: torch.Tensor):
    """
    Implements Equations (7)–(13) from the paper.
    """
    # Evidence mapping
    e = torch.exp(z)

    # Total evidence
    S = torch.sum(e)

    # Normalized evidence (for entropy only)
    p = e / (S + UNCERTAINTY_EPS)

    # Shannon entropy (uncertainty strength)
    entropy = -torch.sum(p * torch.log(p + UNCERTAINTY_EPS))
    entropy = entropy / torch.log(torch.tensor(len(HYPOTHESES), device=DEVICE))

    # Belief mass assignment
    belief = e / (S + 1.0)

    # Ignorance mass
    ignorance = 1.0 / (S + 1.0)

    # Plausibility
    plausibility = belief + ignorance

    return {
        "belief": belief.cpu(),
        "plausibility": plausibility.cpu(),
        "ignorance": ignorance.cpu().item(),
        "entropy": entropy.cpu().item()
    }
def generate_explanation(epistemic_state: dict, decision: str, idx: int) -> str:
    """
    Generates a human-readable epistemic explanation
    aligned with the proposed System.
    """

    belief = epistemic_state["belief"][idx].item()
    ignorance = epistemic_state["ignorance"]
    entropy = epistemic_state["entropy"]

    hypothesis = HYPOTHESES[idx]

    if decision == "COMMIT":
        explanation = (
            f"The system commits to '{hypothesis}' because the belief "
            f"assigned to this hypothesis is sufficiently high "
            f"(belief = {belief:.2f}), while epistemic ignorance remains limited "
            f"(ignorance = {ignorance:.2f}). "
            f"The low-to-moderate predictive entropy (H = {entropy:.2f}) "
            f"indicates reliable evidential discrimination among hypotheses."
        )

    elif decision == "DEFER":
        explanation = (
            f"The system defers a definitive diagnosis. Although '{hypothesis}' "
            f"shows the highest belief (belief = {belief:.2f}), the available "
            f"evidence is not sufficiently reliable to justify commitment. "
            f"Epistemic ignorance (ignorance = {ignorance:.2f}) and predictive "
            f"entropy (H = {entropy:.2f}) indicate partial or ambiguous support, "
            f"suggesting that additional information or expert review is required."
        )

    else:  # ABSTAIN
        explanation = (
            f"The system abstains from diagnosis because epistemic ignorance "
            f"dominates the evidential state (ignorance = {ignorance:.2f}). "
            f"The high predictive entropy (H = {entropy:.2f}) reflects weak or "
            f"conflicting evidence, making reliable discrimination among "
            f"hypotheses unjustified. Abstention is therefore issued as a "
            f"safety-oriented and epistemically justified outcome."
        )

    return explanation


# ============================================================
# DECISION CONTROL (UNCERTAINTY-AWARE)
# ============================================================

def make_decision(epistemic_state: dict):
    belief = epistemic_state["belief"]
    ignorance = epistemic_state["ignorance"]

    dominant_idx = torch.argmax(belief).item()
    dominant_belief = belief[dominant_idx].item()

    if dominant_belief >= BELIEF_THRESHOLD and ignorance < (1 - BELIEF_THRESHOLD):
        return "COMMIT", dominant_idx
    else:
        return "ABSTAIN", dominant_idx

# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

def display_results(epistemic_state: dict, decision: str, idx: int,
                    is_blurry: bool, blur_score: float):

    print("\n================ EPISTEMIC DIAGNOSTIC OUTPUT ================")
    if is_blurry:
        print("⚠️  IMAGE QUALITY STATUS : LOW (BLUR DETECTED)")
    else:
        print("✅ IMAGE QUALITY STATUS : ACCEPTABLE")
    



    for i, name in enumerate(HYPOTHESES):
        print(
            f"{name:<18} | "
            f"Belief: {epistemic_state['belief'][i]:.3f} | "
            f"Plausibility: {epistemic_state['plausibility'][i]:.3f}"
        )

    print("\nIgnorance Mass      :", f"{epistemic_state['ignorance']:.3f}")
    print("Predictive Entropy  :", f"{epistemic_state['entropy']:.3f}")
    print("\nDecision            :", decision)
    print("Selected Hypothesis :", HYPOTHESES[idx])

    # 🔹 Explanation
    explanation = generate_explanation(epistemic_state, decision, idx)
    print("\nExplanation:")
    print(explanation)
    print("\nImage Quality (Contextual Information):")
    print(generate_quality_explanation(is_blurry, blur_score))



def generate_quality_explanation(is_blurry: bool, blur_score: float) -> str:
    if not is_blurry:
        return (
            "The uploaded image exhibits sufficient sharpness for diagnostic analysis. "
            f"The measured focus score (variance = {blur_score:.1f}) indicates adequate "
            "structural detail for reliable evidence extraction."
        )

    return (
        "The uploaded image appears to be of low visual quality due to blurring. "
        f"The measured focus score (variance = {blur_score:.1f}) suggests loss of fine "
        "structural detail, which can weaken evidential discrimination. "
        "This degradation may contribute to increased epistemic uncertainty and "
        "can justify deferral or abstention."
    )



def detect_blur(image_path: str, threshold: float = 100.0):
    """
    Detects image blur using Laplacian variance.
    Returns (is_blurry, blur_score).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return True, 0.0

    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    is_blurry = laplacian_var < threshold
    return is_blurry, laplacian_var



# ============================================================
# USER INTERFACE (UPLOAD MODE)
# ============================================================

def upload_and_run(model):
    print("\n=== Select Chest X-ray Image ===")

    root = tk.Tk()
    root.withdraw()          # Hide the main window
    root.attributes('-topmost', True)

    image_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not image_path:
        print("❌ No image selected.")
        return

    print(f"✔ Selected image: {image_path}")
    is_blurry, blur_score = detect_blur(image_path)


    image_tensor = preprocess_image(image_path)
    z = extract_evidence(model, image_tensor)
    epistemic_state = compute_epistemic_state(z)
    decision, idx = make_decision(epistemic_state)

    display_results(epistemic_state, decision, idx, is_blurry, blur_score)




# ============================================================
# ENTRY POINT
# ============================================================

def start_interface(model: nn.Module):
    while True:
        print("\n===================================")
        print(" Uncertainty-Aware Diagnostic System ")
        print("===================================")
        print("1 → Upload image")
        print("2 → Exit")

        choice = input("Select option (1/2): ").strip()

        if choice == "1":
            upload_and_run(model)
        elif choice == "2":
            print("Session terminated.")
            break
        else:
            print("Invalid selection.")
