
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HYPOTHESES = [
    "Pneumonia",
    "Lung Opacity",
    "Pleural Effusion"
]

BELIEF_THRESHOLD = 0.5          # τ in the paper
UNCERTAINTY_EPS = 1e-8



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



def extract_evidence(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Produces raw evidential activations z_i.
    """
    with torch.no_grad():
        z = model(image_tensor).squeeze(0)
    return z
def explain_abstention_cause(epistemic_state: dict) -> str:
    belief = epistemic_state["belief"]
    ignorance = epistemic_state["ignorance"]
    entropy = epistemic_state["entropy"]

    max_belief = belief.max().item()

    reasons = []

    if max_belief < BELIEF_THRESHOLD:
        reasons.append(
            f"no hypothesis achieved sufficient belief "
            f"(maximum belief = {max_belief:.2f} < threshold = {BELIEF_THRESHOLD:.2f})"
        )

    if ignorance >= (1 - BELIEF_THRESHOLD):
        reasons.append(
            f"epistemic ignorance remains high "
            f"(ignorance = {ignorance:.2f})"
        )

    if entropy > 0.5:
        reasons.append(
            f"predictive entropy is elevated "
            f"(H = {entropy:.2f})"
        )

    return "The system abstains because " + "; and ".join(reasons) + "."


def compute_epistemic_state(z: torch.Tensor):
    """
    Implements Equations (7)–(13) from the paper.
    """
    e = torch.exp(z)
    S = torch.sum(e)

    p = e / (S + UNCERTAINTY_EPS)

    entropy = -torch.sum(p * torch.log(p + UNCERTAINTY_EPS))
    entropy = entropy / torch.log(torch.tensor(len(HYPOTHESES), device=DEVICE))

    belief = e / (S + 1.0)
    ignorance = 1.0 / (S + 1.0)
    plausibility = belief + ignorance

    return {
        "belief": belief.cpu(),
        "plausibility": plausibility.cpu(),
        "ignorance": ignorance.cpu().item(),
        "entropy": entropy.cpu().item()
    }

def generate_explanation(epistemic_state: dict, decision: str, idx: int) -> str:
    belief = epistemic_state["belief"][idx].item()
    ignorance = epistemic_state["ignorance"]
    entropy = epistemic_state["entropy"]
    hypothesis = HYPOTHESES[idx]

    if decision == "COMMIT":
        return (
            f"The system commits to '{hypothesis}' because the belief "
            f"assigned to this hypothesis is sufficiently high "
            f"(belief = {belief:.2f}), while epistemic ignorance remains limited "
            f"(ignorance = {ignorance:.2f}). "
            f"The low-to-moderate predictive entropy (H = {entropy:.2f}) "
            f"indicates reliable evidential discrimination among hypotheses."
        )

    return (
        f"The system abstains from diagnosis because epistemic ignorance "
        f"dominates the evidential state (ignorance = {ignorance:.2f}). "
        f"The predictive entropy (H = {entropy:.2f}) reflects weak or "
        f"conflicting evidence, making reliable discrimination among "
        f"hypotheses unjustified. Abstention is therefore issued as a "
        f"safety-oriented and epistemically justified outcome."
    )



def make_decision(epistemic_state: dict):
    belief = epistemic_state["belief"]
    ignorance = epistemic_state["ignorance"]

    dominant_idx = torch.argmax(belief).item()
    dominant_belief = belief[dominant_idx].item()

    if dominant_belief >= BELIEF_THRESHOLD and ignorance < (1 - BELIEF_THRESHOLD):
        return "COMMIT", dominant_idx
    else:
        return "ABSTAIN", dominant_idx



def generate_quality_explanation(is_blurry: bool, blur_score: float) -> str:
    if not is_blurry:
        return (
            "The uploaded image exhibits sufficient optical sharpness for feature "
            "and evidence extraction by the model. "
            f"The measured focus score (variance = {blur_score:.1f}) indicates that "
            "fine structural details are preserved. "
            "Optical sharpness alone does not guarantee full diagnostic adequacy, "
            "which is evaluated through the system’s evidential reasoning process."
        )

    return (
        "The uploaded image appears to be affected by optical blurring. "
        f"The measured focus score (variance = {blur_score:.1f}) suggests reduced "
        "edge definition and loss of fine structural detail. "
        "This degradation can weaken evidence extraction and increase epistemic "
        "uncertainty, supporting deferral or abstention."
    )



def detect_blur(image_path: str, threshold: float = 100.0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return True, 0.0

    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    is_blurry = laplacian_var < threshold
    return is_blurry, laplacian_var
