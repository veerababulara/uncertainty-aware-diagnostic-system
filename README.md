# Entropy-Driven Uncertainty-Aware Diagnostic System with Explicit Ignorance

This repository contains the implementation of an **uncertainty-aware decision support system** for chest X-ray analysis.  
The system explicitly models **belief**, **plausibility**, and **ignorance**, and uses **entropy-driven evidential reasoning** to decide whether to **COMMIT**, **DEFER**, or **ABSTAIN** instead of forcing potentially unreliable predictions.

The implementation is intended for **research and decision-support purposes** and follows principles from **Approximate Reasoning** and **evidential uncertainty modeling**.
1
---

## ✨ Key Features

- Explicit modeling of **belief**, **plausibility**, and **ignorance**
- Entropy-driven uncertainty estimation
- Three epistemic outcomes:
  - **COMMIT** – sufficient epistemic support
  - **DEFER** – partial but unreliable evidence
  - **ABSTAIN** – dominant ignorance
- Automatic **image quality (blur) detection**
- Human-readable **epistemic explanations**
- Safe handling of ambiguous and low-quality inputs
- Designed to avoid forced predictions under uncertainty

---

## 📂 Project Structure

.
├── inference.py # Core epistemic reasoning and inference logic
├── main.py # Entry point and user interaction loop
├── model.pt # Trained CNN used as an evidence generator
├── requirements.txt # Python dependencies
├── Unclear_Images/ # (Optional) extracted low-quality images
└── README.md


---

## 🧠 Datasets Used

This project studies epistemic uncertainty using **multiple large-scale chest X-ray datasets**, selected specifically for their **label noise**, **diagnostic ambiguity**, and **clinical variability**.

> **Note:** The datasets themselves are **not redistributed** in this repository.  
> Users must download them from the official sources listed below.

---

### 🩻 NIH ChestX-ray14

- **Samples:** 112,120  
- **Description:**  
  One of the earliest large-scale chest X-ray datasets.  
  Known for **weak supervision** and **label noise**, making it suitable for studying uncertainty under imperfect annotations.
- **Usage in this project:**  
  Used to evaluate abstention and deferral behavior under noisy and degraded inputs.
- **Link:**  
  https://nihcc.app.box.com/v/ChestXray-NIHCC

---

### 🩻 CheXpert

- **Samples:** 224,316  
- **Description:**  
  Large clinical chest X-ray dataset explicitly designed to represent **diagnostic uncertainty**, including uncertain and conflicting labels.
- **Usage in this project:**  
  Primary dataset for epistemic reasoning and uncertainty-aware decision analysis.
- **Link:**  
  https://stanfordmlgroup.github.io/competitions/chexpert/

---

### 🩻 MIMIC-CXR

- **Samples:** 377,110  
- **Description:**  
  Very large, heterogeneous dataset derived from real clinical radiology reports.  
  Contains substantial annotation variability and ambiguity.
- **Usage in this project:**  
  Used to assess cross-dataset robustness and generalization of abstention behavior.
- **Link:**  
  https://physionet.org/content/mimic-cxr/

---

### 🩻 PadChest

- **Samples:** 160,868  
- **Description:**  
  Multi-label chest X-ray dataset with diverse radiological findings.  
  Includes **label imbalance** and **semantic overlap**, which are useful for testing abstention behavior.
- **Usage in this project:**  
  Used to study uncertainty handling under class overlap and semantic ambiguity.
- **Link:**  
  https://bimcv.cipf.es/bimcv-projects/padchest/

---

## 🚀 How to Run (Local)

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run the system
python main.py
3️⃣ Upload an image
A local file picker will open

Select a chest X-ray image (.jpg, .png)

The system outputs:

Belief and plausibility for each hypothesis

Ignorance mass and predictive entropy

Decision: COMMIT / DEFER / ABSTAIN

Epistemic explanation

Image quality assessment

🖼 Input and Output Examples
📥 Input Images
(Place example chest X-ray images here)

examples/input/
├── clear_image.jpg
├── ambiguous_image.jpg
└── blurry_image.jpg
📤 Output Example
Decision            : ABSTAIN
Selected Hypothesis : Pleural Effusion

Explanation:
The system abstains from diagnosis because epistemic ignorance dominates
the evidential state. High predictive entropy indicates insufficient
discriminative evidence.

Image Quality (Contextual Information):
The uploaded image exhibits sufficient sharpness for diagnostic analysis.
(Screenshots or text logs of outputs can be placed here)

⚠️ Disclaimer
This software is intended solely for research and decision-support purposes.
It does not provide medical diagnoses and must not be used for clinical decision-making without qualified expert oversight.

📌 Citation
If you use this code in your research, please cite the associated article:

Entropy-Driven Evidential Reasoning with Explicit Ignorance for Uncertainty-Aware Decision Support

📬 Contact
For questions, feedback, or collaboration, please open an issue in this repository or contact the authors.

