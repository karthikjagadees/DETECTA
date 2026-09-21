# DETECTA

### 🔬 Multi-Model Camouflaged-Object Detection & Animal Recognition

**DETECTA** is a deep-learning system designed to find **hard-to-see animals hidden in natural environments**. It combines multiple camouflage-segmentation architectures with animal recognition to identify objects that are difficult to distinguish from their surroundings.

The system can:

* 🔍 Detect camouflaged objects
* 🎯 Segment the detected animal from its environment
* 🧠 Compare multiple deep-learning segmentation models
* ✂️ Extract the detected animal
* 🐾 Recognize the animal using a **69-class COD10K classifier**
* 📊 Compare model performance using official benchmark metrics
* 🖥️ Provide an interactive **Streamlit dashboard**

> 🔄 **Project Evolution:** DETECTA was previously developed under the internal names **TUKU Deep Learning** and **Camouflage Breaker**. The existing datasets, checkpoints, architectures, and model behavior remain unchanged.

---

## 🚀 Key Features

### 🧩 Multi-Model Camouflage Detection

DETECTA supports **three segmentation architectures** operating on the same input image:

| Model           | Architecture       | Role                            |
| --------------- | ------------------ | ------------------------------- |
| 🟦 **ResUNet**  | ResNet50 Encoder   | Default segmentation model      |
| 🟩 **SINet-V2** | Res2Net50 Backbone | Camouflage-focused segmentation |
| 🟨 **ESCNet**   | PVT-v2 B4          | Fine-tuned segmentation option  |

This allows DETECTA to compare different approaches to the same camouflage-detection problem.

### 🐾 Animal Recognition

After segmentation, the detected object is cropped and passed to a shared:

**ResNet50 classifier → 69 COD10K animal classes**

An optional **OpenCLIP refinement stage** can provide additional semantic recognition information.

### ⚔️ Compare Mode

DETECTA can run all supported segmentation models on the same image:

```text
Input Image
     │
     ├── 🟦 ResUNet
     │
     ├── 🟩 SINet-V2
     │
     └── 🟨 ESCNet
            │
            ▼
     Best Detection
            │
            ▼
       Object Crop
            │
            ▼
      🧠 ResNet50
            │
            ▼
     🐾 Animal Class
```

This makes it possible to visually and quantitatively compare how different models handle difficult camouflage.

---

# 🧠 Model Architecture

DETECTA currently uses the following model stack:

| Component         | Architecture                     | Purpose                                 |
| ----------------- | -------------------------------- | --------------------------------------- |
| 🟦 **ResUNet**    | ResNet50 Encoder + U-Net Decoder | Default camouflaged-object segmentation |
| 🟩 **SINet-V2**   | Res2Net50 Backbone               | Camouflage-specific segmentation        |
| 🟨 **ESCNet**     | PVT-v2 B4                        | Fine-tuned segmentation                 |
| 🐾 **Classifier** | ResNet50                         | 69-class animal recognition             |
| 🔤 **OpenCLIP**   | Vision-Language Model            | Optional semantic refinement            |

---

# 📊 Official COD10K-v3 Test Benchmark

DETECTA uses the **official COD10K-v3 Test set** as a frozen evaluation benchmark.

**Test set size: 4,000 images**

The model registry currently reports:

| Segmentation Model |        IoU |       Dice |
| ------------------ | ---------: | ---------: |
| 🟦 ResUNet         | **0.5710** | **0.6200** |
| 🟩 SINet-V2        | **0.7425** | **0.7908** |
| 🟨 ESCNet          | **0.7892** | **0.8304** |

> 📌 These benchmark values are surfaced from the project's **model registry** rather than being manually hardcoded into the Streamlit interface.

### 📐 Evaluation Metrics

**IoU — Intersection over Union**

Measures the overlap between the predicted segmentation mask and the ground-truth mask.

**Dice Score**

Measures segmentation similarity between prediction and ground truth, with greater sensitivity to overlap.

---

# 🖥️ Interactive Dashboard

DETECTA includes a **Streamlit-based interface** for visual experimentation.

The dashboard supports:

### 📤 Image Upload

Upload a natural scene containing a potentially camouflaged animal.

### 🔬 Multi-Model Comparison

Run:

* 🟦 ResUNet
* 🟩 SINet-V2
* 🟨 ESCNet

on the same image.

### 🎭 Segmentation Visualization

Inspect the predicted camouflage mask and detected object.

### ✂️ Object Extraction

The detected region can be isolated and cropped.

### 🐾 Animal Recognition

The cropped object is passed through the 69-class ResNet50 classifier.

### 📊 Model Inspection

Compare segmentation outputs and benchmark information directly inside the dashboard.

---

# 📁 Repository Structure

```text
DETECTA/
│
├── 🧠 Camouflage_Breaker-main/
│   │
│   ├── app.py                    # 🖥️ Streamlit dashboard
│   │
│   ├── config/
│   │   ├── model_registry.*      # 📊 Model & benchmark registry
│   │   └── escnet.json           # ⚙️ ESCNet configuration
│   │
│   ├── inference/
│   │   └── ...                   # 🔍 Detection & classification pipeline
│   │
│   ├── models/
│   │   └── ...                   # 🧠 Model architectures & wrappers
│   │
│   ├── training/
│   │   └── ...                   # 🏋️ Training scripts
│   │
│   ├── evaluation/
│   │   └── ...                   # 📊 Benchmark & comparison scripts
│   │
│   ├── saved_models/
│   │   └── ...                   # 💾 Trained checkpoints
│   │
│   ├── dataset/
│   │   └── ...                   # 🗂️ COD10K-v3 dataset
│   │
│   └── requirements.txt          # 📦 Python dependencies
│
├── 🌿 TUKU_DATASET_V2/
│   └── README.md                 # 🔬 Hard-camouflage research track
│
├── 📄 LICENSE                    # MIT License
│
└── 📘 README.md
```

---

# 🗂️ Dataset

## 🏆 COD10K-v3

Production training and evaluation use **COD10K-v3**.

```text
Camouflage_Breaker-main/dataset/

├── Train/
│   ├── Image/
│   └── GT_Object/
│
├── Test/
│   ├── Image/
│   └── GT_Object/
│
└── Info/
    ├── CAM_train.txt
    └── CAM_test.txt
```

### 🔒 Official Test Set

The **COD10K-v3 Test set contains 4,000 images** and is treated as a frozen benchmark for evaluation.

---

# 🌿 TUKU_DATASET_V2

`TUKU_DATASET_V2` is a **separate hard-camouflage research dataset** created for future experimentation beyond the standard COD10K training distribution.

Its purpose is to investigate more difficult camouflage scenarios and support future research.

### ⚠️ Dataset Separation Rule

**TUKU_DATASET_V2 must not contain Official COD10K Test images or masks.**

This separation helps prevent benchmark contamination and keeps the official evaluation track independent.

See:

```text
TUKU_DATASET_V2/README.md
```

for dataset-specific information.

---

# 🏋️ Training & Evaluation

All commands below are executed from:

```bash
cd Camouflage_Breaker-main
```

## 🟩 SINet-V2 Training

```bash
python training/train_sinetv2.py
```

### 🧪 Smoke / Low-VRAM Training

For a lightweight test run:

```bash
set SINET_EPOCHS=1
set SINET_BATCH_SIZE=2
set SINET_MAX_SAMPLES=400

python training/train_sinetv2.py
```

---

## 📊 Evaluate SINet-V2

```bash
python evaluation/evaluate_sinetv2.py
```

## 📊 Evaluate ResUNet

```bash
python evaluation/evaluate_resunet.py
```

## ⚔️ Build Model Comparison

```bash
python evaluation/build_model_comparison.py
```

Evaluation metrics are written to:

```text
outputs/
```

The Streamlit application reads these generated metrics rather than manually embedding benchmark values inside the UI.

---

# 💾 Model Checkpoints

Place trained checkpoints under:

```text
saved_models/
```

Expected components include:

```text
saved_models/
│
├── resunet_best.pth
├── classifier_best.pth
│
└── sinetv2/
    └── sinetv2_cod10k_best.pth
```

The ESCNet checkpoint location is configured through:

```text
config/escnet.json
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/karthikjagadees/DETECTA.git
```

```bash
cd DETECTA/Camouflage_Breaker-main
```

## 2️⃣ Create a Virtual Environment

### 🪟 Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 🐧 Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Add Model Checkpoints

Place the required trained weights inside `saved_models/`.

## 5️⃣ Launch DETECTA

```bash
streamlit run app.py
```

The dashboard should then be available through the Streamlit local server.

---

# 🖥️ Hardware & GPU Support

### 💻 Recommended Environment

```text
Python      : 3.10+
PyTorch     : CUDA-enabled build recommended
GPU         : NVIDIA CUDA-capable GPU
RAM         : Depends on model and batch size
```

CPU inference is supported, although GPU acceleration is strongly recommended for practical experimentation.

---

# ⚡ RTX 50-Series / Blackwell Note

Some older PyTorch CUDA wheels do not contain kernels compatible with the **`sm_120`** architecture used by RTX 50-series GPUs.

For example, certain older `cu124` builds may fail to execute GPU kernels correctly and can fall back to CPU.

### 🔧 Recommendation

For RTX 50-series hardware:

1. Install a PyTorch build with appropriate Blackwell support.
2. Verify CUDA availability.
3. Confirm that the GPU is being used before starting training.

Example verification:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

# 🔬 Research Direction

DETECTA is structured to support research beyond the current COD10K benchmark.

Future development can include:

* 🌳 Harder real-world camouflage environments
* 📱 Real-time mobile-camera inference
* 🎥 Live video camouflage detection
* 🧠 Additional segmentation architectures
* 🐛 Expanded animal and insect classes
* 🌍 Cross-environment evaluation
* ⚡ Model optimization for edge devices
* 📦 ONNX / TensorRT deployment
* 🔬 New hard-camouflage datasets
* 📊 More comprehensive benchmark analysis

The long-term direction is to move from **static image analysis** toward **real-time environmental camouflage detection**.

---

# 📜 License

This project is released under the:

**MIT License**

Third-party models and datasets retain their respective licenses.

This includes, but is not limited to:

* SINet-V2
* COD10K
* ESCNet
* ResNet / Res2Net components
* OpenCLIP
* Other third-party dependencies

Please review and comply with the original licenses before redistributing models, datasets, or derived assets.

---

# 🙏 Acknowledgements

DETECTA builds upon work from the broader computer-vision and camouflage-detection research community.

### 🏆 COD10K

Camouflaged Object Detection benchmark and dataset.

🔗 https://github.com/DengPingFan/COD10K

### 🧠 SINet-V2

Camouflaged-object detection architecture.

🔗 https://github.com/GewelsJI/SINet-V2

### 🤖 Additional Technologies

* PyTorch
* ResNet
* Res2Net
* PVT-v2
* Segmentation Models
* OpenCLIP
* Streamlit
* OpenCV
* The broader computer-vision research community

-----

# 🦎 DETECTA

### **See what the eye misses.**

> **Detect. Segment. Recognize.**

🔬 **Multi-model camouflage intelligence for difficult natural environments.**
