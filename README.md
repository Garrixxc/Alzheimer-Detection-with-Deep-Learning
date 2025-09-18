# 🧠 Alzheimer’s Disease Detection with Deep Learning

Early diagnosis of Alzheimer’s can slow progression and improve patient outcomes.  
This project applies **Convolutional Neural Networks (CNNs)** and **EfficientNet** to MRI scans for automatic classification of Alzheimer’s severity.

---

## 🚀 Project Highlights

- **Dataset**: OASIS MRI dataset (~85,000 scans)
- **Classes**: 
  - None (67,222 scans)  
  - Very Mild (13,725 scans)  
  - Mild (5,002 scans)  
  - Moderate (488 scans)  
- **Models**: 
  - Custom CNN (**33.6M parameters**)  
  - EfficientNetB0 (transfer learning)  
- **Performance**:  
  - CNN → **96.6% accuracy**  
  - EfficientNetB0 → **93.9% accuracy**  
- **Techniques Used**: Class rebalancing, dropout, batch normalization, early stopping, learning rate tuning

---

## 📂 Repository Structure

```

.
├── notebooks/                # Jupyter notebooks for EDA, training, evaluation
│   └── AlzheimerDetection.ipynb
├── src/                      # Python scripts for modular training
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
├── saved\_models/             # Trained models
├── requirements.txt          # Dependencies
└── README.md                 # Project overview

````

---

## ⚙️ Installation

```bash
# clone this repository
git clone https://github.com/<your-username>/Alzheimer-Detection-with-Deep-Learning.git
cd Alzheimer-Detection-with-Deep-Learning

# create environment
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# install dependencies
pip install -r requirements.txt
````

---

## 📊 Training & Evaluation

### Run via Notebook

Open the Jupyter notebook and execute all cells:

```bash
jupyter lab
```

### Run via Script

Train CNN:

```bash
python src/train.py --model cnn --epochs 20 --batch_size 16
```

Train EfficientNetB0:

```bash
python src/train.py --model efficientnet_b0 --epochs 20 --batch_size 64
```

Evaluate a saved model:

```bash
python src/eval.py --model_path saved_models/cnn.keras
```

---

## 🧠 Inference Example

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

CLASSES = ["None", "Very Mild", "Mild", "Moderate"]

model = load_model("saved_models/cnn.keras")
img = Image.open("sample.jpg").resize((128,128))
x = np.array(img)[None, ...] / 255.0
pred = model.predict(x)
print(CLASSES[np.argmax(pred)], pred)
```

---

## ✅ Results

| Model          | Accuracy | Notes                             |
| -------------- | -------- | --------------------------------- |
| CNN            | 96.6%    | Strong overall performance        |
| EfficientNetB0 | 93.9%    | Better recall on minority classes |

* CNN generalized well with minimal overfitting by epoch \~10.
* EfficientNetB0 excelled at **minority class detection** due to rebalancing.

---

## ⚖️ Ethics & Disclaimer

* This project is for **research and educational purposes only**.
* Not intended for clinical or diagnostic use.
* Always ensure ethical handling of medical data.

---

## 📚 References

* OASIS Dataset
* Project Report & Presentation

---

## 📝 Citation

```
@project{alz_detect_2025,
  title   = {Alzheimer’s Disease Detection with Deep Learning},
  author  = {Gaurav Salvi et al.},
  year    = {2025},
  note    = {CNN + EfficientNet classification of MRI scans}
}
```


