# 🏞️ DeepScene: Scene Classification with CNNs

**DeepScene** is a deep learning-based image scene classification system that uses convolutional neural networks to identify the environment depicted in an image. It was built and trained using the Intel Image Classification dataset, and supports deployment-ready models for web applications.

---

## 🚀 Overview

- 📁 Uses the **Intel Image Classification** dataset with scenes like: `mountain`, `street`, `forest`, `coast`, `buildings`, and `glacier`.
- 🧠 Trains and evaluates multiple CNN architectures (including custom CNNs and pretrained models).
- 🔄 Includes preprocessing, augmentation, training loops, and model saving.
- 🖥️ A simple `webapp.py` script allows for basic deployment of the trained model.
- 📦 Model weights saved as `best.pt`.

---

## 🧠 Model Pipeline

1. **Dataset Loading**  
   Downloaded via `kagglehub` from `puneet6060/intel-image-classification`.

2. **Exploratory Analysis**  
   - Image size consistency check  
   - Class balance check  
   - Random image previews for visual inspection

3. **Data Augmentation**  
   Custom transformations using `PIL`, such as:
   - Horizontal flipping
   - Random rotations
   - Brightness enhancement

4. **Model Training**  
   - Training on labeled images using CNNs
   - Evaluation with validation accuracy and loss
   - Saving the best-performing model (`best.pt`)

5. **Web Deployment (WIP)**  
   - Basic interface using `webapp.py` for testing model predictions

---

## 🔮 DeepScene UI Preview

![DeepScene UI](https://drive.google.com/uc?export=view&id=1uLbF0Fbo7GVjOxDKocqpv4YrNNpkhttf)


---

## 📁 Project Structure

```
DeepScene/
├── code.ipynb         # Full training and evaluation notebook
├── best.pt            # Saved PyTorch model
├── webapp.py          # Simple app interface (in progress)
├── README.md          # Project documentation
```

---

## ⚙️ Installation & Setup

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Inference

Once trained:
```bash
python webapp.py
```
Upload an image and get the scene classification instantly.

---

## 📦 Requirements

- torch
- torchvision
- matplotlib
- pillow
- kagglehub
- streamlit (for webapp)

---

## 📄 License

MIT License — feel free to use, modify, and distribute with credit.

