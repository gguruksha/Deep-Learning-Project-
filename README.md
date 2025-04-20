# 🔍 Violence Detection in Video Using Deep Learning

This project explores a deep learning–based framework to automatically detect violent content in videos. With growing video surveillance and online video content, the need for scalable, real-time violence detection is more critical than ever. This solution combines both static frame analysis and spatiotemporal modeling for accurate classification, supported by a user-friendly Streamlit interface.

🏅 This was a group project submitted as part of a graduate-level data science course.

---

## 📁 Project Structure

The project consists of:

- 📦 A static frame classifier using ResNet-50  
- 🎥 A temporal model using R3D-18 (3D CNN)  
- 🖥️ A Streamlit-based interactive frontend  
- 📊 Real-time prediction with video annotation output  

---

## 📌 Problem Statement

Violent incidents captured on video are on the rise, creating a pressing need for automated monitoring. Manual monitoring is error-prone and resource-intensive. This project tackles:

- How can we identify violence in real-time from video input?  
- Can deep learning models outperform traditional rule-based systems?  
- How can we make the results interpretable and user-friendly?  

---

## 📊 Dataset

- Source: [Real Life Violence Situations Dataset on Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)  
- 1000 violent + 1000 non-violent videos  
- Preprocessed for frame quality, consistency, and balanced class distribution  
- Train-test split: 80:20  

---

## 🧠 Models Used

### 1. Static Model: ResNet-50 (Image-Level)

- Pretrained on ImageNet  
- Fine-tuned for binary classification (violence vs. non-violence)  
- Achieved:  
  - ✅ 97.99% training accuracy  
  - ✅ 95.14% test accuracy  
- Visualizations: Grad-CAM for interpretability  

### 2. Temporal Model: R3D-18 (Video-Level)

- Captures spatiotemporal patterns across frames  
- Trained on 16-frame clips  
- Pretrained on Kinetics-400  
- Modified final FC layer with sigmoid for binary classification  
- Achieved:  
  - ✅ 92.77% test accuracy  
  - ✅ 93.18% F1 score  

---

## 💻 Streamlit Frontend

- Drag-and-drop video upload  
- Real-time inference and annotated video preview  
- Video metadata preview (duration, FPS, resolution)  
- Sidebar with model architecture details  
- Downloadable annotated output  
- Custom styling and responsive feedback  

---


## 🚀 How to Run

1. Clone this repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the frontend:

```bash
streamlit run app.py
```
---

---
## 🎯 Results

- ✅ High accuracy and generalization with both static and temporal models  
- ✅ Seamless user experience with visual cues and annotated output  
- ✅ Strong baseline for deploying real-world violence detection systems  

---

## 👥 Contributors
This project was developed as a team effort. Forked and maintained here by Guruksha Gurnani

---






