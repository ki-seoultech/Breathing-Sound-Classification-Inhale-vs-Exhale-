# Breathing Sound Classification (Inhale vs Exhale)

## 📌 Project Overview
This project is a Kaggle-based audio classification competition.  
Given breathing sound `.wav` files, the goal was to develop a binary classifier that can distinguish **Inhale (I)** from **Exhale (E)**.  
We applied Residual CNNs, hyperparameter optimization (Optuna), and ensemble learning to maximize performance.  

---

## 🎯 Objectives
- Preprocess audio signals and extract meaningful features  
- Develop a Residual CNN model for binary classification  
- Optimize model hyperparameters using **Optuna (60 trials)**  
- Build a **2-Model Soft Voting Ensemble**  
- Tune prediction threshold based on validation F1 score  

---

## 🛠️ Techniques Used
- **Feature Engineering**: Mel-Spectrogram, RMS, Zero Crossing Rate, Spectral Contrast, MFCC + Delta, Delta-Delta  
- **Model**: Residual CNN with Global Average & Max Pooling  
- **Optimization**: Optuna-based hyperparameter search  
- **Ensemble**: Top 2 trial models (Trial 8, Trial 54) combined via Soft Voting  
- **Threshold Tuning**: Validation F1 optimization (best threshold = 0.43)  

---

## 📂 Code Structure
