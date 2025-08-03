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

## 🔍 Additional Result Analysis

During final evaluation, we compared different ensemble strategies:

- **Weighted Soft Voting**  
  - Private Score: **0.78333**  
  - Public Score: 0.77285  

- **2-Model Ensemble (Trial 8 + Trial 54)**  
  - Private Score: 0.76666  
  - Public Score: 0.78000  

- **3-Model Ensemble**  
  - Private Score: 0.77000  
  - Public Score: 0.77428  

From these results, we found that:
- Weighted 2-Model Ensemble achieved the **highest private test accuracy (0.78333)**.
- Although its public score was slightly lower, it generalized better to unseen test data.
- This highlights that **weighted voting** with proper adjustment can outperform simple averaging.

💡 **Key Insight:**  
Deep learning models show significant performance variation depending on input data distribution.  
To improve generalization, carefully adjusting model weights in ensembles is often more effective than relying solely on a larger number of models or simple averaging.
