# ============================================================
# Script: train_save_models.py
# Project: Breathing Sound Classification (Inhale vs Exhale)
# Description:
#   - Optuna로 탐색된 Best Trial 3개(Trial 8, 12, 54) 모델 학습
#   - 학습 완료된 모델을 .pt 파일로 각각 저장
#   - 향후 Ensemble을 위해 다양한 파라미터 조합 확보
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ 모델 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class CNNResSimple(nn.Module):
    def __init__(self, input_dim=112, dropout_rate=0.5):
        super().__init__()
        self.permute = lambda x: x.permute(0, 2, 1)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.resblock1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.resblock2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout_rate)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.permute(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.dropout_conv(x)
        avg_pool = self.gap(x)
        max_pool = self.gmp(x)
        x = torch.cat([avg_pool, max_pool], dim=1).squeeze(-1)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1)

# ✅ Feature 추출

def extract_mel_spec_from_y(y, sr, n_mels=64, n_fft=1024, hop_length=256):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0).T
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length).T
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length).T
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).T
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.concatenate([mfcc, delta, delta2], axis=0).T

    min_frames = min(mel_spec.shape[0], rms.shape[0], zcr.shape[0], contrast.shape[0], mfcc_all.shape[0])
    return np.concatenate([
        mel_spec[:min_frames],
        rms[:min_frames],
        zcr[:min_frames],
        contrast[:min_frames],
        mfcc_all[:min_frames]
    ], axis=1)

# ✅ 데이터 로딩 함수

def load_data(train_dir, train_df):
    actual_files = set(os.listdir(train_dir))
    X, y = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        base_id = row['ID'].replace('_I_', '_').replace('_E_', '_') + '.wav'
        if base_id not in actual_files:
            continue
        file_path = os.path.join(train_dir, base_id)
        try:
            audio, sr = librosa.load(file_path, sr=None)
            feat = extract_mel_spec_from_y(audio, sr)
            X.append(feat)
            y.append(1 if row['Target'] == 'I' else 0)
        except:
            continue
    return pad_sequences(X, padding='post', dtype='float32'), np.array(y)

# ✅ 학습 함수

def train_and_save_model(params, model_name):
    kaggle_data = '/content/drive/MyDrive/kaggle /kaggle'
    train_df = pd.read_csv(os.path.join(kaggle_data, 'train.csv'))
    X, y = load_data(os.path.join(kaggle_data, 'train'), train_df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()), batch_size=64)

    model = CNNResSimple(input_dim=112, dropout_rate=params['dropout_rate']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=params['gamma'])
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_state = None
    counter, patience = 0, 10

    for epoch in range(25):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss, y_true, y_prob = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(pred.cpu().numpy())

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), f"/content/drive/MyDrive/kaggle /kaggle/{model_name}.pt")
        print(f"✅ Saved {model_name}.pt")

# ✅ 학습 실행: trial 8/12/54
trial_params = [
    {"name": "trial_8", "params": {"dropout_rate": 0.5320, "lr": 0.000545, "weight_decay": 0.000285, "gamma": 0.5447}},
    {"name": "trial_12", "params": {"dropout_rate": 0.4456, "lr": 0.000531, "weight_decay": 6.18e-5, "gamma": 0.5326}},
    {"name": "trial_54", "params": {"dropout_rate": 0.3638, "lr": 0.002176, "weight_decay": 1.29e-5, "gamma": 0.5838}},
]

for trial in trial_params:
    train_and_save_model(trial['params'], trial['name'])
