# ============================================================
# Script: ensemble_predict.py
# Project: Breathing Sound Classification (Inhale vs Exhale)
# Description:
#   - Best Trial 모델 2개(Trial 8, 54) 로드
#   - Soft Voting 기반 앙상블 예측 수행
#   - Validation 기반 최적 Threshold 적용 (0.43)
#   - 최종 CSV 파일(submission_ensemble_2models.csv) 생성
# ============================================================

# ✅ Soft Voting <가장 성능향상 가능성 큰 후보모델 2개 ensemble - Trial 8 , 54>

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm
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
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length).T
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

# ✅ 테스트셋 로딩 및 전처리
kaggle_data = '/content/drive/MyDrive/kaggle /kaggle'
TEST_DIR = os.path.join(kaggle_data, 'test')
TEST_CSV = os.path.join(kaggle_data, 'test.csv')
test_df = pd.read_csv(TEST_CSV)

X_test, valid_ids = [], []
for fname in tqdm(test_df['ID']):
    base_id = fname.replace('_I_', '_').replace('_E_', '_')
    wav_name = base_id if base_id.endswith('.wav') else base_id + '.wav'
    file_path = os.path.join(TEST_DIR, wav_name)

    if not os.path.exists(file_path):
        X_test.append(np.zeros((1, 112)))
    else:
        try:
            y, sr = librosa.load(file_path, sr=None)
            feat = extract_mel_spec_from_y(y, sr)
        except:
            feat = None
        if feat is not None:
            X_test.append(feat)
        else:
            X_test.append(np.zeros((1, 112)))
    valid_ids.append(fname)

X_test_pad = pad_sequences(X_test, padding='post', dtype='float32')
X_test_tensor = torch.tensor(X_test_pad, dtype=torch.float32)

# ✅ Soft Voting (2-Model Ensemble: Trial 8 + Trial 54)
ensemble_settings = [
    {"model_path": os.path.join(kaggle_data, "trial_8.pt"), "dropout_rate": 0.5320},
    {"model_path": os.path.join(kaggle_data, "trial_54.pt"), "dropout_rate": 0.3638},
]

ensemble_probs = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_test_tensor = X_test_tensor.to(device)

for setting in ensemble_settings:
    model = CNNResSimple(input_dim=112, dropout_rate=setting['dropout_rate']).to(device)
    model.load_state_dict(torch.load(setting['model_path'], map_location=device))
    model.eval()
    y_test_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), 64):
            batch = X_test_tensor[i:i+64]
            preds = model(batch)
            y_test_probs.extend(preds.cpu().numpy())
    ensemble_probs.append(np.array(y_test_probs))

# ✅ 평균 + 최적 threshold(0.43) 적용
avg_probs = np.mean(ensemble_probs, axis=0)  # 2개 모델에 대해서만 계산됨
final_preds = (avg_probs > 0.43).astype(int)
final_labels = ['I' if p == 1 else 'E' for p in final_preds]


# ✅ 제출 파일 생성
submission = pd.DataFrame({"ID": valid_ids, "Target": final_labels})
submission.to_csv("submission_ensemble_2models.csv", index=False)
print("✅ submission_ensemble_2models.csv 저장 완료")

