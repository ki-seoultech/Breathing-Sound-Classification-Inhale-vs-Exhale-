# ============================================================
# Script: train_optuna.py
# Project: Breathing Sound Classification (Inhale vs Exhale)
# Description:
#   - Residual CNN 모델 기반
#   - Optuna를 사용하여 60 Trial 하이퍼파라미터 최적화
#   - Dropout, Learning Rate, Weight Decay, Scheduler Gamma 탐색
#   - Validation F1 Score 기반 Best Threshold 도출
# ============================================================

  # 동일한 모델에서 하이퍼 파라미터 최적화, optuma 60으로 best parameter 도출

  from google.colab import drive
  drive.mount('/content/drive')

  !pip install optuna
  import os
  import numpy as np
  import pandas as pd
  import librosa
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import Dataset, DataLoader
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import f1_score, accuracy_score, classification_report
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  import optuna
  from tqdm import tqdm


  # ✅ device 설정
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")

  # ✅ 경로 설정
  kaggle_data = '/content/drive/MyDrive/kaggle /kaggle'
  TRAIN_DIR = os.path.join(kaggle_data, 'train')
  TEST_DIR = os.path.join(kaggle_data, 'test')
  TRAIN_CSV = os.path.join(kaggle_data, 'train.csv')
  TEST_CSV = os.path.join(kaggle_data, 'test.csv')
  train_df = pd.read_csv(TRAIN_CSV)

  # ✅ Feature 추출 함수
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
      mel_spec = mel_spec[:min_frames]
      rms = rms[:min_frames]
      zcr = zcr[:min_frames]
      contrast = contrast[:min_frames]
      mfcc_all = mfcc_all[:min_frames]

      return np.concatenate([mel_spec, rms, zcr, contrast, mfcc_all], axis=1)

  # ✅ Dataset 로딩

  def load_data():
      actual_files = set(os.listdir(TRAIN_DIR))
      X, y = [], []

      for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
          base_id = row['ID'].replace('_I_', '_').replace('_E_', '_') + '.wav'
          if base_id not in actual_files:
              continue
          file_path = os.path.join(TRAIN_DIR, base_id)
          try:
              audio, sr = librosa.load(file_path, sr=None)
              feat = extract_mel_spec_from_y(audio, sr)
              X.append(feat)
              y.append(1 if row['Target'] == 'I' else 0)
          except:
              continue

      X_pad = pad_sequences(X, padding='post', dtype='float32')
      y_arr = np.array(y)
      return X_pad, y_arr


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

  # ✅ 전역 학습/검증 데이터 재정의
  X_all, y_all = load_data()
  X_train, X_val, y_train, y_val = train_test_split(
      X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
  )

  # ✅ Optuna + threshold 튜닝 포함 objective 함수
  def objective(trial):
      dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7)
      lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
      weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
      gamma = trial.suggest_float("gamma", 0.3, 0.9)

      model = CNNResSimple(input_dim=112, dropout_rate=dropout_rate).to(device)
      criterion = nn.BCELoss()
      optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
      scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

      train_loader = DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
      val_loader = DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64, shuffle=False)

      for epoch in range(10):
          model.train()
          for xb, yb in train_loader:
              xb, yb = xb.to(device), yb.to(device)
              optimizer.zero_grad()
              pred = model(xb)
              loss = criterion(pred, yb.float())
              loss.backward()
              optimizer.step()
          scheduler.step()

      # Threshold 튜닝 포함
      model.eval()
      y_true, y_prob = [], []
      with torch.no_grad():
          for xb, yb in val_loader:
              xb = xb.to(device)
              prob = model(xb).cpu().numpy()
              y_prob.extend(prob)
              y_true.extend(yb.numpy())

      best_f1, best_th = 0, 0.5
      for th in np.arange(0.3, 0.71, 0.01):
          pred = (np.array(y_prob) > th).astype(int)
          f1 = f1_score(y_true, pred)
          if f1 > best_f1:
              best_f1 = f1
              best_th = th

      trial.set_user_attr("best_threshold", best_th)
      return best_f1

  # ✅ Optuna 실행
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=60)

  # ✅ 최종 best threshold 확인
  best_th = study.best_trial.user_attrs["best_threshold"]
  print("📌 Best Threshold:", best_th)
  print("📌 Best Params:", study.best_params)

  best_params = study.best_params



  # ✅ 전역 변수 준비 (앞 코드에서 정의된 X_train, X_val, y_train, y_val 사용)
  X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
  y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

  train_ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
  val_ds = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
  train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

  # ✅ 모델 초기화 (최적 파라미터 적용)
  model = CNNResSimple(input_dim=112, dropout_rate=best_params['dropout_rate']).to(device)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=best_params['gamma'])

  # ✅ 학습 루프
  best_val_loss = float('inf')
  best_model_state = None
  num_epochs = 25
  patience = 10
  counter = 0

  for epoch in range(num_epochs):
      model.train()
      total_loss, correct, total = 0.0, 0, 0
      for Xb, yb in train_loader:
          Xb, yb = Xb.to(device), yb.to(device)
          optimizer.zero_grad()
          outputs = model(Xb)
          loss = criterion(outputs, yb)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
          preds = (outputs > 0.5).int()
          correct += (preds == yb.int()).sum().item()
          total += yb.size(0)

      val_loss, correct, total, y_true, y_prob = 0.0, 0, 0, [], []
      model.eval()
      with torch.no_grad():
          for Xb, yb in val_loader:
              Xb, yb = Xb.to(device), yb.to(device)
              outputs = model(Xb)
              loss = criterion(outputs, yb)
              val_loss += loss.item()
              preds = (outputs > 0.5).int()
              correct += (preds == yb.int()).sum().item()
              total += yb.size(0)
              y_true.extend(yb.cpu().numpy())
              y_prob.extend(outputs.cpu().numpy())

      val_loss /= len(val_loader)
      val_acc = correct / total

      print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model_state = model.state_dict()
          best_val_probs = y_prob
          best_val_trues = y_true
          counter = 0
      else:
          counter += 1
          if counter >= patience:
              print(f"⛔ Early stopping at epoch {epoch+1}")
              break

      scheduler.step()

  if best_model_state:
      model.load_state_dict(best_model_state)

  # ✅ Threshold 튜닝
  best_th, best_f1 = 0.5, 0
  for th in np.arange(0.3, 0.71, 0.01):
      preds = (np.array(best_val_probs) > th).astype(int)
      f1 = f1_score(np.array(best_val_trues), preds)
      if f1 > best_f1:
          best_f1 = f1
          best_th = th

  print(f"🔍 Best threshold: {best_th:.2f}, F1: {best_f1:.4f}")

  # ✅ 테스트셋 예측 및 제출 파일 생성
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
  X_test_tensor = torch.tensor(X_test_pad, dtype=torch.float32).to(device)

  model.eval()
  y_test_probs = []
  with torch.no_grad():
      for i in range(0, len(X_test_tensor), 64):
          batch = X_test_tensor[i:i+64]
          preds = model(batch)
          y_test_probs.extend(preds.cpu().numpy())

  final_test_preds = (np.array(y_test_probs) > best_th).astype(int)
  final_labels = ['I' if p == 1 else 'E' for p in final_test_preds]



  submission = pd.DataFrame({"ID": valid_ids, "Target": final_labels})
  submission.to_csv("submission.csv", index=False)
  print("✅ submission.csv 저장 완료")
##############################################################################

