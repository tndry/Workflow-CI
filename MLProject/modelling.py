# Isi file: Workflow-CI/MLProject/modelling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle  # Kita pakai pickle untuk menyimpan model
import warnings

warnings.filterwarnings('ignore')

print("Memuat data...")
df = pd.read_csv('namadataset_preprocessing/heart_attack_preprocessed.csv')

print("Mempersiapkan data...")
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Melatih model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy:.4f}")

print("Menyimpan model ke model.pkl...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan!")