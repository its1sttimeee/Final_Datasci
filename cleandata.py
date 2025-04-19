import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

df = pd.read_csv('cv-valid-train.csv')
df = df.drop(['up_votes','down_votes','duration'], axis=1)

null_rows = df[df.isnull().any(axis=1)]

print(f"พบแถวที่มี null: {len(null_rows)} แถว")

# ตรวจสอบชื่อคอลัมน์ก่อนใช้
print(df.columns)

audio_files_to_delete = null_rows['filename'].tolist()  #ชื่อคอลัมน์คือ 'filename'

# ลบไฟล์เสียงที่เกี่ยวข้อง
audio_folder = 'cv-valid-train'  # path ไปยังโฟลเดอร์ที่เก็บไฟล์เสียง


for filename in audio_files_to_delete:
    file_path = filename  # ใช้ path ตรงจาก CSV
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"ลบไฟล์: {filename}")
    # else:
    #     print(f"ไม่พบไฟล์: {filename}")

# ลบแถว null จาก DataFrame
df = df.dropna()

#ตรวจสอบจำนวนRowหลังลบแถว null
print(df.shape)

df_thai = pd.read_csv('ThaiSound.csv')

def age_to_group(age):
    if pd.isnull(age):
        return None
    elif 10 <= age < 20:
        return 'teens'
    elif 20 <= age < 30:
        return 'twenties'
    elif 30 <= age < 40:
        return 'thirties'
    elif 40 <= age < 50:
        return 'fourties'
    elif 50 <= age < 60:
        return 'fifties'
    elif 60 <= age < 70:
        return 'sixties'
    elif 70 <= age < 80:
        return 'seventies'
    elif 80 <= age < 90:
        return 'eighties'
    else:
        return 'other'

# ใช้ฟังก์ชันกับคอลัมน์ age
df_thai['age'] = df_thai['age'].apply(age_to_group)

print(df_thai)

df_combined = pd.concat([df, df_thai], ignore_index=True)
print(df_combined.shape)

import pandas as pd
import librosa
import numpy as np
import os

base_path = 'cv-valid-train'

features_audio = []
features_text = []
features_age = []
features_gender = []
labels = []

for idx, row in df_combined.iterrows():
    file_path = row['filename']

    try:
        y, sr = librosa.load(file_path, sr=16000)
        y = y[:3*sr] if len(y) > 3*sr else np.pad(y, (0, 3*sr - len(y)))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        mfcc_flat = mfcc.flatten()
        features_audio.append(mfcc_flat)

        features_text.append(row['text'])
        features_age.append(row['age'])
        features_gender.append(row['gender'])
        labels.append(row['accent'])

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ข้อความเป็น TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(features_text).toarray()

# แปลงอายุและเพศเป็นตัวเลข
le_age = LabelEncoder()
X_age = le_age.fit_transform(features_age).reshape(-1, 1)

le_gender = LabelEncoder()
X_gender = le_gender.fit_transform(features_gender).reshape(-1, 1)

# แปลง label สำเนียง
le_label = LabelEncoder()
y = le_label.fit_transform(labels)

from sklearn.preprocessing import StandardScaler

X_audio = np.array(features_audio)
scaler = StandardScaler()
X_audio_scaled = scaler.fit_transform(X_audio)

# รวม: เสียง + อายุ + เพศ + ข้อความ
X_full = np.hstack([X_audio_scaled, X_age, X_gender, X_text])
print("Shape of input:", X_full.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le_label.classes_))