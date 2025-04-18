import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

df = pd.read_csv('cv-valid-train.csv')
df = df.drop(['up_votes','down_votes','duration'], axis=1)
print(df)

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