import pandas as pd
import os
import random

# โหลดไฟล์ CSV
df = pd.read_csv("cv-valid-train.csv")


df = df.drop(['up_votes','down_votes','duration'], axis=1)
print(df.columns)

df = df.dropna()

# กำหนดจำนวนสูงสุดที่อยากให้มีต่อ accent (เช่น 4000)
max_per_accent = 4000

# สุ่มลดจำนวน accent "us" และ "england" ให้เหลือ max_per_accent
df_us = df[df['accent'] == 'us'].sort_values(by='filename').head(max_per_accent)
df_eng = df[df['accent'] == 'england'].sort_values(by='filename').head(max_per_accent)

# เก็บ accents อื่น ๆ ที่ไม่ใช่ us และ england
df_others = df[~df['accent'].isin(['us', 'england'])]

# รวมทั้งหมดเข้าด้วยกัน
df_balanced = pd.concat([df_us, df_eng, df_others])

# ---- ลบไฟล์เสียงที่ไม่ได้อยู่ใน df_balanced ----
# เอาชื่อ path ที่ยังคงอยู่
kept_files = set(df_balanced['filename'])


for path in df['filename']:
    if path not in kept_files:
        try:
            os.remove(path)  # ใช้ path ตรง ๆ ได้เลย
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error deleting {path}: {e}")

accent_counts = df_balanced["accent"].value_counts()

# แสดงผล
print(accent_counts)
print(df_balanced.shape)

import os

# ดึงเฉพาะชื่อไฟล์จาก df_balanced['filename'] โดยตัด path ออก (เผื่อมี path อยู่)
kept_files = set([os.path.basename(f) for f in df_balanced['filename']])

# รายชื่อไฟล์ในโฟลเดอร์ (ไม่มี path อยู่แล้ว)
files_in_folder = [f for f in os.listdir("cv-valid-train") if f.endswith(".mp3")]

# หาไฟล์ที่อยู่ในโฟลเดอร์แต่ไม่อยู่ใน df_balanced
missing_files = [f for f in files_in_folder if f not in kept_files]

# แสดงผล
if missing_files:
    print(f"พบไฟล์ที่ไม่ได้อยู่ใน df_balanced จำนวน {len(missing_files)} ไฟล์:")
    for f in missing_files:
        print(f"- {f}")
else:
    print("✅ ทุกไฟล์ในโฟลเดอร์อยู่ใน df_balanced แล้ว!")


