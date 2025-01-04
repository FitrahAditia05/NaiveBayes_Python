import pandas as pd
import sqlite3

# File CSV sumber data
csv_file = "diabetes.csv"  # Pastikan file diabetes.csv ada di folder yang sama
db_name = "diabetes.db"    # Nama file database SQLite

# Membaca file CSV
try:
    df = pd.read_csv(csv_file)
    print("File diabetes.csv berhasil dibaca.")
except FileNotFoundError:
    print(f"File {csv_file} tidak ditemukan! Pastikan file tersebut ada di direktori yang sama.")
    exit()

# Hapus kolom 'Pregnancies' (jika diperlukan)
if 'Pregnancies' in df.columns:
    df = df.drop("Pregnancies", axis=1)

# Buat database SQLite dan simpan data ke dalam tabel
try:
    conn = sqlite3.connect(db_name)  # Membuat database SQLite
    table_name = "diabetes_data"
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Database '{db_name}' berhasil dibuat dan tabel '{table_name}' ditambahkan.")
except Exception as e:
    print("Error saat membuat database:", e)
finally:
    conn.close()
    print("Koneksi database ditutup.")
