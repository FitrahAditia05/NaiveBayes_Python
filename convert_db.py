from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import pandas as pd
import sqlite3

class ConvertDatabaseScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # LAYOUT UTAMA
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # TOMBOL KONVERSI
        convert_button = Button(text="Mulai Konversi Database", size_hint=(1, 0.2))
        convert_button.bind(on_press=self.convert_database)

        # TOMBOL KEMBALI KE HOME SCREEN
        back_button = Button(text="Kembali", size_hint=(1, 0.2))
        back_button.bind(on_press=self.goto_home_screen)

        # MENAMBAHKAN TOMBOL KE LAYOUT
        self.layout.add_widget(convert_button)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def convert_database(self, instance):
        csv_file = "diabetes.csv"  # FILE DATABASE.CSV HARUS TERSEDIA
        db_name = "diabetes.db"    # NAMA FILE SETELAH DI CONVERT

        try:
            # MEMBACA FILE CSV
            df = pd.read_csv(csv_file)
            print("File diabetes.csv berhasil dibaca.")

            # MENGHAPUS KOLOM PREGNANCY KARENA KARENA TIDAK DIPERLUKAN
            if 'Pregnancies' in df.columns:
                df = df.drop("Pregnancies", axis=1)

            # BUAT DATABASE SQL LITE DAN SIMPAN TABEL
            conn = sqlite3.connect(db_name)  
            table_name = "diabetes_data"
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"Database '{db_name}' berhasil dibuat dan tabel '{table_name}' ditambahkan.")
            conn.close()
            
            # POPUP JIKA BERHASIL
            self.show_popup("Konversi Berhasil", "Database berhasil dikonversi ke SQLite.")

        except FileNotFoundError:
            print(f"File {csv_file} tidak ditemukan! Pastikan file tersebut ada di direktori yang sama.")
            self.show_popup("Konversi Gagal", f"File {csv_file} tidak ditemukan.")
        except Exception as e:
            print("Error saat membuat database:", e)
            self.show_popup("Konversi Gagal", "Terjadi kesalahan saat mengonversi database.")
            
    def show_popup(self, title, message):
        # PENGATURAN LAYOUT POP UP
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        label = Label(text=message, size_hint=(1, 0.7))
        close_button = Button(text="Tutup", size_hint=(1, 0.3))
        
        popup_layout.add_widget(label)
        popup_layout.add_widget(close_button)

        # MEMBUAT POP UP
        popup = Popup(title=title, content=popup_layout, size_hint=(0.8, 0.4))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

    def goto_home_screen(self, instance):
        self.manager.current = 'home'
