import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.core.window import Window
import matplotlib
matplotlib.use('Agg')  # Agar Matplotlib kompatibel dengan Kivy
import matplotlib.pyplot as plt
import re
import sqlite3

# Hubungkan ke database SQL
conn = sqlite3.connect("diabetes.db")
table_name = "diabetes_data"

# Validasi karakter input hanya angka dan titik
class NumericInput(TextInput):
    def insert_text(self, substring, from_undo=False):
        s = re.sub(r'[^0-9.]', '', substring)
        return super().insert_text(s, from_undo=from_undo)

# Load dataset Lama
# data = pd.read_csv("diabetes.csv")
# df = data.drop("Pregnancies", axis=1)

# Load Database
df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
conn.close()

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.drop('Outcome', axis=1))
X = pd.DataFrame(scaled_data, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = GaussianNB()
model.fit(X_train, Y_train)

class DiabetesApp(App):
    def build(self):
        self.title = "Prediksi Diabetes"
        self.layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.inputs = []

        fields = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        for field in fields:
            label = Label(text=f"{field}:", size_hint=(1, 0.1))
            text_input = NumericInput(hint_text=f"Masukkan nilai {field}", multiline=False, size_hint=(1, 0.1))
            self.layout.add_widget(label)
            self.layout.add_widget(text_input)
            self.inputs.append(text_input)

        self.predict_button = Button(text="Prediksi", size_hint=(1, 0.2))
        self.predict_button.bind(on_press=self.predict_diabetes)
        self.layout.add_widget(self.predict_button)

        return self.layout

    def predict_diabetes(self, instance):
        try:
            user_input = [[float(text_input.text) for text_input in self.inputs]]
            user_input_scaled = scaler.transform(user_input)
            probability = model.predict_proba(user_input_scaled)[0][1]

            result = self.get_prediction_result(probability)
            self.generate_pie_chart(probability)
            self.show_popup(f"Hasil Prediksi: {result}\nRating Probabilitas: {probability:.2f}", probability)

        except ValueError:
            self.show_popup("Silakan masukkan nilai yang valid di semua input!", None)

    def get_prediction_result(self, probability):
        if probability < 0.4:
            return "TIDAK DIABETES"
        elif 0.4 <= probability < 0.5:
            return "KEMUNGKINAN TIDAK DIABETES"
        elif 0.5 <= probability < 0.6:
            return "KEMUNGKINAN DIABETES"
        else:
            return "DIABETES"

    def generate_pie_chart(self, probability):
        labels = ['Diabetes', 'Tidak Diabetes']
        sizes = [probability, 1 - probability]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)
        
        popup_width = Window.size[0] * 0.8  # 80% dari lebar Window
        figsize = (popup_width / 100, popup_width / 100)  # Atur figsize berdasarkan skala

        plt.figure(figsize=figsize)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Pastikan pie chart lingkaran sempurna
        plt.savefig("pie_chart.png", bbox_inches='tight', dpi=200)
        plt.close()


    def show_popup(self, message, probability):
        popup_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        popup_label = Label(text=message, size_hint=(1, 0.2))
        popup_layout.add_widget(popup_label)

        if probability is not None:
            # Tampilkan Pie Chart
            img = Image(source="pie_chart.png", size_hint=(1, 0.6))  # Atur ukuran gambar
            popup_layout.add_widget(img)

        # Tombol untuk mengulang atau keluar
        button_layout = BoxLayout(size_hint=(1, 0.2), spacing=10)
        retry_button = Button(text="Ulangi", size_hint=(0.5, 1))
        exit_button = Button(text="Keluar", size_hint=(0.5, 1))

        retry_button.bind(on_press=self.restart_prediction)
        exit_button.bind(on_press=self.stop_app)

        button_layout.add_widget(retry_button)
        button_layout.add_widget(exit_button)
        popup_layout.add_widget(button_layout)

        # Ukuran popup
        popup = Popup(title="Hasil Prediksi", content=popup_layout, size_hint=(0.8, 0.8))  # Popup 80% dari layar
        popup.open()

    def restart_prediction(self, instance):
        self.stop()
        DiabetesApp().run()

    def stop_app(self, instance):
        App.get_running_app().stop()

# Jalankan aplikasi
if __name__ == "__main__":
    Window.size = (500, 700)
    DiabetesApp().run()
