from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.core.window import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")
df = data.drop('Pregnancies', axis=1)

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.drop('Outcome', axis=1))
X = pd.DataFrame(scaled_data, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
Y = df['Outcome']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = GaussianNB()
model.fit(X_train, Y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)

class CekAkurasiScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # Button to show accuracy
        accuracy_button = Button(text="Tampilkan Akurasi", size_hint=(1, 0.2))
        accuracy_button.bind(on_press=self.show_accuracy)
        layout.add_widget(accuracy_button)

        # Back button
        back_button = Button(text="Kembali", size_hint=(1, 0.2))
        back_button.bind(on_press=self.goto_HomeScreen)
        layout.add_widget(back_button)

        self.add_widget(layout)

    def show_accuracy(self, instance):
        self.generate_pie_chart(accuracy)
        self.show_popup(f"Akurasi Model: {accuracy * 100:.2f}%", accuracy)

    def generate_pie_chart(self, accuracy):
        labels = ['Akurasi', 'Error']
        sizes = [accuracy, 1 - accuracy]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.1, 0)

        popup_width = Window.size[0] * 0.8
        figsize = (popup_width / 100, popup_width / 100)

        plt.figure(figsize=figsize)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.savefig("accuracy_pie_chart.png", bbox_inches='tight', dpi=200)
        plt.close()

    def show_popup(self, message, accuracy):
        popup_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        popup_label = Label(text=message, size_hint=(1, 0.2))
        popup_layout.add_widget(popup_label)

        # Add pie chart
        img = Image(source="accuracy_pie_chart.png", size_hint=(1, 0.6))
        popup_layout.add_widget(img)

        # Add buttons
        button_layout = BoxLayout(size_hint=(1, 0.2), spacing=10)
        close_button = Button(text="Tutup", size_hint=(1, 1))
        close_button.bind(on_press=self.close_popup)

        button_layout.add_widget(close_button)
        popup_layout.add_widget(button_layout)

        self.popup = Popup(title="Hasil Akurasi", content=popup_layout, size_hint=(0.8, 0.8))
        self.popup.open()

    def close_popup(self, instance):
        self.popup.dismiss()

    def goto_HomeScreen(self, instance):
        self.manager.current = 'home'
