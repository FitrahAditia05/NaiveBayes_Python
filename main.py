from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

from cek_akurasi import CekAkurasiScreen
from cek_diabetes import CekDiabetesScreen

class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        with self.canvas.before:
            self.rect = Rectangle(size=Window.size, source='background.png')

        Window.bind(size=self.update_rect)

        logo = Image(source='logo.png', size_hint=(1, 0.2))
        layout.add_widget(logo)

        labelapp = Label(
            text="DIABETES CHECKER",
            size_hint=(1, 0.05),
            color=(1, 0, 0, 1),  # Warna teks putih
            font_size='32sp',
            bold=True  # Teks tebal
            
        )
        layout.add_widget(labelapp)
        
        labelver = Label(
            text="Apps v.01",
            size_hint=(1, 0.05),
            color=(0.22,0.86,0.07,1),
            font_size='14sp',
            halign='center'
        )
        layout.add_widget(labelver)

        names = Label(
            text="Nama Anggota: \n Agus Saputra | SI-22566008 \n Fitrah Aditia \n Miftahudin Rifki | SI-19563003 \n Steven Ario Timotheus | SI-22563003",
            size_hint=(1, 0.15),
            color=(1, 1, 1, 1),
            font_size='15sp',
            halign='center'
        )
        layout.add_widget(names)

        btn_layout = BoxLayout(size_hint=(1, 0.1), padding=20, spacing=20)
        btn_layout.add_widget(BoxLayout(size_hint=(0.1, 1)))  # Spacer di kiri
        start_btn = Button(
            text="Mulai",
            size_hint=(0.8, 0.6),
            background_color=(0, 1, 0, 1),  # Hijau Warna FF7F00 dalam RGBA
            color=(1, 1, 1, 1),  # Warna teks putih
            bold=True  # Teks tebal
        )
        start_btn.bind(on_press=self.goto_dashboard)
        btn_layout.add_widget(start_btn)
        btn_layout.add_widget(BoxLayout(size_hint=(0.1, 1)))  # Spacer di kanan
        layout.add_widget(btn_layout)

        self.add_widget(layout)

    def update_rect(self, *args):
        self.rect.size = Window.size

    def goto_dashboard(self, instance):
        self.manager.current = 'dashboard'

class DashboardScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        with layout.canvas.before:
            Color(rgba=(0.99, 0.98, 0.96, 1))
            self.rect = Rectangle(size=Window.size)

        Window.bind(size=self.update_rect)

        logo = Image(source='logo.png', size_hint=(1, 0.3))
        layout.add_widget(logo)

        buttons = [
            ("Cek Akurasi", self.goto_cek_akurasi, (0, 0, 1, 1)),  # Biru
            ("Cek Diabetes", self.goto_cek_diabetes, (1, 0.5, 0, 1)),  # Oranye
            ("Kembali", self.goto_home, (0, 1, 0, 1)),  # Hijau
            ("Keluar", self.exit_app, (1, 0, 0, 1))  # Merah
        ]

        for text, callback, color in buttons:
            btn = Button(text=text, size_hint=(1, 0.2), background_color=color)
            btn.bind(on_press=callback)
            layout.add_widget(btn)

        self.add_widget(layout)

    def update_rect(self, *args):
        self.rect.size = Window.size

    def goto_cek_akurasi(self, instance):
        self.manager.current = 'cek_akurasi'

    def goto_cek_diabetes(self, instance):
        self.manager.current = 'cek_diabetes'

    def goto_home(self, instance):
        self.manager.current = 'home'

    def exit_app(self, instance):
        App.get_running_app().stop()

class MainApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(DashboardScreen(name='dashboard'))
        sm.add_widget(CekAkurasiScreen(name='cek_akurasi'))
        sm.add_widget(CekDiabetesScreen(name='cek_diabetes'))
        return sm

if __name__ == "__main__":
    Window.size = (500, 650)
    MainApp().run()
