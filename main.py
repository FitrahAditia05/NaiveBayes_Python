from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.core.window import Window

# IMPORT MODUL LAIN (HALAMAN LAIN)
from cek_akurasi import CekAkurasiScreen
from convert_db import ConvertDatabaseScreen
from cek_diabetes import CekDiabetesScreen

# HALAMAN AWAL (DASHBOARD)
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        cek_akurasi_btn = Button(text="Cek Akurasi", size_hint=(1, 0.2))
        convert_db_btn = Button(text="Convert Database", size_hint=(1, 0.2))
        cek_diabetes_btn = Button(text="Cek Diabetes", size_hint=(1, 0.2))
        exit_btn = Button(text="Exit", size_hint=(1, 0.2))

        # Bind tombol ke metode transisi layar
        cek_akurasi_btn.bind(on_press=self.goto_cek_akurasi)
        convert_db_btn.bind(on_press=self.goto_convert_database)
        cek_diabetes_btn.bind(on_press=self.goto_cek_diabetes)
        exit_btn.bind(on_press=self.exit_app)

        layout.add_widget(cek_akurasi_btn)
        layout.add_widget(convert_db_btn)
        layout.add_widget(cek_diabetes_btn)
        layout.add_widget(exit_btn)

        self.add_widget(layout)
        
    def goto_cek_akurasi(self, instance):
        self.manager.current = 'cek_akurasi'
            
    def goto_convert_database(self, instance):
        self.manager.current = 'convert_db'
            
    def goto_cek_diabetes(self, instance):
        self.manager.current = 'cek_diabetes'
        
    def exit_app(self, instance):
        App.get_running_app().stop()

# APLIKASI UTAMA
class MainApp(App):
    def build(self):
        sm = ScreenManager()

        # TAMBAKAN KE SCREEN HOME
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(CekAkurasiScreen(name='cek_akurasi'))
        sm.add_widget(ConvertDatabaseScreen(name='convert_db'))
        sm.add_widget(CekDiabetesScreen(name='cek_diabetes'))
        sm.add_widget(CekDiabetesScreen(name='keluar'))

        return sm

# MENJALANKAN APLIKASI
if __name__ == "__main__":
    Window.size = (500, 700)
    MainApp().run()
