from kivy.app import App
from kivy.uix.gridlayout import GridLayout

class MyWidget (GridLayout):
    def selected(self, filename):
        try:
           self.ids.image.source = filename[0]
        except:
            pass

class FileChooserWindow(App):
    def build(self):
        return MyWidget()


if __name__ == '__main__':
    window = FileChooserWindow()
    window.run()