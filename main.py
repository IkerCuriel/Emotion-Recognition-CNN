from CCN_Google_Model import EmotionModel
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2

class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('')
        self.geometry('500x250+500+220')

        self.title_label = ctk.CTkLabel(self, text='ᴍᴏᴏᴅᴍᴀᴘᴘᴇʀ', font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=(20, 20))

        self.subtitle_label = ctk.CTkLabel(self, text='ʀᴇᴀᴅɪɴɢ ʙᴇᴛᴡᴇᴇɴ ᴘɪxᴇʟꜱ, ꜰᴇᴇʟɪɴɢ ʙᴇᴛᴡᴇᴇɴ ʟɪɴᴇꜱ.')
        self.subtitle_label.pack(padx=20, pady=0)

        self.input_frame = ctk.CTkFrame(self, width=50, height=50, corner_radius=15)
        self.input_frame.place(relx=0.5, rely=0.7, anchor=ctk.CENTER)
        
        self.frame_2 = ctk.CTkFrame(self, width=180, height=50, corner_radius=15)
        self.frame_2.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)
        
        self.image_path = None  # Variable para almacenar la ruta de la imagen
        self.modelo = EmotionModel()

        # Cargar el archivo de la imagen
        self.load_song_button = ctk.CTkButton(self.frame_2, text="ᴜᴘʟᴏᴀᴅ ɪᴍᴀɢᴇ", command=self.load_archive)
        self.load_song_button.pack(padx=20, pady=5)
        
        self.button_1 = ctk.CTkButton(self.input_frame, text="ᴀɴᴀʟʏᴢᴇ", command=self.open_toplevel)
        self.button_1.pack(padx=20, pady=5)
        self.toplevel_window = None
        
    def categorizar(self, ruta):
        img = Image.open(ruta)
        img = img.convert('RGB')
        img = np.array(img).astype(float)/255
        img = cv2.resize(img, (224, 224))
        prediction = self.modelo.model.predict(img.reshape(-1, 224, 224, 3))
        return np.argmax(prediction[0], axis=-1) 
        
    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ToplevelWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it
     
    def load_img(self, file_path):
        # Aquí puedes implementar la lógica para analizar la canción ctk
        print("Imagen cargada:", file_path)
        prediction = self.categorizar(file_path)
        print("Predicción:", prediction)

    def load_archive(self):
        file_path = filedialog.askopenfilename(title='Seleccionar la imagen', filetypes=[('Imagenes', '*.jpg *.png')])
        if file_path:
            self.load_img(file_path)

    def create_login(self):
        pass

class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("800x650+300+100")
        self.title('ᴀɴᴀʟʏꜱɪꜱ')
        
        self.segemented_frame = ctk.CTkFrame(self, width=700, height=600, corner_radius=15)
        self.segemented_frame.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        self.segemented_button = ctk.CTkSegmentedButton(self, values=["ᴘʀᴇᴅɪᴄᴛɪᴏɴꜱ", "ᴄᴏʀʀᴇᴄᴛɪᴏɴꜱ", "ɪᴍᴘʀᴏᴠᴇᴍᴇɴᴛꜱ"], 
                                                        command=self.segmented_button_callback)
        self.segemented_button.set("ᴘʀᴇᴅɪᴄᴛɪᴏɴꜱ")
        self.segemented_button.pack()
    
    def segmented_button_callback(self, value):
        print("segmented button clicked:", value)
        
app = App()
app.mainloop() 