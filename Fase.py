import numpy as np
from scipy.io import wavfile
from tkinter import *
import tkinter as tk

class Proyecto(Frame):
    # Constructor
    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args, **kwargs)
        self.parent = master
        self.grid()
        self.interfaz()
    
    # Interfaz del programa
    def interfaz(self):        
        # Columna 1
        # --------------------------------------------------------------------------------------------------------------------------------------------
        # Mensaje de archivo de audio
        self.display = Label(self, font=("Arial", 13), borderwidth=1, justify="center", text="Escribe el archivo de audio", bg="#FEFFCA", width=55, height=1)
        self.display.grid(row=2, column=0, columnspan=1, sticky="nsew")

        # Input de archivo de audio
        audio_input = Entry(self, font=("Arial", 13), borderwidth=1, justify="center", bg="#CAE0FF", width=55)
        self.display = audio_input
        self.display.grid(row=3, column=0, columnspan=1, sticky="nsew")

        # Mensaje de mensaje a ocultar
        self.display = Label(self, font=("Arial", 13), borderwidth=1, justify="center", text="Escribe el mensaje a ocultar", bg="#FEFFCA", width=55, height=1)
        self.display.grid(row=4, column=0, columnspan=1, sticky="nsew")

        # Input de mensaje a ocultar
        mensaje = Text(self, font=("Arial", 13), borderwidth=1, bg="#CAE0FF", width=55, height=15)
        self.display = mensaje
        self.display.grid(row=5, column=0, columnspan=1, sticky="nsew")
        
        # Botón 'Ocultar'
        self.ceButton = Button(self, font=("Arial", 13), fg='black', text="Ocultar", command = lambda: self.encode(audio_input, mensaje))
        self.ceButton.grid(row=6, column=0, sticky="nsew")

        # Columna 2
        # --------------------------------------------------------------------------------------------------------------------------------------------
        # Mensaje de archivo de audio
        self.display = Label(self, font=("Arial", 13), borderwidth=0, justify="center", text="Escribe el archivo con el mensaje oculto", bg="#F5DFFF", width=55, height=1)
        self.display.grid(row=7, column=0, columnspan=1, sticky="nsew")

        # Input de archivo de audio
        hidden_input = Entry(self, font=("Arial", 13), borderwidth=0, justify="center", bg="#D8FFE0", width=55)
        self.display = hidden_input
        self.display.grid(row=8, column=0, columnspan=1, sticky="nsew")

        # Mensaje de mensaje a recuperar
        self.display = Label(self, font=("Arial", 13), borderwidth=0, justify="center", text="Mensaje recuperado", bg="#F5DFFF", width=55, height=1)
        self.display.grid(row=9, column=0, columnspan=1, sticky="nsew")

        # Recuadro de mensaje recuperado
        mensaje_recuperado = Text(self, font=("Arial", 13), borderwidth=0, bg="#D8FFE0", width=55, height=15)
        self.display = mensaje_recuperado
        self.display.grid(row=10, column=0, columnspan=1, sticky="nsew")

        # Botón 'Recuperar'
        self.inverseButton = Button(self, font=("Arial", 13), fg='black', text="Recuperar", command = lambda: self.decode(mensaje_recuperado, hidden_input))
        self.inverseButton.grid(row=11, column=0, sticky="nsew")
    
    # Ocultar mensaje
    def encode(self, audio_input, mensaje):

        # Obtener el nombre del archivo de audio y el mensaje
        audio_input = audio_input.get()
        mensaje = mensaje.get("1.0",'end-1c')
    
        # Lee el archivo de audio
        # Saca la frecuencia del audio
        # Guarda los valores del audio en un array
        frecuencia, audio = wavfile.read(audio_input)

        # Especifica el tamaño máximo del mensaje
        mensaje = mensaje.ljust(2000, '~')

        # Calcula el tamaño del mensaje en bits
        mensaje_bits = 8 * len(mensaje)

        # Calcula el tamaño de un chunk
        tamano_chunk = int(2 * 2 ** np.ceil(np.log2(2 * mensaje_bits)))

        # Calcula el número de chunks a generar
        numero_chunk = int(np.ceil(audio.shape[0] / tamano_chunk))

        # Copia el array del audio 
        audio_nuevo = audio.copy()

        # Si el audio es Mono, entonces tiene 1 canal de audio
        # Si el audio es Stereo, entonces tiene 2 canales de audio
        if len(audio.shape) == 1:
            # Cambia la forma del array para que sea de longitud (Número de chunks X Tamaño de un chunk)
            audio_nuevo.resize(numero_chunk * tamano_chunk, refcheck = False)

            # Añade una nueva dimensión de tamaño 1 al array para que sea 2D
            audio_nuevo = audio_nuevo[np.newaxis]

        else:
            # Cambia la forma del array para que sea una matriz de (Número de chunks X Tamaño de un chunk) filas y 2 columnas
            audio_nuevo.resize((numero_chunk * tamano_chunk, audio_nuevo.shape[1]), refcheck = False)

            # Aplica la transpuesta a la matriz
            audio_nuevo = audio_nuevo.T

        # Descompone el audio en el número de chunks con el tamaño especificado
        chunks = audio_nuevo[0].reshape((numero_chunk, tamano_chunk))

        # Calcula el DFT (Transformada Discreta de Fourier) de los chunks
        chunks = np.fft.fft(chunks)

        # Saca la magnitud de los chunks
        magnitud_chunk = np.abs(chunks)

        # Saca la fase de los chunks
        fase_chunk = np.angle(chunks)

        # Calcula la diferencia entre fases
        fase_diferencial = np.diff(fase_chunk, axis=0)

        # Convierte el mensaje a binario
        bits = np.ravel([[int(y) for y in format(ord(x), "08b")] for x in mensaje])

        # Copia el mensaje para realizar el recálculo de fase
        bits_fase = bits.copy()

        # Determina si se va a sumar o restar pi/2 a la antigua fase
        bits_fase[bits_fase == 0] = -1

        # Suma o resta de pi/2 a la antigua fase para calcula la nueva fase
        bits_fase = bits_fase * -np.pi / 2

        # Calcula el punto medio del chunk
        mitad_chunk = tamano_chunk // 2

        # Aplica la nueva fase con los bits del mensaje
        fase_chunk[0, mitad_chunk - mensaje_bits: mitad_chunk] = bits_fase
        fase_chunk[0, mitad_chunk + 1: mitad_chunk + 1 + mensaje_bits] = -bits_fase[::-1]

        # Saca la matriz de fases
        for i in range(1, len(fase_chunk)):
            fase_chunk[i] = fase_chunk[i - 1] + fase_diferencial[i - 1]
            
        # Recalcula los chunks con las fases nuevas
        chunks = (magnitud_chunk * np.exp(1j * fase_chunk))

        # Calcula la DFT inversa de los chunks
        chunks = np.fft.ifft(chunks).real

        # Junta todos los chunks transformados
        audio_nuevo[0] = chunks.ravel().astype(np.int16)    

        # Escribe el nuevo archivo de audio con el mensaje oculto
        wavfile.write("output.wav", frecuencia, audio_nuevo.T)

        # Regresa el nombre del nuevo archivo de audio
        return "output.wav" 

    # ---------------------------------------------------------------------------------------------------------

    # Recupera el mensaje oculto
    def decode(self, mensaje_recuperado, audio_input):

        # Obtener el nombre del archivo de audio
        audio_input = audio_input.get()

        # Lee el archivo de audio
        # Saca la frecuencia del audio
        # Guarda los valores del audio en un array
        frecuencia, audio = wavfile.read(audio_input)
        
        # Tamaño del mensaje en bits
        mensaje_bits = 16000

        # Calcula el tamaño de un chunk
        tamano_chunk = 2 * int(2 ** np.ceil(np.log2(2 * mensaje_bits)))

        # Calcula el punto medio del chunk
        mitad_chunk = tamano_chunk // 2

        # Si el audio es Mono, entonces tiene 1 canal de audio
        # Si el audio es Stereo, entonces tiene 2 canales de audio
        if len(audio.shape) == 1:
            # Extrae el header directamente
            header = audio[:tamano_chunk]
        else:
            # Extrae el header bajo 1 dimensión
            header = audio[:tamano_chunk, 0]

        # Saca las fases con el mensaje oculto
        # Saca el ángulo y DFT del header en los bits del mensaje oculto
        mensaje_fases = np.angle(np.fft.fft(header))[mitad_chunk - mensaje_bits:mitad_chunk]

        # Convierte las fases a binario
        mensaje_binario = (mensaje_fases < 0).astype(np.int16)

        # Convierte el mensaje en binario a entero
        mensaje_int = mensaje_binario.reshape((-1, 8)).dot(1 << np.arange(8 - 1, -1, -1))

        # # Convierte el mensaje en entero a caracteres
        # Junta los caracteres para reconstruir el mensaje
        # Elimina los ~ añadidos durante el ocultamiento
        mensaje_final = "".join(np.char.mod("%c", mensaje_int)).replace("~", "") 
        
        # Limpia el recuadro de mensaje de la interfaz
        mensaje_recuperado.delete(1.0, "end-1c")

        # Inserta el mensaje en la interfaz
        mensaje_recuperado.insert("end-1c", mensaje_final)
        return "".join(np.char.mod("%c", mensaje_int)).replace("~", "") 

# Crear interfaz
# Configurar interfaz
ui = Tk()
ui.title("Esteganografia de audio")
ui.geometry("500x775")
ui.config(cursor="cross", relief="groove")
ui.resizable(False, False)

# Llamar interfaz de programa
root = Proyecto(ui).grid()
ui.mainloop()