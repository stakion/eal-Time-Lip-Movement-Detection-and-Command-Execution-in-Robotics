#######################################################################################################################################################################################
## Instalación de las librerias correspondientes: pip install sounddevice whisper
### Importación de las librerias correspondientes a utilizar.
#######################################################################################################################################################################################
# Parte de librerias a importar referentes a librerias de uso miscelanio.
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sounddevice as sd
import scipy.io.wavfile
import pandas as pd
import numpy as np
import subprocess
import platform
import whisper
import psutil
import glob
import time
import cv2
import os

# Parte de ejecucion asincrona:
import concurrent.futures

# Parte de librerias a importar referentes para la deteccion de movimientos de partes del rostro.
import mediapipe as mp

# Parte de librerias a importar referentes a LLM y apis para su uso.
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
# import langchain

# Parte de librerias a importar de ros2. 
import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Inicialización de las variables globales.
#######################################################################################################################################################################################
sg_aux_999_FF = "%Y-%m-%d %H:%M:%S%f%Z"
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
sg_aux_998_NM = "Hyperx"                                                # Considerar la marca o modelo del microfono.
rclpy.init(args=None)                                                # Iniciar comunicacion con ROS2.
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Validar ubicacion de forma manual.
# Verificacion y cambio de directorio de ser posible ya que presenta algunos bugs en ubuntu la ubicaicion.
#######################################################################################################################################################################################
directorio_original = os.getcwd()                               ## Guarda el directorio original
print("Directorio original:", directorio_original)

# Si se presenta bug de workspace.
# os.chdir('..')                                                  ## Cambia a un nuevo directorio (ejemplo: directorio padre)
print("Directorio después de cambiar:", os.getcwd() )

# Cambio de directorio, a veces ocurre dependiendo del entorno y SO.
#os.chdir("./Desktop/MAESTRIA/2_SEMESTRE/Introducción a la Robotica/Proyecto_Final/SEMANA_02")
#os.chdir("./Documents/CARPETAS/MAESTRIA/2_SEMESTRE/Introducción a la Robotica/Proyecto_Final/SEMANA_07")
os.chdir("./Desktop/MAESTRIA/2_SEMESTRE/Introducción a la Robotica/Proyecto_Final/SEMANA_07")
#os.chdir("./Proyecto_Final/SEMANA_07")
os.getcwd()             ## Verificacion del directorio actual.
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Creación de la función para preguntar si existe o no la carpeta correspondiente.
## Devuelve la respuesta a la pregunta si existe la carpeta que se considera o no.
#######################################################################################################################################################################################
def Validate_Folder( sl_aux_000 ):
    try:	
        os.stat( sl_aux_000 )
        return True
    except:	
        os.mkdir( sl_aux_000 )
        return False
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Definición de los nombres para las carpetas a considerar.
## Preguntar si existen cada una de las carpetas de control con la función previamente definida.
### Definicion de las cadenas de texto globales a utilizar considerando la ubicación correspondiente a cada uno de los archivos.
#######################################################################################################################################################################################
sg_aux_001_AR = "AUDIO_RECORDED";               Validate_Folder( sg_aux_001_AR );           sg_aux_004_FNAR = './' + str( sg_aux_001_AR ) + '/' + str( "Recording.wav" )
sg_aux_002_AT = "AUDIO_TRANSCRIPTED";           Validate_Folder( sg_aux_002_AT );           sg_aux_005_FNAT = './' + str( sg_aux_002_AT ) + '/' + str( "Transcription.csv" )
sg_aux_003_DD = "DETECTED_DEVICES";             Validate_Folder( sg_aux_003_DD );           sg_aux_006_FNDD = './' + str( sg_aux_003_DD ) + '/' + str( "Detected_Devices.csv" )
sg_aux_007_VR = "VIDEO_RECORDED";               Validate_Folder( sg_aux_007_VR );           sg_aux_008_FNVR = './' + str( sg_aux_007_VR ) + '/' + str( "Recorded_Video.mp4" )
sg_aux_009_LD = "LIPS_DETECTION";               Validate_Folder( sg_aux_009_LD );           sg_aux_010_FNLP = './' + str( sg_aux_009_LD ) + '/' + str( "Detected_Motion.csv" )
sg_aux_013_LT = "LIPS_TAGS";                    Validate_Folder( sg_aux_013_LT );           sg_aux_014_FLT = './' + str( sg_aux_013_LT ) + '/' + str( "ETIQUETAS_LABIOS_03.csv" )
sg_aux_015_F = "LANGCHAIN_FRAMES";              Validate_Folder( sg_aux_015_F );            

# Llama 3 tiempos, español y en ingles.
sg_aux_016_FF_llamac_en = './' + str( sg_aux_015_F ) + '/' + str( "LLAMA_EN.png" )
sg_aux_016_FF_llamac_es = './' + str( sg_aux_015_F ) + '/' + str( "LLAMA_ES.png" )
sg_aux_011_FNLP = './' + str( sg_aux_009_LD ) + '/' + str( "Last_Frame.png" )
####################################################################################################################################################################################### 


#######################################################################################################################################################################################
# Partes considerando Whisper, Langchain, LLama3.
#######################################################################################################################################################################################
# Instancia de ollama referente al modelo llama3 (demora de 5 segundos a 10 segundos de respuesta).
llm_3 = Ollama( model="llama3", 
                # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature = 0.9
            )

# Prompt con instrucciones especificas para realizar el etiquetado. (Etiquetado general).
prompt_1 = PromptTemplate(
    input_variables=["topic"],
    template="Considering the following text: {topic}, tell me which of the following classifications fits the best, only tell me the classification once and nothing else: ( Recuperar Posicion , Recuperar Velocidad , Recuperar Orientacion , Recuperar Velocidad Angular , Movimiento , Marcar un punto , Aterrizar , Despegar , Detener ) "
)

# Prompt con instrucciones especificas para realizar el etiquetado. (Etiquetado general).
prompt_2 = PromptTemplate(
    input_variables=["topic"],
    template="Considering the following text: {topic}, tell me which of the following classifications fits the best, only tell me the classification of the axis of movement once and nothing else: ( X - Positivo , X - Negativo , Y - Positivo , Y - Negativo , Z - Positivo , Z - Negativo ) "
)

# Declaracion de la cadena para hacer uso del prompt de forma automatizada dentro del codigo.
chain_31 = LLMChain(llm=llm_3, prompt=prompt_1, verbose=False)
chain_32 = LLMChain(llm=llm_3, prompt=prompt_2, verbose=False)

# Definir una funcion para ver si existe el tag final dentro de la respuesta.
def Correct_Answer_31( S_aux_whisper_text ):
    S_Aux_Tags = str( chain_31.run( S_aux_whisper_text ) )
    if( "Recuperar Posicion" in S_Aux_Tags ): return "Recuperar Posicion" 
    if( "Recuperar Velocidad" in S_Aux_Tags ): return "Recuperar Velocidad" 
    if( "Recuperar Orientacion " in S_Aux_Tags ): return "Recuperar Orientacion"
    if( "Recuperar Velocidad Angular" in S_Aux_Tags ): return "Recuperar Velocidad Angular"
    if( "Marcar un punto" in S_Aux_Tags ): return "Marcar un punto"
    if( "Movimiento" in S_Aux_Tags ): return "Movimiento"
    if( "Aterrizar" in S_Aux_Tags ): return "Aterrizar"
    if( "Despegar" in S_Aux_Tags ): return "Despegar"
    if( "Detener" in S_Aux_Tags ): return "Detener"
    # Si se detecto movimiento realizar un segundo procesamiento de lenguaje.

# Definir una funcion para ver si existe el tag final dentro de la respuesta.
def Correct_Answer_32( S_aux_whisper_text ):
    S_Aux_Tags = str( chain_32.run( S_aux_whisper_text ) )
    if( "X - Positivo" in S_Aux_Tags ): return "X - Positivo" 
    if( "X - Negativo" in S_Aux_Tags ): return "X - Negativo" 
    if( "Y - Positivo" in S_Aux_Tags ): return "Y - Positivo"
    if( "Y - Negativo" in S_Aux_Tags ): return "Y - Negativo"
    if( "Z - Positivo" in S_Aux_Tags ): return "Z - Positivo"
    if( "Z - Negativo" in S_Aux_Tags ): return "Z - Negativo"

# Prueba de la funcion anterior. 
Correct_Answer_31( "I want to move the Dron in the X axis." )
Correct_Answer_32( "I want to move the Dron in the X axis negativly." )

# Banderas y funciones para detener la ejecucion de loops concurrentes:
stop_flag = False

def stop_requested():
    return stop_flag

# Clase para monitorear las variables en el loop de monitoreo.
class Z_Parameters:
    # Variables que involucran el control de las visualizaciones en camera.
    See_Text_On_Camera = True

    # Variables de activacion para el control de Whisper.
    Is_Whisper_Active = False
    Is_Whisper_On_Loop = False
    C_Is_Whisper_Active = 0
    Is_Whisper_Finalizated = False
    Whisper_Text_Transcripted = ""
    Whisper_Time = 0.0

    # Variables que involucran el control que se gestiona con Langchain y LLAMA3.
    Is_LLAMA3_On_Loop = False
    Is_LLAMA3_Active = False
    C_Is_LLAMA3_Active = 0
    The_LLAMA3_TEXT = ""
    The_LLAMA3_Text_Tag_01 = ""                             # Tag correspondiente a la clasificacion de los primeros elementos o tipos de movimiento y lecturas a considerar.
    The_LLAMA3_Text_Tag_02 = ""                             # Tag referente al tipo de movimiento lineal, esta parte se puede mejorar haciendo uso de movimiento circular, orientacion, etc.
    Is_LLAMA3_Finalizated = False
    The_LLAMA3_Time = 0.0


# Instanciacion de la clase para monitoreo de variables de forma general.
Z_00 = Z_Parameters()
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Detectar todos los dispositivos de audio conectados a la computadora:
#######################################################################################################################################################################################
print("Dispositivos disponibles:")
print( sd.query_devices() )
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Detección de los dispositivos de audio haciendo uso de dataframes para mayor orden:
#######################################################################################################################################################################################
def Sound_Devices(sl_aux_000 = sg_aux_006_FNDD):
    dispositivos_info = sd.query_devices()
    lista_dispositivos = []

    for i, dispositivo in enumerate(dispositivos_info):
        dispositivo_dict = {
            "Índice": i,
            "Nombre": dispositivo['name'],
            "Tipo": "Entrada" if dispositivo['max_input_channels'] > 0 else "Salida",
            "Canales de entrada": dispositivo['max_input_channels'],
            "Canales de salida": dispositivo['max_output_channels'],
            "Frecuencia de muestreo predeterminada": dispositivo['default_samplerate'],
            "Fecha de Consulta": str( datetime.now().strftime(sg_aux_999_FF) )
        }
        lista_dispositivos.append(dispositivo_dict)
    
    df_dispositivos = pd.DataFrame(lista_dispositivos)
    df_dispositivos.to_csv(sl_aux_000, index=False)
    print(df_dispositivos)

    # Considerar la marca o modelo del microfono.
    s_dispositivos_aux = str( df_dispositivos[ sg_aux_998_NM in df_dispositivos["Nombre"] ]["Indice"] )
    print( s_dispositivos_aux )

Sound_Devices()
#######################################################################################################################################################################################
 

#######################################################################################################################################################################################
# Detectar la voz por microfono. sg_aux_008_FNVR
## Tiempo de default 5 segundos. 
## Frecuencia de muestreo 44.1 Khz. 
## Indice_dispositivo correspondiente al dispositivo seleccionado anteriormente.
### Almacenar la grabación de audio en un archivo tipo .wav
#### 6 - HyperX Quadcast: USB Audio (hw:3,0)
#######################################################################################################################################################################################
def Record_Audio_File_Mic( duracion = 5 , frecuencia_muestreo = 44100 , indice_dispositivo = 6 , Audio_Rec_File = sg_aux_004_FNAR ):
    print("Grabando...")
    grabacion = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=2, device=indice_dispositivo, dtype='float64')

    # Espera hasta que termine la grabación
    sd.wait()                                   
    
    print("Grabación terminada")
    scipy.io.wavfile.write(Audio_Rec_File, frecuencia_muestreo, grabacion)
    print("Archivo almacenado con exito !...")
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Transcripción del audio haciendo uso de whisper. 
#######################################################################################################################################################################################
def Transcribe_Audio_From_File(Audio_Rec_File = sg_aux_004_FNAR, Transcripted_Audio_File = sg_aux_005_FNAT, Model_Version = "medium"):
    modelo = whisper.load_model(Model_Version)                          # Inicialización del modelo de whisper correspondiente.

    # Transripción al idioma deseado.
    resultado = modelo.transcribe(Audio_Rec_File);                             sl_aux_010_ID = str( resultado['language'] )
    resultado_es = modelo.transcribe(Audio_Rec_File, language="Spanish");      sl_aux_011_TE = str( resultado_es['text'] )
    resultado_en = modelo.transcribe(Audio_Rec_File, language="English");      sl_aux_012_TI = str( resultado_en['text'] ) 
    sl_aux_013_CF = str( datetime.now().strftime(sg_aux_999_FF) )       # Cadena de texto relacionada con la fecha en el momento considerado.

    # Impresión de los resultados correspondientes.
    print("Idioma detectado:", sl_aux_010_ID )
    print("Transcripción es:", sl_aux_011_TE )
    print("Transcripción en:", sl_aux_012_TI )

    # Transcripción sobre la ruta de default.
    ## Se hace uso de un dataframe para almacenar los datos finales de la transcripción en varios idiomas.
    trancript_dict = [ {    "Fecha Transcripcion": str( sl_aux_013_CF ) ,
                            "Modelo Utilizado": str( Model_Version ) ,
                            "Archivo Audio": str( Audio_Rec_File ) ,
                            "Archivo Transcrito": str( Transcripted_Audio_File ) ,
                            "Transcripcion Español": str( sl_aux_011_TE )  ,
                            "Transcripción Ingles": str( sl_aux_012_TI )  ,
                            "Idioma Detectado": str( sl_aux_010_ID )        
                    } ] 

    # Información de la frase transcrita almacenada en un archivo csv. (Considerar la opción de concatenar).
    df_transcript = pd.DataFrame(trancript_dict)
    df_transcript.to_csv(Transcripted_Audio_File, index=False)

    return df_transcript
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Función para grabar directamente del microfono seleccionado, esperar un tiempo y de forma automática usar whisper para transcribir el texto escuchado.
## En el momento en el que se detecte hablar a una persona via webcam accionar las funciones de grabar y transcribir.
#######################################################################################################################################################################################
def Transcribe_Audio_Mic_Full( duracion = 5 , frecuencia_muestreo = 44100 , indice_dispositivo = 6 , Audio_Rec_File = sg_aux_004_FNAR, Transcripted_Audio_File = sg_aux_005_FNAT, Model_Version = "large" ):
    
    # Graba el audio del micrófono
    Record_Audio_File_Mic(duracion=duracion , frecuencia_muestreo=frecuencia_muestreo, indice_dispositivo=indice_dispositivo, Audio_Rec_File = Audio_Rec_File)

    # Transcribe el audio grabado y detecta el idioma.
    df_aux_01 = Transcribe_Audio_From_File(Audio_Rec_File = Audio_Rec_File , Transcripted_Audio_File = Transcripted_Audio_File , Model_Version = Model_Version)
            
    return df_aux_01['Transcripcion Español']
    #
#Transcribe_Audio_Mic_Full( indice_dispositivo = 5 )
aux_0 = Transcribe_Audio_Mic_Full( indice_dispositivo = 5 , Model_Version = "large")
str(aux_0[0])
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Ciclo para iniciar toda la ejecucion.
#######################################################################################################################################################################################
def Main_Loop( arg ):
    print("Entre al main loop")

    # Ciclo paralelo para considerar la ejecucion de Whisper.
    while( True ):
        #################################################################################################################################################
        # Esperar para no saturar el hardware de la computadora. Se considera una frecuencia de monitoreo de 144 hz.
        print("Monitoreando variables")
        time.sleep(1/144)

        #################################################################################################################################################
        # Parte de Whisper.
        if( arg.Is_Whisper_Active and arg.C_Is_Whisper_Active == 1 ):
            # Medir tiempos de ejecucion de whisper para considerar ejecuciones.
            arg.Is_Whisper_On_Loop = True
            t_whisper_i = time.time()
            Whisper_Aux_01 = Transcribe_Audio_Mic_Full( indice_dispositivo = 6 , Model_Version = "large")
            Whisper_Aux_01 = Whisper_Aux_01[0]
            t_whisper_f = time.time()

            # Actualizacion de las variables referentes a los estados de control.
            arg.Is_Whisper_Finalizated = True
            arg.Is_Whisper_Active = False
            arg.Whisper_Text_Transcripted = Whisper_Aux_01
            arg.The_LLAMA3_TEXT = arg.Whisper_Text_Transcripted
            arg.Whisper_Time = t_whisper_f - t_whisper_i
            arg.Is_Whisper_On_Loop = False
            arg.C_Is_Whisper_Active += 1

        #################################################################################################################################################
        # Parte Langchain.
        if( arg.Is_LLAMA3_Active and arg.C_Is_LLAMA3_Active == 1 ):
            # Medir tiempos de ejecucion de llama3 para considerar ejecuciones.
            t_llama3_i_01 = time.time()
            LLama3_Aux_02 = Correct_Answer_31( str(arg.The_LLAMA3_TEXT) )
            t_llama3_f_01 = time.time()

            # Declaraciones de las variables de tiempo iguales a cero si no se utilizaran:
            t_llama3_i_02 = 0.0
            t_llama3_f_02 = 0.0

            # Considerar caso en donde se desea mover el dron.
            if( LLama3_Aux_02 == "Movimiento" ):
                # Medir tiempos de ejecucion de llama3 para considerar ejecuciones.
                # Detectar la direccion positiva o negartiva y en que eje me deseo mover del mundo 3d.
                t_llama3_i_02 = time.time()
                LLama3_Aux_03 = Correct_Answer_32( str(arg.The_LLAMA3_TEXT))
                t_llama3_f_02 = time.time()
            
            # Tag para indicar que no existe movimiento.
            else:
                LLama3_Aux_03 = "No Movimiento"

            # Actualizacion de las variables referentes a los estados de control.
            arg.Is_LLAMA3_Finalizated = True
            arg.Is_LLAMA3_Active = False
            arg.C_Is_LLAMA3_Active = 0
            arg.The_LLAMA3_Text_Tag_01 = LLama3_Aux_02
            arg.The_LLAMA3_Text_Tag_02 = LLama3_Aux_03
            arg.The_LLAMA3_Time = ( t_llama3_f_01 + t_llama3_f_02 - t_llama3_i_01 - t_llama3_i_02 )/2
            arg.Is_LLAMA3_On_Loop = False

        #################################################################################################################################################
        # Parte para detener la ejecucion concurrente.
        if( stop_requested() ): 
            print("Sali del while")
            break

#######################################################################################################################################################################################



#######################################################################################################################################################################################
# Parte para ejecutar el loop de monitoreo de variables.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
f1 = executor.submit(Main_Loop, Z_00)
#######################################################################################################################################################################################


#######################################################################################################################################################################################
# Funcion para ejecutar el proceso de captura de la camara de forma paralela. 
## Se intento utilizar ROS2 pero daba los mismos resultados que si se utilizaba unicamente CPU.
### No realizar movimientos de cabeza o labios bruscos ya que la libreria realizar predicciones del movimiento de los labios.
#### Recursos para la deteccion de video a partir de la webcam detectada (usar en Ubuntu y solo con la webcam de la laptop).
#######################################################################################################################################################################################
# Parte de la captura de video en camara.
aux_cv_cap = cv2.VideoCapture(0)                                                    # Definir el objeto de captura de video
aux_cv_fourcc = cv2.VideoWriter_fourcc(*'H264')                                     # Definir el codec como H264 y crear un objeto VideoWriter
aux_cv_out = cv2.VideoWriter(sg_aux_008_FNVR, aux_cv_fourcc, 60.0, (1080, 720))   
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)                             

# Variables de control para obtener la informacion cada i frames. Tienen mayor impacto en la deteccion de frames.
c_i_frames = 0
c_i_frames_a = 100                                                                  # Frames para imprimir la posicion central de los labios.
c_i_frames_b = 10                                                                   # Frames para comprobar si se estan moviendo los labios.
c_i_conv_s_frame = 60                                                               # Conversion de cantidad de frames por segundo.
c_introduction_frames = 1.5*c_i_conv_s_frame                                          # Cantidad de frames para mostrar el mensaje de bienvenida.
# cg_LLAMA3 = False

# Realizar la lectura del dataframe auxiliar construido considerando los puntos que se localizan en la parte superior como en la inferior
df_sec_lips_aux = pd.read_csv(sg_aux_014_FLT)
lips_dict = []                                                                      # Variable utilizada para el almacenamiento de datos en archivo tipo csv.

# Sumas considerando los valores de comparar de la columna PUNTO_02 con la columna PARTE_LABIO_02, si son del conjunto SUPERIOR sumar a una variable, si son del INFERIOR sumar a otra.
aux_sum_sup_lips_y = 0.0
aux_sum_inf_lips_y = 0.0
aux_diff_sum_lips_y = 0.0
aux_mean_diff_sum_lips_y = 0.0
aux_mean_diff_sum_lips_last_y = 0.0
aux_lip_movement_threshold_y = 0.008                                                # Considerar mediante pruebas que tanto se tiene que abrir la boca para considerar que se esta hablando o se desea hablar.

# Variables asignadas a los colores de puntos en pantalla.
color_l1 = (0, 0, 255)                                                              # Color (B, G, R) de los puntos referentes a los labios y estado default.
color_l2 = (0, 255, 0)                                                              # Color (B, G, R) del estatus de apertura de los labios. 
color_l3 = (255, 0, 0)                                                              # Color (B, G, R) del estatus de movimiento de los labios. (Comparar frarme anterior).
radius = 1                                                                          # Radio del círculo

# Loop para grabar de forma directa, se abrirar despues de capturar el video con los valores asociados al maximo y minimo de la apertura de los labios.
while(aux_cv_cap.isOpened()):
    success, frame = aux_cv_cap.read()

    # Si se detecta algun frame con exito, se procede a realizar la captura de video correspondiente.
    if(success):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # Convertir la imagen de BGR a RGB
        results = face_mesh.process(frame)                      # Procesar la imagen y encontrar los puntos clave faciales
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)          # Dibujar los puntos clave faciales en la imagen
        
        if(results.multi_face_landmarks):

            # Ciclo for deteccion de landmarks del rostro. 
            for face_landmarks in results.multi_face_landmarks:

                # Ciclo para detectar los puntos relacionados a los labios y colorearlos en tiempo real.
                for l1, l2 in np.array( list( mp_face_mesh.FACEMESH_LIPS ) ):
                    aux_l1_x = face_landmarks.landmark[l1].x                                    # Obtencion de los pares de puntos en la coordenada X.         
                    aux_l1_y = face_landmarks.landmark[l1].y                                    # Obtencion de los pares de puntos en la coordenada Y.   
                    aux_abs_l1_x = int(aux_l1_x * frame.shape[1])                               # Coordenada X en la imagen normalizada.
                    aux_abs_l1_y = int(aux_l1_y * frame.shape[0])                               # Coordenada Y en la imagen normalizada.
                    aux_l1 = (aux_abs_l1_x, aux_abs_l1_y)                                       # Construcion de los vectores para colorear en la imagen.
                    cv2.circle(frame, aux_l1, radius, color_l1 , -1)                            # Dibujar el círculo en el frame en el primer conjunto.

                    # Preguntar si l1 se encuentra dentro del archivo csv de puntos etiquetados con las secciones de los labios. 
                    if( l1 in df_sec_lips_aux['PUNTO_02'].values ):
                        ls1 = df_sec_lips_aux.loc[df_sec_lips_aux['PUNTO_02'] == l1, 'PARTE_LABIO_02'].values[0]
                        if( ls1 == "SUPERIOR" ):    aux_sum_sup_lips_y += aux_l1_y              # Preguntar si se trata de un punto presente en la seccion superior de los labios.
                        if( ls1 == "INFERIOR" ):    aux_sum_inf_lips_y += aux_l1_y              # Preguntar si se trata de un punto presente en la seccion inferior de los labios.

            # Calcular la diferencia entre la suma de coordenadas y en labios superiores vs inferiores y su correspondiente division entre 19 correspondiente al promedio. 
            aux_diff_sum_lips_y = abs( aux_sum_sup_lips_y - aux_sum_inf_lips_y) 
            aux_mean_diff_sum_lips_y = aux_diff_sum_lips_y/19
            aux_temp_diff_lips = abs( aux_mean_diff_sum_lips_y - aux_mean_diff_sum_lips_last_y )
            b_aux_00 = bool( aux_temp_diff_lips >= aux_lip_movement_threshold_y)                                        # Condicion de diferencia con frame anterior.

            # Colorear la esquina superior izquierda para saber si se detecta o no movimiento de los labios. 
            if( Z_00.See_Text_On_Camera ):
                cv2.circle(frame, ( 7 , 15 ), 5, color_l1*(b_aux_00) + color_l3*( not b_aux_00) , -1)                   # Dibujar el círculo para monitorear el movimiento de labios.

            # Mensaje de bienvenida. 
            if( c_i_frames < c_introduction_frames - 1 and Z_00.See_Text_On_Camera ):
                cv2.putText(frame, "Deteccion de Movimiento " , ( 50 , frame.shape[0] // 4 + 40 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )
                cv2.putText(frame, "de labios para usar Whisper " , ( 50 , frame.shape[0] // 4 + 80 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )

            # Mensaje para considerar la deteccion del movimiento de labios.
            if( ( not Z_00.Is_Whisper_Active ) and c_i_frames >= c_introduction_frames and ( not Z_00.Is_Whisper_Finalizated ) and Z_00.See_Text_On_Camera ):
                cv2.putText(frame, "Por Favor Hable : " , ( 50 , frame.shape[0] // 4 + 40 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )

            # Si se esta capturando el audio a transcribir con whisper.
            if( Z_00.Is_Whisper_Active and c_i_frames >= c_introduction_frames and ( not Z_00.Is_Whisper_Finalizated ) and Z_00.See_Text_On_Camera ):
                cv2.putText(frame, "Capturando Audio " , ( 50 , frame.shape[0] // 4 + 40 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )
                cv2.putText(frame, "Porfavor Espere ..." , ( 50 , frame.shape[0] // 4 + 80 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,255,255), 2 )

            ###################################################################################################################################################################################
            # Deteccion del movimiento de los labios. Mandar a llamar a la funcion de activacion de Whisper.
            # Se pone en pausa la imagen despues de la deteccion. (Da igual la localizacion del condicional).
            ###################################################################################################################################################################################
            if( b_aux_00 and ( not Z_00.Is_Whisper_Active ) and ( c_i_frames >= c_introduction_frames ) and ( Z_00.C_Is_Whisper_Active == 0 ) ):
                Z_00.Is_Whisper_Active = True
                Z_00.C_Is_Whisper_Active += 1

            # Texto detectado. Mensaje para imprimir en pantalla el texto transcrito.
            if( Z_00.Is_Whisper_Finalizated and ( not Z_00.Is_Whisper_Active ) and Z_00.See_Text_On_Camera and ( not Z_00.Is_LLAMA3_Active ) and ( not Z_00.Is_LLAMA3_Finalizated ) ):
                # Imprimir el texto transcrito en camara.
                cv2.putText(frame, "Texto Transcrito: " , ( 50 , frame.shape[0] // 4 + 120 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )
                cv2.putText(frame, Z_00.Whisper_Text_Transcripted , ( 50 , frame.shape[0] // 4 + 160 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255), 2 )
                
                # Variable para poder avanzar a utilizar llama3.
                cg_LLAMA3 = True

            ###################################################################################################################################################################################
            # Parte de la clasificacion debida a la deteccion del prompt transcrito con whisper.
            ###################################################################################################################################################################################
            # Trtigger para mandar a llamar a llama3:
            if( cg_LLAMA3 and ( not Z_00.Is_LLAMA3_Active ) and ( c_i_frames >= c_introduction_frames ) and ( Z_00.C_Is_LLAMA3_Active == 0 ) ):
                Z_00.Is_LLAMA3_Active = True
                Z_00.C_Is_LLAMA3_Active += 1


            # Si se esta capturando el audio a transcribir con whisper.
            if( Z_00.Is_LLAMA3_Active and c_i_frames >= c_introduction_frames and ( not Z_00.Is_LLAMA3_Finalizated ) and Z_00.See_Text_On_Camera and cg_LLAMA3 ):
                cv2.putText(frame, "Clasificando Prompt " , ( 50 , frame.shape[0] // 4 + 40 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )
                cv2.putText(frame, "Porfavor Espere ..." , ( 50 , frame.shape[0] // 4 + 80 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,255,255), 2 )


            # Texto detectado. Mensaje para imprimir en pantalla el texto transcrito.
            if( Z_00.Is_LLAMA3_Finalizated and ( not Z_00.Is_LLAMA3_Active ) and Z_00.See_Text_On_Camera and cg_LLAMA3 ):
                # Imprimir el texto transcrito en camara.
                cv2.putText(frame, "Class 1: " , ( 50 , frame.shape[0] // 4 + 120 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )
                cv2.putText(frame, Z_00.The_LLAMA3_Text_Tag_01 , ( 290 , frame.shape[0] // 4 + 120 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255), 2 )
                cv2.putText(frame, "Class 2: " , ( 50 , frame.shape[0] // 4 + 160 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255,255,255), 2 )
                cv2.putText(frame, Z_00.The_LLAMA3_Text_Tag_02 , ( 290 , frame.shape[0] // 4 + 160 ) , cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255), 2 )

            ###################################################################################################################################################################################

            # Variable de control que se actualiza cada frame.
            c_i_frames += 1

            # Concatenacion de los elementos dentro del arreglo correspondiente.
            lips_dict_aux = {
                "INDEX": str( c_i_frames ) ,
                "L1_X": str( aux_l1_x ) ,
                "L1_Y": str( aux_l1_y ) ,
                "SUM_SUP_LIPS_Y" : str( aux_sum_sup_lips_y ),
                "SUM_INF_LIPS_Y" : str( aux_sum_inf_lips_y ),
                "DIFF_SUP_INF_LIPS_Y" : str( aux_diff_sum_lips_y ),
                "MEAN_SUP_INF_LIPS_Y" : str( aux_mean_diff_sum_lips_y ),
                "MEAN_SUP_INF_LIPS_LAST_Y" : str( aux_mean_diff_sum_lips_last_y ),
                "TEMP_DIFF_MEAN_LIPS" : str( aux_temp_diff_lips ),
                "QUEUE_DATE": str( datetime.now().strftime(sg_aux_999_FF) )
            }
            lips_dict.append(lips_dict_aux)

            # Volver cero las variables relacionadas al calculo de los movimientos de los labios. 
            # Considerar la variable de diferencia correspondiente asignada para que sea el valor del frame anterior.
            aux_sum_sup_lips_y = 0.0
            aux_sum_inf_lips_y = 0.0
            aux_diff_sum_lips_y = 0.0
            aux_mean_diff_sum_lips_last_y = aux_mean_diff_sum_lips_y
            aux_mean_diff_sum_lips_y = 0.0
            

        # Bosquejo realizado sobre el frame final para obtener los landmarks correspondientes al rostro. 
        aux_cv_out.write(frame)
        cv2.imshow('MediaPipe FaceMesh', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            aux_cv_cap.release()                                        # Liberar el dispositivo de captura.
            aux_cv_out.release()                                        # Liberar el dispositivo de salida.
            cv2.destroyAllWindows()                                     # Cerrar las ventanas donde se muestra la salida de la webcam.
            df_lips_detection = pd.DataFrame(lips_dict)                 # Conversion de arreglo de diccionarios a dataframe.
            df_lips_detection.to_csv(sg_aux_010_FNLP, index=False)      # Almacenamiento del archivo csv.
            cv2.imwrite( sg_aux_011_FNLP , frame)                       # Almacenamiento del ultimo frame.

            # Detencion de los ciclos concurrentes en paralelo:
            stop_flag = True
            executor.shutdown(True)
            stop_flag = False

            break
    else:
        break
#######################################################################################################################################################################################

'''
Z_00.Whisper_Text_Transcripted
Z_00.Is_Whisper_Active
Z_00.Is_Whisper_Finalizated
Z_00.C_Is_Whisper_Active
Z_00.Is_Whisper_On_Loop
Z_00.Whisper_Time

Z_00.Is_Whisper_Finalizated = True
Z_00.Is_Whisper_Finalizated and ( not Z_00.Is_Whisper_Active ) and Z_00.See_Text_On_Camera


Z_00.Is_LLAMA3_Finalizated
Z_00.Is_LLAMA3_Active
Z_00.C_Is_LLAMA3_Active
Z_00.The_LLAMA3_Text_Tag_01
Z_00.The_LLAMA3_Text_Tag_02
Z_00.The_LLAMA3_Time
Z_00.Is_LLAMA3_On_Loop

Z_00.Is_LLAMA3_Finalizated 
Z_00.Is_LLAMA3_Active = False

( not Z_00.Is_LLAMA3_Active ) and Z_00.See_Text_On_Camera and cg_LLAMA3

cg_LLAMA3
Z_00.Is_LLAMA3_Active

bool( Z_00.Is_LLAMA3_Active and c_i_frames >= c_introduction_frames and ( not Z_00.Is_LLAMA3_Finalizated ) and Z_00.See_Text_On_Camera and cg_LLAMA3 )

bool( Z_00.Is_LLAMA3_Finalizated and ( not Z_00.Is_LLAMA3_Active ) and Z_00.See_Text_On_Camera and cg_LLAMA3 )

( Z_00.Is_LLAMA3_Finalizated and ( not Z_00.Is_LLAMA3_Active ) and Z_00.See_Text_On_Camera and cg_LLAMA3 )

( Z_00.Is_LLAMA3_Finalizated and ( not Z_00.Is_LLAMA3_Active ) and Z_00.See_Text_On_Camera and cg_LLAMA3 )

( Z_00.Is_LLAMA3_Finalizated and ( not Z_00.Is_LLAMA3_Active ) and Z_00.See_Text_On_Camera and cg_LLAMA3 )

Z_00.Is_LLAMA3_Finalizated
Z_00.Is_LLAMA3_Active
cg_LLAMA3 = True

# Tiempos de ejecucion de cada modelo en Ubuntu 22.04 LTS.
# WHISPER:  59.5974 segundos.
# LLAMA 3:  51.8411 segundos.
# LLAMA 2:  

aux_cv_cap.release()
aux_cv_out.release()
cv2.destroyAllWindows()


# Parte de medicion de transcricion de whisper.

t_whisper_i = time.time()
Whisper_Aux_01 = Transcribe_Audio_Mic_Full( indice_dispositivo= 6 , Model_Version = "large")
t_whisper_f = time.time()
a_whisper_time = t_whisper_f - t_whisper_i 
print("Tiempo de ejecucion: " + str( a_whisper_time ) )


### Sesgos a considerar:
# Angulo de visualizacion a la camara.
# Distancia de separacion de la cara a la camara.
# Movimientos fuertes del cuerpo.
# Movimientos fuertes de la cara o cabeza.
# Tapar los labios, el sistema de mediapipe realiza la prediccion pero no es capaz de detecar movimiento con labios tapados.
# Entorno con demasiada luz.
# Saturacion de la memoria ram afecta el rendimiento de la camara.


# Implementar PID
# Validar funcionamiento entregable


llm_3 = Ollama( model="llama3", 
                # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature = 0.9
            )


llm_2 = Ollama( model="llama2", 
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature = 0.9
            )


chain_2 = LLMChain(llm=llm_2, prompt=prompt, verbose=False)
chain_3 = LLMChain(llm=llm_3, prompt=prompt, verbose=False)

# Definir una funcion para ver si existe el tag final dentro de la respuesta.
def Correct_Answer_3( S_aux_whisper_text ):
    S_Aux_Tags = str( chain_3.run( S_aux_whisper_text ) )

    if( "Recuperar Posicion" in S_Aux_Tags ): return "Recuperar Posicion" 
    if( "Recuperar Velocidad" in S_Aux_Tags ): return "Recuperar Velocidad" 
    if( "Recuperar Orientacion " in S_Aux_Tags ): return "Recuperar Orientacion"
    if( "Recuperar Velocidad Angular" in S_Aux_Tags ): return "Recuperar Velocidad Angular"
    if( "Marcar un punto" in S_Aux_Tags ): return "Marcar un punto"

    if( "Movimiento" in S_Aux_Tags ): return "Movimiento"

    if( "Aterrizar" in S_Aux_Tags ): return "Aterrizar"
    if( "Despegar" in S_Aux_Tags ): return "Despegar"
    if( "Detener" in S_Aux_Tags ): return "Detener"
    # Si se detecto movimiento realizar un segundo procesamiento de lenguaje.


Correct_Answer_3( "The Dron it's rotating?" )



t_llama3_i = time.time()
print(ollama_2('why is the sky blue?'))
t_llama3_f = time.time()
a_llama3_time = t_llama3_f - t_llama3_i 
print("Tiempo de ejecucion: " + str( a_llama3_time ) )

def Correct_Answer_2( S_aux_whisper_text ):
    S_Aux_Tags = str( chain_2.run( S_aux_whisper_text ) )

    if( "Recuperar Posicion" in S_Aux_Tags ): return "Recuperar Posicion" 
    if( "Recuperar Velocidad" in S_Aux_Tags ): return "Recuperar Velocidad" 
    if( "Recuperar Orientacion " in S_Aux_Tags ): return "Recuperar Orientacion"
    if( "Recuperar Velocidad Angular" in S_Aux_Tags ): return "Recuperar Velocidad Angular"
    if( "Marcar un punto" in S_Aux_Tags ): return "Marcar un punto"

    if( "Movimiento" in S_Aux_Tags ): return "Movimiento"

    if( "Aterrizar" in S_Aux_Tags ): return "Aterrizar"
    if( "Despegar" in S_Aux_Tags ): return "Despegar"
    if( "Detener" in S_Aux_Tags ): return "Detener"
    # Si se detecto movimiento realizar un segundo procesamiento de lenguaje.

# Zona para graficar
x = range(1,21)
y_en_llama3 = []
y_es_llama3 = []
y_en_llama2 = []
y_es_llama2 = []

# Arreglo para considerar los prompts de entrada:
Helping_Prompts_EN = [  "I want to go to a position",
                        "Move the dron 2 meters to the right",
                        "what is the actual position of the Dron",
                        "How far it's the dron from the origin?",
                        "How fast the dron it moving?",
                        "How fast the dron it's rotating?",
                        "How many revolutions per second it's performing the dron",
                        "I want to put a tag in this point",
                        "Recover x,y,z",
                        "Recover Vx,Vy,Vz",
                        "Recover Wx,Wy,Wz",
                        "Where is facing the Dron?",
                        "Land please",
                        "Take off the Dron please",
                        "Can you stop Dron please?",
                        "I don't going to use the dron know, can you please stop the dron?",
                        "If the Dron is stopped, please land the dron.",
                        "If the Dron has Movement, please stop the dron",
                        "Is the dron moving right now?",
                        "The Dron it's rotating?"
                    ]


Helping_Prompts_ES = [  "Quiero ir a una posición",
                        "Mueve el dron 2 metros hacia la derecha",
                        "¿Cuál es la posición actual del Dron?",
                        "¿A qué distancia está el dron del origen?",
                        "¿Qué tan rápido se está moviendo el dron?",
                        "¿Qué tan rápido está rotando el dron?",
                        "¿Cuántas revoluciones por segundo está realizando el dron?",
                        "Quiero poner una etiqueta en este punto",
                        "Recuperar x,y,z",
                        "Recuperar Vx,Vy,Vz",
                        "Recuperar Wx,Wy,Wz",
                        "¿Hacia dónde está mirando el Dron?",
                        "Por favor, aterriza",
                        "Por favor, haz despegar al Dron",
                        "¿Puedes detener al Dron, por favor?",
                        "No voy a usar el dron ahora, ¿puedes detener el dron?",
                        "Si el Dron está detenido, por favor aterriza el dron.",
                        "Si el Dron tiene movimiento, por favor detén el dron",
                        "¿Se está moviendo el dron en este momento?",
                        "¿El Dron está rotando?",
                    ]

for i in range(0,20):
    # Parte de ingles llama 3
    t_llama3_en_i = time.time()
    Correct_Answer_3( str( Helping_Prompts_EN[i] ) )
    t_llama3_en_f = time.time()
    y_en_llama3.append( t_llama3_en_f - t_llama3_en_i )

    # Parte de español llama 3
    t_llama3_es_i = time.time()
    Correct_Answer_3( str( Helping_Prompts_ES[i] ) )
    t_llama3_es_f = time.time()
    y_es_llama3.append( t_llama3_es_f - t_llama3_es_i )

    # Parte de ingles llama 3
    t_llama2_en_i = time.time()
    Correct_Answer_2( str( Helping_Prompts_EN[i] ) )
    t_llama2_en_f = time.time()
    y_en_llama2.append( t_llama2_en_f - t_llama2_en_i )

    # Parte de español llama 3
    t_llama2_es_i = time.time()
    Correct_Answer_2( str( Helping_Prompts_ES[i] ) )
    t_llama2_es_f = time.time()
    y_es_llama2.append( t_llama2_es_f - t_llama2_es_i )



# Run the chain only specifying the input variable.
t_llama3_i = time.time()
print( Correct_Answer( "I want to go to a position" ) )
print( Correct_Answer( "Move the dron 2 meters to the right" ) )
print( Correct_Answer( "what it's the actual position of the Dron" ) )
print( Correct_Answer( "How far it's the dron from the origin?" ) )
print( Correct_Answer( "How fast the dron it moving?" ) )
print( Correct_Answer( "How fast the dron it's rotating?" ) )
print( Correct_Answer( "How many revolutions per second it's performing the dron" ) )
print( Correct_Answer( "I want to put a tag in this point" ) )
print( Correct_Answer( "Recover x,y,z" ) )
print( Correct_Answer( "Recover Vx,Vy,Vz" ) )
print( Correct_Answer( "Recover Wx,Wy,Wz" ) )
print( Correct_Answer( "Where is facing the Dron?" ) )
print( Correct_Answer( "Land please" ) )
print( Correct_Answer( "Take off the Dron please" ) )
print( Correct_Answer( "Can you stop Dron please?" ) )
print( Correct_Answer( "I don't going to use the dron know, can you please stop the dron?" ) )
print( Correct_Answer( "If the Dron is stopped, please land the dron." ) )
print( Correct_Answer( "If the Dron has Movement, please stop the dron" ) )
print( Correct_Answer( "Is the dron moving right now?" ) )
print( Correct_Answer( "The Dron it's rotating?" ) )
t_llama3_f = time.time()

a_llama3_time = t_llama3_f - t_llama3_i 
print("Tiempo de ejecucion: " + str( a_llama3_time ) )

Correct_Answer_2( "I want to go to a position" )

y_en_llama2
y_en_llama3
y_es_llama2
y_es_llama3

# Graficar la parte de los tiempos con los prompts en ingles
plt.figure( num = None, dpi = 1400, facecolor = 'w', edgecolor = 'r' )
plt.xlabel( " Number of sentence " )
plt.ylabel( " Time Elapsed to answer (s) " )
plt.title( "Llama 2 vs Llama 3 - English", fontsize = 12 )
plt.grid( True )
plt.grid( color = '0.5', linestyle = '--', linewidth = 0.3 )
plt.plot( x, y_en_llama3, 'ro-', linewidth = 0.850, label = "LLama 3" )
plt.plot( x, y_en_llama2, 'go-', linewidth = 0.850, label = "LLama 2"  )
plt.legend( loc = 'upper right' )
plt.savefig( sg_aux_016_FF_llamac_en , bbox_inches = 0, dpi = 1400 )
plt.close() 

# Graficar la parte de los tiempos con los prompts en español
plt.figure( num = None, dpi = 1400, facecolor = 'w', edgecolor = 'r' )
plt.xlabel( " Number of sentence " )
plt.ylabel( " Time Elapsed to answer (s) " )
plt.title( "Llama 2 vs Llama 3 - Spanish", fontsize = 12 )
plt.grid( True )
plt.grid( color = '0.5', linestyle = '--', linewidth = 0.3 )
plt.plot( x, y_es_llama3, 'ro-', linewidth = 0.850, label = "LLama 3" )
plt.plot( x, y_es_llama2, 'go-', linewidth = 0.850, label = "LLama 2"  )
plt.legend( loc = 'upper right' )
plt.savefig( sg_aux_016_FF_llamac_es , bbox_inches = 0, dpi = 1400 )
plt.close() '''
