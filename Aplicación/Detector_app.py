import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import serial
import serial.tools.list_ports
import os
import sys
from datetime import datetime

# ================================
# CONFIGURACIÓN INICIAL
# ================================
st.set_page_config(page_title="Detección de EPP", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #B5B0B0;'>Detección de EPP</h1>
""", unsafe_allow_html=True)

# === Manejo de rutas para PyInstaller ===
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

def resource_path(relative_path):
    return os.path.join(base_path, relative_path)

# ================================
# CARGAR MODELO YOLO
# ================================
model_path = resource_path(os.path.join("runs", "detect", "train29", "weights", "best.pt"))
model = YOLO(model_path)

# ================================
# CONFIGURACIÓN SERIAL AUTOMÁTICA
# ================================
st.sidebar.subheader("🔌 Conexión Serial")

# Guardar conexión serial en session_state
if "serial_con" not in st.session_state:
    st.session_state.serial_con = None

puertos = list(serial.tools.list_ports.comports())

if not puertos:
    st.sidebar.error("⚠️ No se detectaron puertos COM disponibles")
else:
    lista_puertos = [f"{p.device} - {p.description}" for p in puertos]
    st.sidebar.write("Puertos detectados:")
    for p in lista_puertos:
        st.sidebar.write(f"• {p}")

    # Selector manual de puerto
    puerto_seleccionado = st.sidebar.selectbox(
        "Selecciona el puerto COM:",
        [p.device for p in puertos],
        index=0
    )

    # Cerrar conexión anterior si sigue abierta
    if st.session_state.serial_con is not None:
        try:
            if st.session_state.serial_con.is_open:
                st.session_state.serial_con.close()
                time.sleep(0.5)  # liberar el puerto antes de reabrir
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error al cerrar puerto anterior: {e}")

    # Intentar abrir nueva conexión
    try:
        st.session_state.serial_con = serial.Serial(puerto_seleccionado, 115200, timeout=1)
        time.sleep(2)
        st.sidebar.success(f"✅ Conectado al {puerto_seleccionado}")
    except Exception as e:
        st.sidebar.error(f"❌ Error al conectar con {puerto_seleccionado}: {e}")
        st.session_state.serial_con = None

# Alias local
ser = st.session_state.serial_con

# ================================
# DETECCIÓN DE CÁMARAS DISPONIBLES
# ================================
st.sidebar.subheader("📷 Selección de cámara")

def listar_camaras(max_index=5):
    disponibles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles

camaras = listar_camaras()

if not camaras:
    st.sidebar.error("⚠️ No se detectaron cámaras disponibles")
    st.stop()

camara_seleccionada = st.sidebar.selectbox(
    "Selecciona la cámara:",
    camaras,
    index=0,
    format_func=lambda x: f"Cámara {x}"
)

# ================================
# LOGO
# ================================
logo_path = resource_path("Cont.png")
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

def overlay_image(background, overlay, x, y, scale=1.0):
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay
    return background

# ================================
# CONFIGURACIÓN DE CLASES OBJETIVO
# ================================
clases_objetivo = ["Con_Casco", "Con_Mascarrilla", "Con_Chaleco"]

# ================================
# INTERFAZ STREAMLIT
# ================================
col_video, col_indicadores = st.columns([3, 1])
video_placeholder = col_video.empty()
indicadores = {clase: col_indicadores.empty() for clase in clases_objetivo}

Selector = st.selectbox(
    "Selecciona la confianza del modelo",
    ["Media", "Alta", "Baja"],
    index=0,
    key="selector_confianza"
)

confi = {"Baja": 0.2, "Media": 0.5, "Alta": 0.7}[Selector]

stop_placeholder = st.empty()
stop = stop_placeholder.button(" Detener cámara")

# ================================
# CREAR CARPETA DE DETECCIONES
# ================================
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop", "Detector_EPP", "detecciones")
os.makedirs(user_desktop, exist_ok=True)
ultima_deteccion_path = os.path.join(user_desktop, "ultima_deteccion.jpg")
ultima_deteccion_placeholder = st.sidebar.empty()

ultima_captura_tiempo = 0
intervalo_captura = 5  # segundos

# ================================
# INICIO DE CÁMARA SELECCIONADA
# ================================
cap = cv2.VideoCapture(camara_seleccionada)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    st.error(f"No se pudo acceder a la cámara {camara_seleccionada}")
    st.stop()
else:
    st.success(f"Cámara {camara_seleccionada} iniciada correctamente")

# ================================
# BUCLE PRINCIPAL
# ================================
while cap.isOpened() and not stop:
    ret, frame = cap.read()
    if not ret:
        st.warning("No se detecta la cámara")
        break

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (640, 480))

    resultados = model.predict(source=frame_resized, imgsz=640, conf=confi, verbose=False)
    result = resultados[0]
    frame_annotated = result.plot()

    clases_detectadas = set(model.names[int(c)] for c in result.boxes.cls)
    clases_filtradas = [c for c in clases_detectadas if c in clases_objetivo]
    cantidad = len(clases_filtradas)

    # Enviar cantidad al puerto serial
    if ser is not None:
        try:
            ser.write(f"{cantidad}\n".encode())
        except Exception as e:
            print("Error enviando por serial:", e)

    # Captura si EPP completo
    if all(clase in clases_detectadas for clase in clases_objetivo):
        tiempo_actual = time.time()
        if tiempo_actual - ultima_captura_tiempo > intervalo_captura:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(user_desktop, f"deteccion_{timestamp}.jpg")
            cv2.imwrite(filename, frame_annotated)
            cv2.imwrite(ultima_deteccion_path, frame_annotated)
            ultima_captura_tiempo = tiempo_actual
            st.sidebar.success("✅ EPP completo detectado, siga así!")
            ultima_deteccion_placeholder.image(
                ultima_deteccion_path, caption="Última detección", use_column_width=True
            )

    frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", width=700)

    # Indicadores visuales
    for clase in clases_objetivo:
        color = "#455E31" if clase in clases_filtradas else "#943737"
        texto = clase.replace('_', ' ')
        indicadores[clase].markdown(f"""
            <div style='
                background-color:{color};
                padding:15px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-weight:bold;
                font-size:18px;
                margin-bottom:15px;
                box-shadow:0 4px 8px rgba(0,0,0,0.2);
            '>{texto}</div>
        """, unsafe_allow_html=True)

# ================================
# FINALIZAR SISTEMA
# ================================
cap.release()
if ser is not None:
    ser.close()
    print("Puerto serial cerrado")

st.success("Cámara detenida")
