from ultralytics import YOLO

# 1. Cargar un modelo pre-entrenado
model = YOLO("yolov8n.pt")

# 2. Entrenar en tu dataset
model.train(
    data="data.yaml",   # ruta al archivo de configuración
    epochs=100,          # número de épocas
    imgsz=640,          # tamaño de imagen
    batch=16,           # ajusta según RAM/GPU
    workers=0,         # hilos de carga


)
# 3. Validar después del entrenamiento
# Evaluar métricas en entrenamiento


