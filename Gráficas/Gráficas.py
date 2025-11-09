import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Rutas ---
labels_dir = r"C:\Users\edgar\OneDrive - UVG\TESIS\TensorFlow\Detector_EPP\train\labels"
data_yaml = r"C:\Users\edgar\OneDrive - UVG\TESIS\TensorFlow\Detector_EPP\data.yaml"

# --- Cargar nombres de clases desde data.yaml ---
with open(data_yaml, 'r') as f:
    data = yaml.safe_load(f)
names = data['names']  # diccionario {id: "nombre"}

# --- Leer todas las etiquetas ---
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

y_true = []
y_pred = []

for file in label_files:
    path = os.path.join(labels_dir, file)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = float(parts[0])
            # Ignorar background (por ejemplo clase 0)
            if class_id != 0:
                class_name = names[class_id]
                y_true.append(class_name)
                y_pred.append(class_name)  # aquí se puede poner la predicción si difiere

# --- Etiquetas únicas (para ejes de la matriz) ---
labels = sorted(list(set(y_true) | set(y_pred)))

# --- Crear matriz de confusión normalizada ---
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]  # normalizar por fila

# --- Graficar ---
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas reales")
plt.title("Matriz de Confusión Normalizada - YOLO (sin background)")
plt.show()
