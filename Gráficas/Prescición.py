import pandas as pd
import matplotlib.pyplot as plt

# Ruta del CSV
csv_path = r"C:\Users\edgar\OneDrive - UVG\TESIS\TensorFlow\Detector_EPP\runs\detect\train29\results.csv"

# Leer el CSV
df = pd.read_csv(csv_path)

# Columna correcta de precisión
precision_column = 'metrics/mAP50(B)'

# Graficar precisión vs épocas
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df[precision_column], marker='o', color='g', label='Precisión ')

# Añadir valores sobre cada punto
for x, y in zip(df['epoch'], df[precision_column]):
    plt.text(x, y + 0.005, f"{y:.3f}", ha='center', va='bottom', fontsize=8)

plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión del Modelo')
plt.grid(True)
plt.legend()
plt.show()
