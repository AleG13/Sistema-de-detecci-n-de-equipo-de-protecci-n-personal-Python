import os
import json
from PIL import Image

# Carpetas
json_dir = "labelme_json_adjusted"  # JSONs ya ajustados
images_dir = "images"                # carpeta con las imágenes originales
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

# Diccionario de clases a índices
class_mapping = {
    "Con_Casco": 0,
    "Sin_Casco": 1,
    "Sin_Chaleco": 2,
    "Con_Chaleco": 3,
    "Con_Mascarrilla": 4,
    "Sin_Mascarrilla": 5,
}

def convert_points_to_yolo(points, img_width, img_height):
    # points es [[x1, y1], [x2, y2]]
    x1, y1 = points[0]
    x2, y2 = points[1]

    # Rectángulo delimitador
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Normalizar
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = box_width / img_width
    height_norm = box_height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm

for filename in os.listdir(json_dir):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, filename)
    with open(json_path, "r") as f:
        data = json.load(f)

    # Obtener tamaño de imagen
    img_path = os.path.join(images_dir, data["imagePath"])
    if not os.path.exists(img_path):
        print(f"Imagen no encontrada: {img_path}. Se salta el archivo.")
        continue

    img = Image.open(img_path)
    img_width, img_height = img.size

    yolo_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_mapping:
            # Puedes imprimir si quieres saber etiquetas no mapeadas
            # print(f"Etiqueta no mapeada: {label} en {filename}")
            continue  # Ignorar etiquetas no mapeadas

        class_id = class_mapping[label]
        points = shape["points"]
        x_c, y_c, w, h = convert_points_to_yolo(points, img_width, img_height)

        yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Guardar archivo .txt con mismo nombre que JSON pero extensión .txt
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"Generado: {txt_filename}")

