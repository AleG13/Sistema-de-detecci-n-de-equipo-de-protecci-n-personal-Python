import os
import json
from PIL import Image

# Rutas de entrada
images_dir = "images"
labels_dir = "labels"
output_dir = "labelme_json"
os.makedirs(output_dir, exist_ok=True)

# Nombres de clases opcional (si lo tienes)
classes = ["Con_Casco", "Sin_Casco_Mascarrilla", "Sin_Chaleco", "Con_Chaleco", "Con_Mascarrilla"]

def yolo_to_labelme(image_path, label_path):
    image = Image.open(image_path)
    width, height = image.size

    image_filename = os.path.basename(image_path)
    label_filename = os.path.basename(label_path)

    with open(label_path, "r") as f:
        lines = f.readlines()

    shapes = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, w, h = map(float, parts)
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        # Convertir a puntos de esquina
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        shape = {
            "label": classes[int(class_id)] if int(class_id) < len(classes) else str(class_id),
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }

        shapes.append(shape)

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    output_path = os.path.join(output_dir, image_filename.replace(".jpg", ".json").replace(".png", ".json"))
    with open(output_path, "w") as f:
        json.dump(labelme_data, f, indent=4)

    print(f"Convertido: {label_filename} -> {os.path.basename(output_path)}")

# Recorrer todos los archivos
for txt_file in os.listdir(labels_dir):
    if txt_file.endswith(".txt"):
        image_name = txt_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            image_name = txt_file.replace(".txt", ".png")
            image_path = os.path.join(images_dir, image_name)

        label_path = os.path.join(labels_dir, txt_file)

        if os.path.exists(image_path):
            yolo_to_labelme(image_path, label_path)
