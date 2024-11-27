from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

class AutoLabeler:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.coords = []  # Lista para coordenadas temporales
        self.all_boxes = []  # Lista para almacenar todos los cuadros
        self.confirmed = False
        self.image_copy = None
        self.current_image_path = None

    def on_click(self, event: MouseEvent):
        """Maneja los clics para crear nuevos cuadros."""
        if event.inaxes and not self.confirmed:
            x, y = event.xdata, event.ydata
            print(f"Coordenadas del clic: ({x:.2f}, {y:.2f})")
            
            self.coords.append((x, y))
            
            if len(self.coords) == 2:
                # Agregar el nuevo cuadro a la lista de todos los cuadros
                self.all_boxes.append(self.coords.copy())
                print(f"Nuevo cuadro añadido: {self.coords}")
                self.draw_all_boxes()
                self.coords = []  # Limpiar para el siguiente cuadro
    
    def on_key_press(self, event):
        """Maneja las teclas para confirmar o reiniciar."""
        if event.key == 'enter':
            print("Cuadros confirmados")
            self.confirmed = True
            plt.close()
        elif event.key == 'r':
            print("Reiniciando todos los cuadros...")
            self.coords = []
            self.all_boxes = []
            self.image_copy = Image.open(self.current_image_path).copy()
            plt.clf()
            plt.imshow(self.image_copy)
            plt.draw()
        elif event.key == 'z' and self.all_boxes:  # Deshacer último cuadro con 'z'
            print("Deshaciendo último cuadro...")
            self.all_boxes.pop()
            self.draw_all_boxes()

    def draw_all_boxes(self):
        """Dibuja todos los cuadros en la imagen."""
        self.image_copy = Image.open(self.current_image_path).copy()
        draw = ImageDraw.Draw(self.image_copy)
        
        for i, box in enumerate(self.all_boxes, 1):
            x1, y1 = box[0]
            x2, y2 = box[1]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"Cuadro {i}", fill="red")
        
        plt.clf()
        plt.imshow(self.image_copy)
        plt.draw()

    def manual_adjustment(self, image_path):
        """Permite al usuario crear múltiples cuadros manualmente."""
        self.current_image_path = image_path
        self.image_copy = Image.open(image_path).copy()
        self.confirmed = False
        self.all_boxes = []
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(self.image_copy)
        
        print("\nInstrucciones:")
        print("- Haz clic dos veces para crear un cuadro")
        print("- Presiona 'r' para reiniciar todos los cuadros")
        print("- Presiona 'z' para deshacer el último cuadro")
        print("- Presiona 'enter' para confirmar todos los cuadros")
        
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()
        
        return self.all_boxes

    def process_image(self, image_path, save=False, input_dir=None):
        results = self.model(image_path)[0]
        detections = []
        
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Mostrar detecciones automáticas
        auto_boxes = []
        for box in results.boxes:
            if box.conf.item() >= self.confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                class_name = results.names[class_id] if input_dir is None else os.path.basename(input_dir)
                
                auto_boxes.append({
                    'coords': [x1, y1, x2, y2],
                    'class': class_name,
                    'conf': confidence
                })
                
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
                draw.text((x1, y1 - 10), f"{class_name} {confidence:.2%}", fill="blue")
        
        # Mostrar imagen con detecciones automáticas
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title("Detecciones automáticas (azul)")
        plt.show()
        
        # Preguntar si desea modificar las detecciones
        user_input = input("¿Deseas modificar las detecciones (s/n)? ").strip().lower()
        
        if user_input == 's':
            # Obtener cuadros modificados manualmente
            manual_boxes = self.manual_adjustment(image_path)
            
            # Convertir formato de cajas manuales
            for i, box in enumerate(manual_boxes):
                x1, y1 = box[0]
                x2, y2 = box[1]
                class_name = os.path.basename(input_dir) if input_dir else "objeto"
                
                detection = {
                    'label': class_name,
                    'confidence': 1.0,  # Confianza manual
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                }
                detections.append(detection)
        else:
            # Usar detecciones automáticas
            for box in auto_boxes:
                detection = {
                    'label': box['class'],
                    'confidence': box['conf'],
                    'bbox': {
                        'x1': float(box['coords'][0]),
                        'y1': float(box['coords'][1]),
                        'x2': float(box['coords'][2]),
                        'y2': float(box['coords'][3])
                    }
                }
                detections.append(detection)
        
        if save:
            # Guardar imagen final con todos los cuadros
            final_image = Image.open(image_path)
            draw = ImageDraw.Draw(final_image)
            
            for det in detections:
                bbox = det['bbox']
                draw.rectangle(
                    [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], 
                    outline="red", 
                    width=3
                )
                draw.text(
                    (bbox['x1'], bbox['y1'] - 10), 
                    f"{det['label']} {det['confidence']:.2%}", 
                    fill="red"
                )
            
            labeled_dir = os.path.join(os.path.dirname(image_path), 'labeled_images')
            Path(labeled_dir).mkdir(parents=True, exist_ok=True)
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            new_image_path = os.path.join(labeled_dir, f"{file_name}_labeled.jpg")
            final_image.save(new_image_path)
            print(f"Imagen guardada como {new_image_path}")
        
        return detections

    def process_directory(self, input_dir, output_file, save=False):
        """Procesa todas las imágenes en un directorio."""
        all_detections = []
        folder_name = os.path.basename(input_dir)
        
        for image_file in os.listdir(input_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, image_file)
                print(f"\nProcesando: {image_file}")
                
                detecciones = self.process_image(image_path, save=save, input_dir=input_dir)
                
                all_detections.append({
                    'image': image_file,
                    'label': folder_name,
                    'detections': detecciones
                })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)
        
        print(f"\nTodas las detecciones guardadas en {output_file}")

# Ejemplo de uso
if __name__ == "__main__":
    labeler = AutoLabeler()
    
    # Definir las rutas
    input_directory = "C:/Users/hkev2/OneDrive/Escritorio/packages/Modelo_Reconocimiento_Imagenes/model/gatos"

    # Obtener el nombre de la carpeta donde están las imágenes
    folder_name = os.path.basename(os.path.normpath(input_directory))

    # Crear el nombre del archivo JSON basándose en el nombre de la carpeta
    output_json_file = f"C:/Users/hkev2/OneDrive/Escritorio/packages/Modelo_Reconocimiento_Imagenes/model/coordenadas/detecciones_{folder_name}.json"

    # Procesar todas las imágenes en el directorio y guardar los datos de detección en un solo archivo JSON
    labeler.process_directory(input_directory, output_json_file, save=True)
