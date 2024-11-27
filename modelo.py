from ultralytics import YOLO
import json
import shutil
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageEnhance

class YOLOTrainer:

    def __init__(self, data_dir):
        """
        Inicializa el entrenador de YOLO
        Args:
            data_dir: Directorio base donde se guardarán los datos de entrenamiento
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'
        
        # Crear estructura de directorios
        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.class_mapping = None

    def add_negative_examples(self, negative_dir):
        """
        Agrega ejemplos negativos (imágenes sin la clase objetivo) al conjunto de datos
        Args:
            negative_dir: Directorio con imágenes negativas
        """
        if not Path(negative_dir).exists():
            print(f"El directorio {negative_dir} no existe")
            return

        for img_file in Path(negative_dir).glob('*.[jp][pn][g]'):
            # Decidir split (70% train, 15% val, 15% test)
            rand = random.random()
            if rand < 0.7:
                split = 'train'
            elif rand < 0.85:
                split = 'val'
            else:
                split = 'test'
            
            # Copiar imagen
            shutil.copy(str(img_file), str(self.images_dir / split / img_file.name))
            
            # Crear archivo de etiquetas vacío
            label_file = self.labels_dir / split / f"{img_file.stem}.txt"
            label_file.touch()  # Crear archivo vacío

    def apply_augmentation(self, image):
        """
        Aplica aumentación de datos a una imagen
        """
        augmented_images = []
        
        # Rotaciones
        for angle in [90, 180, 270]:
            rotated = image.rotate(angle, expand=True)
            augmented_images.append(rotated)
        
        # Volteos
        augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        augmented_images.append(image.transpose(Image.FLIP_TOP_BOTTOM))
        
        # Ajustes de brillo y contraste
        enhancer = ImageEnhance.Brightness(image)
        augmented_images.append(enhancer.enhance(0.8))
        augmented_images.append(enhancer.enhance(1.2))
        
        enhancer = ImageEnhance.Contrast(image)
        augmented_images.append(enhancer.enhance(0.8))
        augmented_images.append(enhancer.enhance(1.2))
        
        return augmented_images
    

    def convert_json_to_yolo(self, json_file, class_mapping=None, negative_examples_dir=None):
        """
        Convierte las anotaciones JSON a formato YOLO
        Args:
            json_file: Archivo JSON con las detecciones
            class_mapping: Diccionario opcional para mapear nombres de clases a índices
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        # Si no se proporciona mapping, crear uno automáticamente
        if class_mapping is None:
            unique_classes = set()
            for item in annotations:
                unique_classes.add(item['label'])
            class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
        
        self.class_mapping = class_mapping

        # Guardar el mapping de clases
        with open(self.data_dir / 'classes.json', 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, indent=2)

        # Distribuir datos (70% train, 15% val, 15% test)
        random.shuffle(annotations)
        train_split = int(len(annotations) * 0.7)
        val_split = int(len(annotations) * 0.85)
        
        splits = {
            'train': annotations[:train_split],
            'val': annotations[train_split:val_split],
            'test': annotations[val_split:]
        }


        for split, split_annotations in splits.items():
            for annotation in split_annotations:
                image_file = annotation['image']
                image_path = self.data_dir / 'labeled_images' / f"{Path(image_file).stem}_labeled.jpg"
                
                if not image_path.exists():
                    print(f"Advertencia: No se encuentra la imagen {image_path}")
                    continue

                # Copiar imagen original
                shutil.copy(str(image_path), str(self.images_dir / split / image_file))
                
                # Crear archivo de etiquetas YOLO
                label_file = self.labels_dir / split / f"{Path(image_file).stem}.txt"
                
                with open(label_file, 'w') as f:
                    for detection in annotation['detections']:
                        bbox = detection['bbox']
                        class_id = class_mapping[detection['label']]
                        
                        # Convertir a formato YOLO
                        x1, y1 = bbox['x1'], bbox['y1']
                        x2, y2 = bbox['x2'], bbox['y2']
                        
                        width = x2 - x1
                        height = y2 - y1
                        x_center = x1 + width/2
                        y_center = y1 + height/2
                        
                        image = Image.open(image_path)
                        img_width, img_height = image.size

                        x_center /= img_width
                        y_center /= img_height
                        width /= img_width
                        height /= img_height
                        
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Agregar ejemplos negativos si se proporcionan
        if negative_examples_dir:
            self.add_negative_examples(negative_examples_dir)


    def create_yaml(self, name='custom_dataset'):
        """
        Crea el archivo YAML necesario para el entrenamiento
        """
        # Cargar mapping de clases
        with open(self.data_dir / 'classes.json', 'r') as f:
            class_mapping = json.load(f)
        
        yaml_content = {
            'path': str(self.data_dir),
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'names': {int(v): k for k, v in class_mapping.items()}
        }
        
        yaml_path = self.data_dir / f'{name}.yaml'
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(yaml_content, f, sort_keys=False)
        
        return yaml_path

    def train(self, epochs=100, imgsz=640, batch_size=8, output_dir=None):
        """
        Inicia el entrenamiento del modelo
        """
        # Crear archivo YAML
        yaml_path = self.create_yaml()

        # Si no se proporciona un directorio de salida, usar el predeterminado
        if output_dir is None:
            output_dir = self.images_dir / 'output'  # Directorio predeterminado
        output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar modelo
        self.model = YOLO('yolov8n.pt')
        
        # Entrenar con mejores hiperparámetros
        results = self.model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=20,  # Early stopping más temprano
            save=True,
            project=output_dir,
            optimizer='AdamW',  # Usar optimizador AdamW
            lr0=0.001,  # Learning rate inicial más bajo
            lrf=0.01,   # Learning rate final
            momentum=0.937,
            weight_decay=0.0005,  # Aumentar weight decay para reducir overfitting
            warmup_epochs=3,      # Época de calentamiento
            cos_lr=True,          # Learning rate con cosine annealing
            augment=True,         # Aumentación de datos
            mixup=0.1,            # Usar mixup
            copy_paste=0.1,       # Usar copy-paste augmentation
        )
        
        return results
    
    def load_model(self, weights_path):
        """
        Carga un modelo previamente entrenado
        Args:
            weights_path: Ruta al archivo de pesos del modelo
        """
        self.model = YOLO(weights_path)
        
        # Cargar el mapping de clases si existe
        classes_path = self.data_dir / 'classes.json'
        if classes_path.exists():
            with open(classes_path, 'r') as f:
                self.class_mapping = json.load(f)
    

    def predict(self, image_path, confidence=0.25, save=True):
        """
        Realiza predicciones en una imagen
        Args:
            image_path: Ruta a la imagen
            confidence: Umbral de confianza para las detecciones
            save: Si es True, guarda la imagen con las detecciones
        Returns:
            Lista de detecciones
        """
        if self.model is None:
            raise ValueError("No hay un modelo cargado. Usa train() o load_model() primero.")
        
        # Realizar predicción
        results = self.model.predict(image_path, conf=confidence)[0]
        
        # Procesar resultados
        detections = []
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Obtener el nombre de la clase
            class_name = results.names[class_id]
            
            # Solo incluir detecciones con alta confianza
            if conf >= confidence:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 10), f"{class_name} {conf:.2%}", fill="red")
                
                detection = {
                    'label': class_name,
                    'confidence': conf,
                    'bbox': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                }
                detections.append(detection)
        
        # Guardar imagen con detecciones
        if save:
            output_dir = Path(image_path).parent / 'predictions'
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f"{Path(image_path).stem}_pred.jpg"
            image.save(output_path)
            print(f"Predicción guardada en: {output_path}")
        
        return detections


    def save_model(self, output_dir=None, name='best_model.pt'):
        """
        Guarda el modelo entrenado en un directorio específico
        Args:
            output_dir: Directorio donde se guardará el modelo. 
                        Si no se especifica, se usa el directorio de datos
            name: Nombre del archivo del modelo
        """
        if self.model is None:
            raise ValueError("No hay un modelo para guardar. Primero entrena o carga un modelo.")
        
        # Si no se proporciona un directorio, usar el directorio de datos
        if output_dir is None:
            output_dir = self.data_dir / 'models'
        
        # Convertir a Path si no lo es
        output_dir = Path(output_dir)
        
        # Crear el directorio si no existe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ruta completa para guardar el modelo
        model_path = output_dir / name
        
        # Guardar el modelo
        self.model.save(str(model_path))
        
        print(f"Modelo guardado en: {model_path}")
        
        return model_path