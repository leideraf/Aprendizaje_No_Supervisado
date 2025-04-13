import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import matplotlib.pyplot as plt

class HandGestureUnsupervisedLearning:
    def __init__(self, input_shape=(128, 128, 1), model_path=None, headless=False):
        self.input_shape = input_shape
        self.model_path = model_path
        self.autoencoder = None
        self.threshold = None
        self.scaler = MinMaxScaler()
        self.headless = headless  # Modo sin interfaz gráfica
        
        # Crear directorio para guardar modelos y datos
        self.data_dir = "hand_gesture_data"
        self.model_dir = "hand_gesture_models"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Si se proporciona un modelo, cargarlo
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Construye un autoencoder convolucional"""
        # Codificador
        input_img = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # Decodificador
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Autoencoder completo
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        print("Modelo construido:")
        self.autoencoder.summary()
    
    def preprocess_frame(self, frame):
        """Preprocesa un fotograma para alimentar al modelo"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar
        resized = cv2.resize(gray, (self.input_shape[0], self.input_shape[1]))
        
        # Normalizar
        normalized = resized / 255.0
        
        # Añadir dimensión de canal y batch
        return normalized.reshape(1, self.input_shape[0], self.input_shape[1], 1)
    
    def extract_hand_roi(self, frame):
        """Extrae región de interés (ROI) de la mano"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar desenfoque para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilatar para cerrar bordes
        dilated = cv2.dilate(edges, None, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si hay contornos, tomar el más grande (asumiendo que es la mano)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Crear una máscara para la mano
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [max_contour], -1, 255, -1)
            
            # Aplicar la máscara
            hand_roi = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Obtener rectángulo delimitador
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Asegurarse de que el rectángulo tenga un tamaño mínimo
            if w > 20 and h > 20:
                # Extraer ROI y redimensionar
                roi = hand_roi[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (self.input_shape[0], self.input_shape[1]))
                
                # Dibujar rectángulo en el frame original
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                return roi_resized, frame, True
        
        return np.zeros((self.input_shape[0], self.input_shape[1])), frame, False
    
    def collect_training_data(self, num_frames=1000):
        """Recopila datos de entrenamiento de la cámara web"""
        print(f"Recopilando {num_frames} frames para entrenamiento...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara. Asegúrate de que esté conectada.")
            return None
        
        training_data = []
        count = 0
        
        try:
            while count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    print("Error al leer de la cámara web")
                    break
                
                # Voltear horizontalmente
                frame = cv2.flip(frame, 1)
                
                # Extraer ROI de la mano
                roi, annotated_frame, success = self.extract_hand_roi(frame)
                
                if success:
                    # Normalizar
                    normalized_roi = roi / 255.0
                    training_data.append(normalized_roi)
                    count += 1
                    
                    # Mostrar progreso sin interfaz gráfica
                    if count % 10 == 0:
                        print(f"Recopilando: {count}/{num_frames}")
                
                # Solo mostrar visualización si no estamos en modo headless
                if not self.headless:
                    try:
                        cv2.putText(annotated_frame, f"Recopilando: {count}/{num_frames}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow('Recopilando datos de entrenamiento', annotated_frame)
                        
                        # Salir si se presiona 'q'
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error en visualización (deshabilitando): {e}")
                        self.headless = True
        
        finally:
            # Liberar recursos
            cap.release()
            
            # Solo intentar cerrar ventanas si no estamos en modo headless
            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"Nota: No se pudieron cerrar ventanas de OpenCV: {e}")
                    self.headless = True
        
        # Convertir a numpy array con la forma correcta para el modelo
        if training_data:
            X_train = np.array(training_data).reshape(-1, self.input_shape[0], self.input_shape[1], 1)
            
            # Guardar datos de entrenamiento
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            np.save(f"{self.data_dir}/training_data_{timestamp}.npy", X_train)
            
            print(f"Se recopilaron {len(training_data)} frames. Datos guardados.")
            return X_train
        else:
            print("No se recopilaron datos suficientes.")
            return None
    
    def train(self, X_train=None, epochs=50, batch_size=32, validation_split=0.2):
        """Entrena el autoencoder con los datos recopilados"""
        if X_train is None:
            # Buscar el archivo de datos más reciente
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith("training_data_")]
            if not data_files:
                print("No hay datos de entrenamiento disponibles.")
                return False
            
            # Cargar el más reciente
            latest_data = max(data_files)
            X_train = np.load(f"{self.data_dir}/{latest_data}")
        
        print(f"Entrenando con {X_train.shape[0]} muestras...")
        
        # Entrenar el autoencoder
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split
        )
        
        # Calcular umbral de reconstrucción para detección de anomalías
        reconstructions = self.autoencoder.predict(X_train)
        mse = np.mean(np.square(X_train - reconstructions), axis=(1, 2, 3))
        
        # Establecer umbral como media + 2*desviación estándar
        self.threshold = np.mean(mse) + 2 * np.std(mse)
        print(f"Umbral de anomalía establecido en: {self.threshold}")
        
        # Guardar modelo
        self.save_model()
        
        # Graficar pérdida de entrenamiento si no estamos en modo headless
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
            plt.plot(history.history['val_loss'], label='Pérdida de validación')
            plt.title('Curva de pérdida del modelo')
            plt.ylabel('Pérdida')
            plt.xlabel('Época')
            plt.legend()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.model_dir}/loss_curve_{timestamp}.png")
            plt.close()
            print(f"Gráfico de pérdida guardado en {self.model_dir}/loss_curve_{timestamp}.png")
        except Exception as e:
            print(f"No se pudo crear el gráfico de pérdida: {e}")
        
        return True
    
    def save_model(self):
        """Guarda el modelo entrenado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.model_dir}/autoencoder_{timestamp}.h5"
        self.autoencoder.save(model_path)
        
        # Guardar también el umbral
        threshold_info = {
            "threshold": self.threshold,
            "input_shape": self.input_shape
        }
        np.save(f"{self.model_dir}/threshold_{timestamp}.npy", threshold_info)
        
        print(f"Modelo guardado en: {model_path}")
        self.model_path = model_path
        return model_path
    
    def load_model(self, model_path):
        """Carga un modelo guardado"""
        try:
            self.autoencoder = load_model(model_path)
            print(f"Modelo cargado desde: {model_path}")
            
            # Intentar cargar el umbral correspondiente
            threshold_path = model_path.replace("autoencoder", "threshold").replace(".h5", ".npy")
            if os.path.exists(threshold_path):
                threshold_info = np.load(threshold_path, allow_pickle=True).item()
                self.threshold = threshold_info["threshold"]
                self.input_shape = threshold_info["input_shape"]
                print(f"Umbral cargado: {self.threshold}")
            else:
                print("Archivo de umbral no encontrado. Se usará un umbral predeterminado.")
                self.threshold = 0.1
            
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self._build_model()
            return False
    
    def detect_anomalies(self, frame):
        """Detecta si un gesto es anómalo basado en el error de reconstrucción"""
        # Extraer ROI de la mano
        roi, annotated_frame, success = self.extract_hand_roi(frame)
        
        if not success:
            return annotated_frame, False, 0
        
        # Preprocesar
        normalized_roi = roi / 255.0
        input_img = normalized_roi.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        
        # Reconstruir con el autoencoder
        reconstructed = self.autoencoder.predict(input_img, verbose=0)
        
        # Calcular error de reconstrucción (MSE)
        mse = np.mean(np.square(input_img - reconstructed))
        
        # Determinar si es anómalo
        is_anomaly = mse > self.threshold
        
        # Visualizar resultado si no estamos en modo headless
        if not self.headless:
            anomaly_text = f"Anomalía: {'SÍ' if is_anomaly else 'NO'}"
            error_text = f"Error: {mse:.6f}/{self.threshold:.6f}"
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            
            cv2.putText(annotated_frame, anomaly_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated_frame, error_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated_frame, is_anomaly, mse
    
    def run_detection(self):
        """Ejecuta detección de anomalías en tiempo real"""
        if self.autoencoder is None:
            print("No hay modelo cargado. Entrene o cargue un modelo primero.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara. Asegúrate de que esté conectada.")
            return
        
        print("Ejecutando detección de anomalías. Presione 'q' para salir.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error al leer de la cámara web")
                    break
                
                # Voltear horizontalmente
                frame = cv2.flip(frame, 1)
                
                # Detectar anomalías
                result_frame, is_anomaly, mse = self.detect_anomalies(frame)
                
                # Mostrar resultado en consola en modo headless
                if self.headless and is_anomaly:
                    print(f"¡ANOMALÍA DETECTADA! Error: {mse:.6f}, Umbral: {self.threshold:.6f}")
                
                # Mostrar resultado visual solo si no estamos en modo headless
                if not self.headless:
                    try:
                        cv2.imshow('Detección de Anomalías en Gestos', result_frame)
                        
                        # Salir si se presiona 'q'
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error en visualización (deshabilitando): {e}")
                        self.headless = True
                        print("Ejecutando en modo headless. Presione Ctrl+C para salir.")
        
        except KeyboardInterrupt:
            print("Detección detenida por el usuario.")
        
        finally:
            # Liberar recursos
            cap.release()
            
            # Solo intentar cerrar ventanas si no estamos en modo headless
            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"Nota: No se pudieron cerrar ventanas de OpenCV: {e}")


def generate_synthetic_data(num_samples=500, shape=(128, 128)):
    """Genera datos sintéticos para entrenamiento cuando no hay cámara disponible"""
    print(f"Generando {num_samples} imágenes sintéticas para entrenamiento...")
    
    synthetic_data = []
    
    for i in range(num_samples):
        # Crear un fondo negro
        img = np.zeros(shape, dtype=np.uint8)
        
        # Generar formas aleatorias similares a manos
        num_shapes = np.random.randint(1, 5)
        
        for _ in range(num_shapes):
            # Coordenadas aleatorias para el centro
            center_x = np.random.randint(30, shape[0] - 30)
            center_y = np.random.randint(30, shape[1] - 30)
            
            # Tamaño aleatorio
            size = np.random.randint(10, 50)
            
            # Forma aleatoria (círculo o elipse)
            shape_type = np.random.choice(['circle', 'ellipse', 'rectangle'])
            
            if shape_type == 'circle':
                cv2.circle(img, (center_x, center_y), size, 255, -1)
            elif shape_type == 'ellipse':
                axes = (np.random.randint(10, 40), np.random.randint(20, 60))
                angle = np.random.randint(0, 180)
                cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, 255, -1)
            else:
                width = np.random.randint(10, 60)
                height = np.random.randint(30, 80)
                cv2.rectangle(img, (center_x, center_y), 
                             (center_x + width, center_y + height), 255, -1)
        
        # Normalizar
        normalized = img / 255.0
        synthetic_data.append(normalized)
        
        # Mostrar progreso
        if (i + 1) % 50 == 0:
            print(f"Generados {i + 1}/{num_samples} ejemplos")
    
    # Convertir a array y añadir dimensión de canal
    X_train = np.array(synthetic_data).reshape(-1, shape[0], shape[1], 1)
    
    # Guardar datos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = "hand_gesture_data"
    os.makedirs(data_dir, exist_ok=True)
    np.save(f"{data_dir}/synthetic_data_{timestamp}.npy", X_train)
    
    print(f"Datos sintéticos guardados en {data_dir}/synthetic_data_{timestamp}.npy")
    return X_train


def main():
    """Función principal para ejecutar el sistema"""
    # Configuración
    input_shape = (128, 128, 1)
    
    # Detectar si estamos en un entorno sin interfaz gráfica
    headless = False
    try:
        # Intentar crear una ventana de prueba
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Test")
    except Exception as e:
        print(f"Ejecutando en modo sin interfaz gráfica: {e}")
        headless = True
    
    # Inicializar el sistema
    system = HandGestureUnsupervisedLearning(input_shape=input_shape, headless=headless)
    
    # Menú interactivo
    while True:
        print("\n=== Sistema de Aprendizaje No Supervisado de Gestos de Mano ===")
        print("1. Recopilar datos de entrenamiento")
        print("2. Generar datos sintéticos (cuando no hay cámara)")
        print("3. Entrenar modelo")
        print("4. Ejecutar detección de anomalías")
        print("5. Cargar modelo existente")
        print("6. Salir")
        
        choice = input("Seleccione una opción (1-6): ")
        
        if choice == '1':
            num_frames = int(input("Número de frames a recopilar (recomendado 500-1000): "))
            system.collect_training_data(num_frames)
        
        elif choice == '2':
            num_samples = int(input("Número de ejemplos sintéticos a generar (recomendado 500-1000): "))
            synthetic_data = generate_synthetic_data(num_samples, shape=(input_shape[0], input_shape[1]))
            
            train_now = input("¿Entrenar modelo con datos sintéticos ahora? (s/n): ").lower()
            if train_now == 's':
                epochs = int(input("Número de épocas (recomendado 30-50): "))
                system.train(synthetic_data, epochs=epochs)
        
        elif choice == '3':
            epochs = int(input("Número de épocas (recomendado 30-50): "))
            data = None
            use_existing = input("¿Usar datos recopilados previamente? (s/n): ").lower()
            
            if use_existing != 's':
                use_synthetic = input("¿Generar datos sintéticos? (s/n): ").lower()
                
                if use_synthetic == 's':
                    num_samples = int(input("Número de ejemplos sintéticos a generar (recomendado 500-1000): "))
                    data = generate_synthetic_data(num_samples, shape=(input_shape[0], input_shape[1]))
                else:
                    num_frames = int(input("Número de frames a recopilar (recomendado 500-1000): "))
                    data = system.collect_training_data(num_frames)
                
                if data is None:
                    continue
            
            system.train(data, epochs=epochs)
        
        elif choice == '4':
            if system.autoencoder is None:
                print("No hay modelo cargado. Cargue o entrene un modelo primero.")
                continue
            
            system.run_detection()
        
        elif choice == '5':
            model_files = [f for f in os.listdir(system.model_dir) if f.endswith('.h5')]
            
            if not model_files:
                print("No hay modelos guardados disponibles.")
                continue
            
            print("Modelos disponibles:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")
            
            model_choice = int(input(f"Seleccione un modelo (1-{len(model_files)}): "))
            
            if 1 <= model_choice <= len(model_files):
                model_path = f"{system.model_dir}/{model_files[model_choice-1]}"
                system.load_model(model_path)
            else:
                print("Selección inválida.")
        
        elif choice == '6':
            print("Saliendo del programa...")
            break
        
        else:
            print("Opción inválida. Por favor intente de nuevo.")


if __name__ == "__main__":
    main()