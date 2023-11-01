import tensorflow as tf
import base64
import io
import shutil
import os
import uuid
import torch
import shutil
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify

if os.path.exists("runs/"):
    shutil.rmtree("runs/")

app = Flask(__name__)

# Carica il modello salvato
model1 = tf.keras.models.load_model('modello_cnn_ResNet50_87.h5')
model2 = tf.keras.models.load_model('modello_cnn_EfficentNet_88.h5')
model3 = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x.pt') 
model4 = YOLO(model=Path("yolov8x.pt")) 

# Dimensioni desiderate per l'input del modello
SIZE_X = 256
SIZE_Y = 256

def make_prediction(model, image):
    # Carica l'immagine utilizzando PIL

    ima = cv2.imread(image, 1)
    # Ridimensiona l'immagine alle dimensioni desiderate
    ima = cv2.resize(ima, (SIZE_X, SIZE_Y), interpolation = cv2.INTER_NEAREST)

    ima2 = np.array(ima)
    
    # Preprocessa l'immagine per adattarla al formato di input del modello
    img = tf.expand_dims(ima2, axis=0)

    # Effettua la previsione utilizzando il modello
    prediction = model.predict(img)
    # Determina il risultato della previsione
    result = 'Buona' if prediction <= 0.5 else 'Cattiva'

    return result

def make_prediction_yolov5(model, image_path):
    # Carica l'immagine
    im = Image.open(image_path)

    # Fai inferenza
    results = model(im)
    results.save()
    
    # Ottieni l'elenco dei file nella directory 'runs/detect/exp'
    yolov5_output_dir = 'runs/detect/exp'
    files_in_output_dir = os.listdir(yolov5_output_dir)
    

    # Inizializza una lista per memorizzare gli URL delle immagini processate
    processed_image_urls_yolov5 = []

    for file in files_in_output_dir:
        # Creiamo un nome di file univoco basato sulla data e sull'ora attuali
        unique_filename = 'yolov5_' + datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'

        # Percorso in cui vogliamo spostare l'immagine
        unique_filepath = os.path.join("static/images/", unique_filename)

        # Sposta l'immagine dalla cartella di output di YOLOv5 alla cartella static/images
        shutil.move(os.path.join(yolov5_output_dir, file), unique_filepath)

        processed_image_url_yolov5 = f"/static/images/{unique_filename}"  # Aggiorna l'URL
        processed_image_urls_yolov5.append(processed_image_url_yolov5)

    processed_image_url_yolov5 = f"/static/images/{unique_filename}"  # Aggiorna l'URL
    print("URLs delle immagini processate da YOLOv5:", processed_image_urls_yolov5)

    # Salviamo i risultati in un DataFrame
    df = results.pandas().xyxy[0]

    # Creiamo un nome di file univoco basato sulla data e sull'ora attuali
    file_name = 'results_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'

    # Stampa i risultati in un file di testo
    with open(file_name, 'w') as f:
        f.write(df.to_string())
      # Verifica se il DataFrame è vuoto (nessun oggetto rilevato)
    if df.empty:
        # Eliminiamo il file dopo averlo letto
        os.remove(file_name)
        return "Nessuna prugna trovata" 
    
    
    # Funzione per ottenere la classe con la confidenza massima da un file
    def get_class_with_highest_confidence(filename):
        class_counts = defaultdict(int)
        class_confidence_sums = defaultdict(float)

        with open(filename, 'r') as file:
            lines = file.readlines()

            # Consideriamo solo le righe che contengono dati
            data_lines = [line for line in lines if not line.startswith(' ')]

            for line in data_lines:
                # I valori sono separati da spazi, quindi splitto la riga
                values = line.split()
                
                # Otteniamo la colonna 'confidence'
                confidence = float(values[-3])

                # Otteniamo la colonna 'name'
                name = values[-1]

                class_counts[name] += 1
                class_confidence_sums[name] += confidence

        num_classes = len(class_counts)

        if num_classes == 1:
            return list(class_counts.keys())[0]
        elif num_classes == 2:
            return max(class_confidence_sums, key=class_confidence_sums.get)
        else:
            max_count = max(class_counts.values())
            max_classes = [cls for cls, count in class_counts.items() if count == max_count]

            if len(max_classes) == 1:
                return max_classes[0]
            else:
                max_class = max(max_classes, key=lambda cls: class_confidence_sums[cls])
                return max_class
    
    # Ottieniamo la classe con la confidenza massima
    predicted_class = get_class_with_highest_confidence(file_name)
        
    if predicted_class == "":
        return "Nessuna prugna rilevata"

    result_str = 'Buona' if predicted_class == 'good_prune' else 'Cattiva'

    # Eliminiamo il file dopo averlo letto
    os.remove(file_name)
    if os.path.exists("runs/"):
        shutil.rmtree("runs/")
    return result_str, processed_image_url_yolov5  

def make_prediction_yolov8(model, image_path):
    # Carica l'immagine
    im = Image.open(image_path)

    # Fai inferenza
    model(im, save_txt=True, save_conf=True, save=True)  

    # Ottengo il percorso del file di testo corrispondente
    base_name = os.path.basename(image_path)
    txt_file_name = os.path.splitext(base_name)[0] + '.txt'
    txt_file_path = os.path.join('runs/detect/predict/labels', txt_file_name)
    
    # Verifica se il file esiste
    if not os.path.exists(txt_file_path):
        return "Nessuna prugna rilevata"
    
    yolov8_output_path = f"runs/detect/predict/{os.path.basename(image_path)}"
    yolov8_new_path = f"static/images/yolov8_{os.path.basename(image_path)}"
    shutil.move(yolov8_output_path, yolov8_new_path)

    processed_image_url_yolov8 = f"/static/images/yolov8_{os.path.basename(image_path)}"  # Aggiorna l'URL

    # Funzione per ottenere la classe con la confidenza massima da un file
    def get_class_with_highest_confidence(filename):
        # Controllo se il file è vuoto
        if os.path.getsize(filename) == 0:  
            return None
        class_counts = defaultdict(int)
        class_confidence_sums = defaultdict(float)
        
        with open(filename, 'r') as file:
            lines = file.readlines()

            for line in lines:
                # I valori sono separati da spazi, quindi splitto la riga
                values = line.split()

                # Otteniamo la colonna 'confidence'
                confidence = float(values[-1])

                # Otteniamo la colonna 'name'
                name = int(values[0])

                class_counts[name] += 1
                class_confidence_sums[name] += confidence

        num_classes = len(class_counts)

        if num_classes == 1:
            return list(class_counts.keys())[0]
        elif num_classes == 2:
            return max(class_confidence_sums, key=class_confidence_sums.get)
        else:
            max_count = max(class_counts.values())
            max_classes = [cls for cls, count in class_counts.items() if count == max_count]

            if len(max_classes) == 1:
                return max_classes[0]
            else:
                max_class = max(max_classes, key=lambda cls: class_confidence_sums[cls])
                return max_class
    

    # Ottengo la classe con la confidenza massima
    try:
        predicted_class = get_class_with_highest_confidence(txt_file_path)
    except FileNotFoundError:
        return "Nessuna prugna rilevata"
    
    if predicted_class is None:
            return "Nessuna prugna rilevata"
    result_str = 'Buona' if predicted_class == 1 else 'Cattiva'

    # Eliminamo il file dopo averlo letto
    shutil.rmtree('runs')

    return result_str, processed_image_url_yolov8

# Funzione personalizzata per il filtro zip nel template
def custom_zip(*args):
    return zip(*args)

# Estendo l'ambiente Jinja2 associando la funzione 'custom_zip' al nome 'zip', 
# in modo da poter utilizzare la funzione zip direttamente nei templates di Jinja2.
app.jinja_env.globals['zip'] = custom_zip

# Quando un utente visita la homepage dell'applicazione, viene visualizzato il contenuto del template 'index.html'.
@app.route('/')
def index():
    return render_template('index.html')

# Funzione per eliminare le foto temporanee per le predizioni
def delete_old_images():
    folder = 'static/images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


@app.route('/predict', methods=['POST'])
def predict():
    # Elimino le foto predette in precedenza
    delete_old_images()
    # Carico le immagini scelte dall'utente 
    images = request.files.getlist('image')

    # Verifico se sono state caricate delle immagini
    if len(images) == 0 or not any(image.filename for image in images):
        error_message = "Nessuna immagine selezionata. Carica delle immagini e riprova."
        return render_template('index.html', predictions=None, error_message=error_message)

    predictions = []    # Array utilizzato per contenere le previsioni dei modelli CNN classiche
    image_urls = []
    processed_image_urls_yolov5 = []  # Nuova lista per gli URL delle immagini processate da YOLOv5
    processed_image_urls_yolov8 = []  # Nuova lista per gli URL delle immagini processate da YOLOv8
    yolov5_predictions = []
    yolov8_predictions = []
    for image in images:
        # Salvo l'immagine in una cartella temporanea
        image.save('static/images/' + image.filename)
        # Abbiniamo ciascun URL dell'immagine alle sue previsioni
        image_url = '/static/images/' + image.filename
        image_urls.append(image_url)

        # Effettuo le previsioni con i 4 modelli
        image_predictions = []
        
        image_predictions.append(make_prediction(model1, 'static/images/' + image.filename))
        image_predictions.append(make_prediction(model2, 'static/images/' + image.filename))
        result_yolov5 = make_prediction_yolov5(model3, 'static/images/' + image.filename)
        if isinstance(result_yolov5, tuple):
            prediction, yolov5_url = result_yolov5
            processed_image_urls_yolov5.append(yolov5_url)
        else:
            prediction = result_yolov5
            processed_image_urls_yolov5.append(image_url)  # Aggiungi l'URL dell'immagine originale

        yolov5_predictions.append(prediction)

        print("URLs delle immagini processate da YOLOv5:", processed_image_urls_yolov5)

        result_yolov8 = make_prediction_yolov8(model4, 'static/images/' + image.filename)
        if isinstance(result_yolov8, tuple):
            prediction, yolov8_url = result_yolov8
            processed_image_urls_yolov8.append(yolov8_url)
        else:
            prediction = result_yolov8
            processed_image_urls_yolov8.append(image_url)  # Aggiungi l'URL dell'immagine originale


        yolov8_predictions.append(prediction)  # Aggiungi la previsione di YOLOv8

        predictions.append(image_predictions)        

    # Passo gli URL delle immagini e i risultati delle previsioni al template
    return render_template('index.html', 
                        predictions=predictions, 
                        image_urls=image_urls, 
                        yolov5_predictions=yolov5_predictions, 
                        yolov8_predictions=yolov8_predictions,
                        processed_image_urls_yolov5=processed_image_urls_yolov5, 
                        processed_image_urls_yolov8=processed_image_urls_yolov8)


@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():

    # Elimino le foto predette
    delete_old_images()
    data = request.get_json()
    image_data = data['image']
    
    if "," in image_data:
        image_data = base64.b64decode(image_data.split(",")[1])
    else:
        # Gestisco il caso in cui non c'è una virgola nella stringa di dati dell'immagine
        image_data = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")
    # Genera un nome univoco per l'immagine utilizzando uuid
    image_name = f'webcam_image_{uuid.uuid4().hex}.png'
    
    # Salva l'immagine con il nuovo nome univoco
    image.save(f'static/images/{image_name}')
    image_url = f'/static/images/{image_name}'  # Aggiorna il percorso dell'immagine

    predictions = []
    predictions.append(make_prediction(model1, f'static/images/{image_name}'))
    predictions.append(make_prediction(model2, f'static/images/{image_name}'))
    yolov5_prediction = make_prediction_yolov5(model3, f'static/images/{image_name}')
    if isinstance(yolov5_prediction, tuple):
        yolov5_prediction, yolov5_url = yolov5_prediction
    else:
        yolov5_url = image_url  # Usa l'URL dell'immagine originale

    yolov8_prediction = make_prediction_yolov8(model4, f'static/images/{image_name}')
    if isinstance(yolov8_prediction, tuple):
        yolov8_prediction, yolov8_url = yolov8_prediction
    else:
        yolov8_url = image_url  # Usa l'URL dell'immagine originale

    predictions.append(yolov5_prediction)
    predictions.append(yolov8_prediction)

    print(jsonify({'predictions': predictions, 'image_url': image_url, 'yolov5_url': yolov5_url, 'yolov8_url': yolov8_url}))

    return jsonify({
        'predictions': predictions, 
        'image_url': image_url, 
        'yolov5_url': yolov5_url, 
        'yolov8_url': yolov8_url
    })

if __name__ == '__main__':
    app.run(debug=True)     #host= '192.168...' nel caso in cui voglia estendere l'applicazione per l'uso di utenti collegati
                            #su una rete internet diversa da quella locale. 
