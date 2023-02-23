import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import model_from_json

# load json
json_file = open('model-ferams/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# Create model
model = model_from_json(loaded_model_json)
# Load weights into model
model.load_weights("model-ferams/model_weights.h5")

print(model)

emotions_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

angry = cv2.imread('emojis/angry.png', -1)
disgust = cv2.imread('emojis/disgust.png', -1)
fear = cv2.imread('emojis/fear.png', -1)
happy = cv2.imread('emojis/happy.png', -1)
neutral = cv2.imread('emojis/neutral.png', -1)
sad = cv2.imread('emojis/sad.png', -1)
surprise = cv2.imread('emojis/surprise.png', -1)

haar_file = 'haarcascade_frontalface_alt.xml'
video = cv2.VideoCapture(0)


def prediction(image, model):
    """Réalise une prédiction d'émotion sur une image sous forme d'un array 
    de dimensions (48,48,1)"""
    image = tf.expand_dims(image, 0)
    return emotions_map[np.argmax(model.predict(image))]

def image_over(emoji, frame, faces):
    """Remplace le visage capturé dans une frame par un emoji"""
    # Copie de la frame
    result_image = frame.copy()
    # On parcourt les visages
    for (x, y, w, h) in faces:
        # Creation de l'emoji de la bonne taille
        this_detection_over = cv2.resize(emoji, (h, w))
        # Conversion des pixels transparents pour match l'image du dessous
        add_x, add_y = 0, 0
        for pixel_row in this_detection_over:
            add_x = 0
            add_y += 1
            for pixel in pixel_row:
                add_x += 1
                if pixel[3] == 0:
                    this_detection_over[add_y - 1, add_x - 1] = result_image[y + add_y - 1, x + add_x - 1]

        # Ajout de l'emoji sur le visage
        result_image[y:y+h, x:x+w] = this_detection_over

    return result_image


while(True): 
    _, frame = video.read()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        
        x, y, w, h = [int(i) for i in (x, y, w, h)]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        face = frame[y:y+h, x:x+w]
        face_g = cv2.cvtColor(face ,cv2.COLOR_BGR2GRAY)
        face_g = cv2.resize(face_g, (48,48), interpolation=cv2.INTER_AREA)
        face_g = face_g.reshape(48, 48, 1)/255.0
        emotion = prediction(model=model, image=face_g)

        match emotion:
            case "angry":
                frame = image_over(angry, frame, faces)
            case "disgust":
                frame = image_over(disgust, frame, faces)
            case "fear":
                frame = image_over(fear, frame, faces)
            case "happy":
                frame = image_over(happy, frame, faces)
            case "neutral":
                frame = image_over(neutral, frame, faces)
            case "sad":
                frame = image_over(sad, frame, faces)
            case "surprise":
                frame = image_over(surprise, frame, faces)

        cv2.imshow('video', frame)
        key=cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()