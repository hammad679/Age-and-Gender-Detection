import pickle
import cv2
import numpy as np
import cvlib as cv
from fastapi import FastAPI, File, UploadFile
from keras.utils import img_to_array
from tensorflow import keras

app = FastAPI()

classes=['Man', 'Woman']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# model = pickle.load(open('AgeAndGenderDetectionmodel.pkl','rb'))

model = keras.models.load_model('my_model.h5')

def prepareData(im):
    faces, confidence = cv.detect_face(im)
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    for face in faces:
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]
        cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face_crop = np.copy(im[startY:endY, startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = model.predict(face_crop)[0]
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        idx = np.argmax(conf)
        label = classes[idx]
        print(age)
        label = "Gender : {}  Age: {} Confidence: {:.2f}%".format(label, age, conf[idx] * 100)
    return label

@app.post('/predict/')
async def get_input(file: UploadFile = File(...)):
    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR).astype(np.float32)
    prediction=prepareData(img)
    return prediction


