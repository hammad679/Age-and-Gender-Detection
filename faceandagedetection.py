from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import glob
import random
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import pickle
import cvlib as cv


from sklearn.model_selection import train_test_split

image_file=[f for f in glob.glob(r'gender_dataset_face'+"/**/*" ,recursive=True) if not os.path.isdir(f)]
image_file

random.shuffle(image_file)

image_file

data=[]
data_label=[]

for img in image_file:
  image=cv2.imread(img)
  image=cv2.resize(image,(96,96))
  image=img_to_array(image)
  data.append(image)
  label=img.split(os.path.sep)[-2]
  if label=='woman':
    label=1
  else:
    label=0
  data_label.append([label])

data=np.array(data,dtype='float')/255.0
data_label=np.array(data_label)

data

X_train,X_test,Y_train,Y_test=train_test_split(data,data_label,test_size=0.2,random_state=42)

X_train

data_aug=ImageDataGenerator(rotation_range=25,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

Y_train=to_categorical(Y_train,num_classes=2)
Y_test=to_categorical(Y_test,num_classes=2)

classes=2
batch_size = 64
epochs=100

model=Sequential([
    layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(96,96,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(3,3)),
    layers.Dropout(0.25),
    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(3,3)),
    layers.Dropout(0.25),
    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(3,3)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(1024,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(classes,activation='softmax'),
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit_generator(data_aug.flow(X_train,Y_train,batch_size=batch_size),
          validation_data=(X_test,Y_test),
          steps_per_epoch=len(X_train)//batch_size,
          epochs=epochs,
          verbose=1)

model.save('model.h5')

model = keras.models.load_model('my_model.h5')

pickle.dump(model,open('AgeAndGenderDetectionmodel.pkl','wb'))

predictions=model.predict(X_test)

np.argmax(predictions[2])

classes=['man','woman']

filename='woman.jpeg'
im = cv2.imread(filename)

faces,confidence=cv.detect_face(im)

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

ageNet=cv2.dnn.readNet(ageModel,ageProto)

# by Uploading the picture.
for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)
    face_crop=np.copy(im[startY:endY,startX:endX])
    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
    blob=cv2.dnn.blobFromImage(face_crop, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    face_crop = cv2.resize(face_crop, (96,96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)
    conf = model.predict(face_crop)[0]
    ageNet.setInput(blob)
    agePred=ageNet.forward()
    age=ageList[agePred[0].argmax()]
    idx = np.argmax(conf)
    label = classes[idx]
    print(age)
    label = "{}: {} {:.2f}%".format(label,age, conf[idx] * 100)
    Y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(im, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,0,0), 2,cv2.LINE_AA)
plt.figure(figsize=(5, 5))
plt.imshow(im)

# for Live video
# camera = cv2.VideoCapture(0)
# while camera.isOpened():
#     status, frame = camera.read()
#     faces, confidence = cv.detect_face(im)
#     for face in faces:
#         (startX, startY) = face[0], face[1]
#         (endX, endY) = face[2], face[3]
#         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#         face_crop = np.copy(frame[startY:endY, startX:endX])
#         if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
#             continue
#         blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#         face_crop = cv2.resize(face_crop, (96, 96))
#         face_crop = face_crop.astype("float") / 255.0
#         face_crop = img_to_array(face_crop)
#         face_crop = np.expand_dims(face_crop, axis=0)
#         conf = model.predict(face_crop)[0]
#         ageNet.setInput(blob)
#         agePred = ageNet.forward()
#         age = ageList[agePred[0].argmax()]
#         idx = np.argmax(conf)
#         label = classes[idx]
#         print(age)
#         label = "{}: {} {:.2f}%".format(label, age, conf[idx] * 100)
#         Y = startY - 10 if startY - 10 > 10 else startY + 10
#         cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8, (255, 0, 0), 2, cv2.LINE_AA)
#       cv2.imshow("gender detection", frame)
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#           break
# webcam.release()
# cv2.destroyAllWindows()



