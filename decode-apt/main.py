from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import cv2
import os

img_h,img_w=(224,224)
batch_size=32

f=r"train\clear_dataset\1.jpg"
img=image.load_img(f)
im=cv2.imread(f).shape
print(im)
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory('train/',target_size=(200,200),batch_size=3,class_mode='binary')
validation_dataset=train.flow_from_directory('validation/',target_size=(200,200),batch_size=3,class_mode='binary')
p=train_dataset.class_indices
print(p)

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                tf.keras.layers.MaxPool2D(2,2),
                                
                                tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                tf.keras.layers.MaxPool2D(2,2),
                                
                                tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                tf.keras.layers.MaxPool2D(2,2),
                                
                                tf.keras.layers.Flatten(),
                                
                                tf.keras.layers.Dense(512,activation='relu'),
                                
                                tf.keras.layers.Dense(1,activation='sigmoid')
                                ])
print(model)

model.compile(loss='binary_crossentropy',optimizer=RMSprop(learning_rate=0.001),
                        metrics=['accuracy'])

model_fit=model.fit(train_dataset,steps_per_epoch=3,
epochs=30,validation_data=validation_dataset)
print(model_fit)

input_path='imput_images'

for i in os.listdir(input_path):
    inp=image.load_img(input_path+'//'+i,target_size=(200,200))
    plt.imshow(inp)
    plt.show()

    X=image.img_to_array(inp)
    X=np.expand_dims(X,axis=0)
    images=np.vstack([X])
    val=model.predict(images)
    print(val)
    if val==0:
        print("clear_data")
    else:
        print("Storm_Image")
