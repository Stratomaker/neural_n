import io

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from ipywidgets import Image, HBox
from ipywebrtc import CameraStream, ImageRecorder
import PIL.Image
import PIL.ImageFilter

import numpy as np

camera = CameraStream(constraints=
                      {'facing_mode': 'user',
                       'audio': False,
                       'video': { 'width': 640, 'height': 480 }
                       })
camera

image_recorder = ImageRecorder(stream=camera)
image_recorder

im = PIL.Image.open(io.BytesIO(image_recorder.image.value))

def transform_image(image, target_size=(224,224), border=0):
    
    half_size = min(image.width,image.height) // 2 - border
    cropped = image.crop((
        image.width // 2 - half_size,
        image.height // 2 - half_size,
        image.width // 2 + half_size,
        image.height // 2 + half_size))
    return cropped.resize(target_size)

    from matplotlib import pyplot as plt
from keras.applications.mobilenet import preprocess_input


transformed_image = transform_image(im, border=50)
im_array = np.asarray(transformed_image, dtype='int')


if (im_array.shape[2] == 4):
    im_array = np.delete(im_array,3,2) 
print(im_array.shape)


im_array = preprocess_input(im_array)

plt.matshow(im_array[:,:,0], interpolation='nearest', cmap='Reds')
plt.matshow(im_array[:,:,1], interpolation='nearest', cmap='Greens')
plt.matshow(im_array[:,:,2], interpolation='nearest', cmap='Blues')

mobilenet = applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling = 'avg')
classification_model = load_model('./models/firstTry/mobilenet_128_model.h5')

def classify(image):

    transformed_image = transform_image(image, border=50)
    im_array = np.asarray(transformed_image, dtype='int')
    if (im_array.shape[2] == 4):
        im_array = np.delete(im_array,3,2) 
    im_array = preprocess_input(im_array)

    feature_vector = mobilenet.predict(np.array([im_array]))
    prediction = classification_model.predict([feature_vector]) #1 - Dog / 2 - Cat

    rounded_prediction = prediction[0].round() 
    is_dog = False
    if rounded_prediction == 1:
        animal_type = 'Dog'
        is_dog = True
    else:
        animal_type = 'Cat'
    return f'Identified Animal as {animal_type} with probability {prediction[0] if is_dog else 1-prediction[0]}'

    