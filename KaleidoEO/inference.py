# importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# tesorflow
import tensorflow as tf
from tensorflow.keras.models import load_model

# loading the model
model = load_model('unet.h5')

# main code
def inferencing_model(input_img):
    image = Image.open(input_img)
    image.save('sample.jpg')
    input_image = cv2.imread('sample.jpg')
    #print(input_image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (512, 512))  

    predictions = model.predict(np.expand_dims(input_image, axis=0))
    
    predicted_classes = np.argmax(predictions[0], axis=-1)  
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax[0].imshow(input_image)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    # Display the class labels
    ax[1].imshow(predicted_classes)  
    ax[1].set_title('Predicted Class Labels')
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.jpg')
    plt.show()
    
    