# Import libraries :
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Load the Model :
my_model = keras.models.load_model("Models/model_47")

img_size = 40
classes = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']

# Prediction function :
def make_prediction(model, img):
  # Reshape the image for the model input
  img = img.reshape(1, img_size, img_size, 1)
  # Make Prediction
  arg_pred = np.argmax(my_model.predict(img))
  pred = classes[arg_pred]
  prediction = "The predicted class of the image : " + pred + "\n"
  return prediction


# App Name
st.set_page_config(page_title='Chess Classifier')

# Title
spaces = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
st.title(spaces + ' CHESS CLASSIFIER')
st.header('--- Simple App to classify Chess pieces ---')

image = Image.open('Images/chess.jpg')
st.image(image)

st.subheader('Description :')
st.write('''This is a Convolutional Neural Network model built with TensorFlow's Keras API, 
It gets as an input an image of a chess piece and it outputs the piece's name.
Notice that there is 6 classes to classify. ''')
st.markdown('***')

# Upload the image
uploaded_image = st.file_uploader('Choose a file')


butt = st.button('Predict')
st.markdown('***')

# Output the Result
if butt :
  if uploaded_image is not None:
    # To read file as bytes:
    image = uploaded_image.getvalue()
    # Decode the image
    image_decode = tf.image.decode_jpeg(image, channels=1)
    # Resize the image
    resize_image = tf.image.resize(image_decode, [img_size, img_size]) / 255.0
    img = np.array(resize_image).reshape(img_size, img_size)
    # Make the Prediction
    prediction = make_prediction(my_model, img)
    st.write('## Results :')
    # st.write('#### '+prediction)
    st.success(prediction)
    st.markdown('***')
    # Plot the image
    fig = plt.figure(figsize=(1,1))
    im = plt.imshow(img, cmap='gray')
    plt.axis('off')
    st.subheader('Preprocessed Image :')
    st.pyplot(fig)
  else :
    st.warning('Please upload an image first !')