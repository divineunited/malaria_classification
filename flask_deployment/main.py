from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
# import flask
import re
from keras.preprocessing import image
from skimage import transform

app=Flask(__name__)

# our ML model should be saved in the same folder as our main.py
loaded_model = load_model("model_malaria.h5")
# prevent tensorflow error caused by running the predict function in a separate thread (which happens when routing).
loaded_model._make_predict_function()
graph = tf.get_default_graph()

def ValuePredictor(np_arr):   
    global graph
    # opening up the graph above and managing the resources "with"
    with graph.as_default():
        result = loaded_model.predict(np_arr)
    return result[0]

def image_preprocess(img):
    # setting the new shape of image to what we trained our model on
    new_shape = (50, 50, 3)

    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # resize the image (must be done after it has turned into a np array):
    image_array = transform.resize(image_array, new_shape, anti_aliasing=True)

    # scaling the image data to fall between 0-1 since images have 255 brightness values:
    image_array /= 255

    # The input shape we have defined is the shape of a single sample. The model itself expects some array of samples as input (even if its an array of length 1). Your output really should be 4-d, with the 1st dimension to enumerate the samples. i.e. for a single image you should return a shape of (1, 50, 50, 3).
    # add an extra dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.route('/')
def home():
    return render_template("home.html")

# when the form press submit, it links it to the action /result which will be sent here:
@app.route('/result', methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        # from the request form, convert it to a dictionary saved as this variable
        img = request.files['pic']

        # pre-process the image that was posted to us
        img_arr = image_preprocess(img)

        # sending to our prediction model
        result = ValuePredictor(img_arr)
        print("result from model", result) # flag

        # most likely class index:
        result = int(np.argmax(result))

        print("result actual", result) # flag

        if result==0:
            prediction='This cell is likely NOT INFECTED with the Malaria Parasite.'
        else:
            prediction='This cell is likely INFECTED with the Malaria Parasite.'

        print(prediction) # flag
        # passing the string of our prediction to our template
        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)


