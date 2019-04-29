from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import flask
import re

app=Flask(__name__)

# our ML model should be saved in the same folder as our main.py
loaded_model = load_model("model_malaria.h5")

def ValuePredictor(np_arr):   
    result = loaded_model.predict(np_arr)
    return result[0]

def string_to_int(s):
    integer = re.sub('[^0-9]','', s)
    return int(integer)

def one_hot_input(s, length):
    index = string_to_int(s)
    arr = [0 for i in range(length)]
    arr[index] = 1
    return arr

def flatten_arr(arr):
    flat_list = []
    for item in arr:
        if isinstance(item, (list)):
            for it in item:
                flat_list.append(it)
        else:
            flat_list.append(item)
    return flat_list


@app.route('/')
def home():
    return render_template("home.html")

# when the form press submit, it links it to the action /result which will be sent here:
@app.route('/result', methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        # from the request form, convert it to a dictionary saved as this variable
        features = request.form.to_list()

        prediction = features

        return render_template("result.html",prediction=prediction)

        '''
        # get the values and turn it into a list
        features=list(features.values())
        # flatten the list:
        features = flatten_arr(features)
        # reshape the list into a np array: (with unknown amount of columns)
        features = np.array(features).reshape(1,-1)
        print("Before sending to model", features) # flag
        # sending to our prediction model (which will reshape it as a numpy array and then return our result)
        result = ValuePredictor(features)
        print("result from model", result) # flag
        if int(result)==0:
            prediction='Statistics say you will make LESS than $50,000 USD per year.'
        else:
            prediction='Statistics say you will make MORE than $50,000 USD per year.'
        print(prediction) # flag
        # passing the string of our prediction to our template
        return render_template("result.html",prediction=prediction)
        '''
        

if __name__ == "__main__":
    app.run(debug=True)


