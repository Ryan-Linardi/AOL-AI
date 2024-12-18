# Importing necessary dependencies
import os
import logging

try:
    # Using PyTorch backend for Keras instead of TensorFlow
    os.environ['KERAS_BACKEND'] = 'torch'
    import keras
    
    # Using the built-in numpy from Keras 3
    import numpy as np
    
    # Using Python Imaging Library to manipulate the inputted images
    from PIL import Image
    
    # Using Flask to handle HTTP requests and to create a server
    from flask import Flask, request, render_template, Response
except Exception as e: 
    # Exit the app if there are any missing dependency
    logging.critical(e)
    logging.critical("Some dependencies are missing")
    logging.critical("Please check `requirements.txt`")
    logging.critical("Try running the following command")
    logging.critical("python -m pip install -r requirements.txt")
    logging.critical("To automatically install all dependencies")
    exit()

# Setting up logging for the app
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('debug/app.log')
formatter = logging.Formatter("{levelname}: {msg}", style='{')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Keys to map prediction output values to class names
keys = [
    'Broccoli',
    'Carrot',
    'Cheese',
    'Corn',
    'Egg',
    'Noodles',
    'Potato',
    'Chicken',
    'Beef',
    'Fish',
    'Chili',
    'Tempe',
    'Tahu',
]

# Loading the keras model
model = keras.models.load_model('models/best_model.keras')

# Initiate the Flask server
app = Flask(__name__)

# The main page of the app
@app.route("/")
def home():
    return render_template('main.html')

# Handle request to load stylesheets
@app.route("/assets/style/<filename>")
def stylesheet(filename: str):
    try:
        logger.debug("GET request for CSS file <{}>".format(filename))
        file = open('assets/style/' + filename)
        file_content = file.read()
        return file_content
    except Exception as err:
        logger.error(err)
        return Response(status=404)

# Handle request to predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the POST request
        file = request.files['file']
        file.save('debug/debug.jpeg') # For debug

        # Open the image file
        image_input = Image.open(file)
        # Resize the image to the model imput of 224 x 224
        image_resized = image_input.resize((224, 224))
        # Convert the image into a pixel brightness array
        image_array = np.array(image_resized)
        # Change the range of pixel from 0~255 into 0~1
        image_array = image_array / 255.0
        
        image_array = np.expand_dims(image_array, axis=0)

        # Uses the model to predict
        prediction = model.predict(image_array)
        # Conversion numpy into python float
        prediction = [float(x) for x in prediction[0]]

        # Pair up the prediction result with class names
        tuples = zip(keys, prediction)
        # Sort the result from highest to lowest confidence
        tuples_sorted = sorted(tuples,
                               key=lambda item: item[1], reverse=True)
        
        # Return the HTML page to the browser
        # Prediction result is passed on to be rendered by Flask
        return render_template(
                'main.html', # Template of the HTML page
                response_top=tuples_sorted.pop(0), # The top result of the prediction
                response_other=tuples_sorted # The other prediction results
            )

    except Exception as e:
        # Return a HTTP 400 code for errors
        return Response(status=400)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
