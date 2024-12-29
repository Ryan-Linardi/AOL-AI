# Basic dependencies
import sys
import os
import base64
import io

if sys.version_info.major == 3 and sys.version_info.minor == 12:
    print("This script requires Python 3.12")
    sys.exit(1)

import downloader

# Run the script to download model from Drive
downloader.download_model()

try:
    try:
        # Using PyTorch backend
        os.environ["KERAS_BACKEND"] = "torch"
        import keras
    except:
        # Fallback to TensorFlow
        import keras

    import numpy as np
    from PIL import Image
    import yaml

    # Using Flask to handle HTTP requests and to create a server
    from flask import Flask, Response, request, render_template, send_from_directory
except Exception as e:
    # Exit the app if there are any missing dependency
    print(e)
    print("Some dependencies are missing")
    print("Please check `requirements.txt`")
    print("Try running the following command")
    print("python -m pip install -r requirements.txt")
    print("To automatically install all dependencies")
    exit()

# Keys to map prediction output values to class names
keys = [
    "Brokoli",
    "Cumi",
    "Daging Ayam",
    "Daging Sapi",
    "Ikan",
    "Jagung",
    "Kentang",
    "Tahu",
    "Telur",
    "Wortel",
]

nutrition = None
with open("data/nutrition.yaml") as data:
    nutrition = yaml.safe_load(data)

# Loading the keras model
model = keras.models.load_model("models/Model_6V3.keras")

# Initiate the Flask server
app = Flask(__name__)


# The main page
@app.route("/")
def home():
    return render_template("main.html")


# Static assets like stylesheets
@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory("assets", path)


# Handle request to predict
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the file from the POST request
        file = request.files["file"]
        # file.save('debug/debug.jpeg') # For debug

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
        tuples_sorted = sorted(tuples, key=lambda item: item[1], reverse=True)

        res_top = tuples_sorted.pop(0)
        res_other = tuples_sorted

        # Display the inputted image back on the page
        buffer = io.BytesIO()
        image_input.save(buffer, format="webp")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_data = f"data:image/webp;base64,{image_base64}"

        # Return the HTML page to the browser
        # Prediction result is passed on to be rendered by Flask
        return render_template(
            "main.html",  # Template of the HTML page
            image=image_data,
            response_top=res_top,  # The top result of the prediction
            response_other=res_other,  # The other prediction results
            nutrition=nutrition[res_top[0]],
        )

    except Exception as e:
        # Return a HTTP 400 code for errors
        return Response(status=400)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
