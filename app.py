from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
import joblib  # For loading the saved Random Forest model
import os

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('image_classifier_model.pkl')

# Function to preprocess and predict using the model
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))  # Resize the image to match training size
    img_array = image.img_to_array(img)  # Convert image to array
    img_flatten = img_array.flatten()  # Flatten the 3D image array into a 1D array
    img_flatten = np.expand_dims(img_flatten, axis=0)  # Add batch dimension

    prediction = model.predict(img_flatten)  # Get model's prediction
    return prediction[0]  # Return the prediction directly

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)  # Load the homepage

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  # Get uploaded image file
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)  # Save the uploaded file to the uploads folder

        # Make a prediction using the trained model
        prediction = model_predict(file_path, model)

        # Convert the numerical prediction to a label
        predicted_label = 'Dog' if prediction == 1 else 'Cat'  # Example logic

        return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
