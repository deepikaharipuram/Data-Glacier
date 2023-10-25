from flask import Flask, render_template, request
import numpy as np
import joblib
import re

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        sepal_length = float(re.sub('[^0-9.]', '', request.form['sepal_length']))
        sepal_width = float(re.sub('[^0-9.]', '', request.form['sepal_width']))
        petal_length = float(re.sub('[^0-9.]', '', request.form['petal_length']))
        petal_width = float(re.sub('[^0-9.]', '', request.form['petal_width']))

        # Create a NumPy array with the input values
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make the prediction
        prediction = model.predict(data)

        # Map prediction to Iris species names
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        predicted_species = species_names[prediction[0]]

        return render_template('index.html', prediction_text=f'The predicted Iris species is: {predicted_species}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
