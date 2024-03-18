from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from model import train_model

app = Flask(__name__)

fish_classification_model = os.path.join(os.path.dirname(__file__), 'fish_classifier.pkl')

if not os.path.isfile(fish_classification_model):
    train_model()

# Load the trained model
model = joblib.load('fish_classifier.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = request.form.get('weight')
    length1 = request.form.get('length1')
    length2 = request.form.get('length2')
    length3 = request.form.get('length3')
    height = request.form.get('height')
    width = request.form.get('width')
    
    data = pd.DataFrame({
        'Weight': [weight], 
        'Length1': [length1], 
        'Length2': [length2], 
        'Length3': [length3], 
        'Height': [height], 
        'Width': [width]
        })

    prediction = model.predict(data)

    return render_template('index.html',
                           input_weight=weight,
                           input_length1=length1,
                           input_length2=length2,
                           input_length3=length3,
                           input_height=height,
                           input_width=width,
                           predicted_fish=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)