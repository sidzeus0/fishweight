from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('fish_weight_predictor.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    # Ensure 'input.html' is in the 'templates' directory within your Flask app directory.
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from form
        form_data = request.form
        species = form_data['species']
        length1 = float(form_data['length1'])
        length2 = float(form_data['length2'])
        length3 = float(form_data['length3'])
        height = float(form_data['height'])
        width = float(form_data['width'])
        
        # Prepare data for prediction
        input_data = {
            'Species': [species],
            'Length1': [length1],
            'Length2': [length2],
            'Length3': [length3],
            'Height': [height],
            'Width': [width]
        }
        
        # Make prediction
        prediction = model.predict(pd.DataFrame.from_dict(input_data))[0]
        
        # Redirect to results page
        return render_template('results.html', prediction=round(prediction, 2))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
