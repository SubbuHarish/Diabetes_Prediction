from flask import Flask,request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input features from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make a prediction using the model
    prediction = model.predict(final_features)
    output = prediction[0]

    # Render the prediction result
    if output == 0:
        return render_template('index.html', prediction_text='You are  likely to have diabetes.')
    else:
        return render_template('index.html', prediction_text='You are  not likely to have diabetes.')

if __name__ == "__main__":
    app.run(debug=True)