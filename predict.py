import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('models/trained_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('input_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the values from the form
        input_data = np.array([
            float(request.form['pm25']),
            float(request.form['pm10']),
            float(request.form['no']),
            float(request.form['so2']),
            float(request.form['co']),
            float(request.form['o3']),
            float(request.form['humidity']),
            float(request.form['temperature']),
            float(request.form['wind_speed']),
            float(request.form['pressure']),
            float(request.form['no2']),
            float(request.form['nh3']),
            float(request.form['ch4']),
            float(request.form['benzene']),
            float(request.form['tsp']),
            float(request.form['sulfate']),
            float(request.form['nitrate']),
            float(request.form['sodium']),
            float(request.form['potassium']),
            float(request.form['calcium']),
            float(request.form['magnesium']),
            float(request.form['ph']),
            float(request.form['acidity']),
            float(request.form['chloride']),
            float(request.form['sulfide']),
            float(request.form['carbonate']),
            float(request.form['nitrate_concentration']),
            float(request.form['ash_content']),
            float(request.form['density']),
            float(request.form['organic_carbon'])
        ])

        # Reshape and scale input data
        input_data = input_data.reshape(1, -1)
        print(f"Input Data: {input_data}")  # Debugging
        input_data_scaled = scaler.transform(input_data)
        print(f"Scaled Input Data: {input_data_scaled}")  # Debugging

        # Predict AQI
        prediction = model.predict(input_data_scaled)[0]
        print(f"Predicted AQI: {prediction}")  # Debugging

        # Map AQI to a category and recommendation
        if prediction <= 50:
            category = "Good"
            recommendation = "Air quality is satisfactory, and air pollution poses little or no risk."
        elif prediction <= 100:
            category = "Moderate"
            recommendation = "Air quality is acceptable. However, there may be a concern for sensitive individuals."
        elif prediction <= 150:
            category = "Unhealthy for Sensitive Groups"
            recommendation = "Sensitive groups should reduce outdoor activity."
        elif prediction <= 200:
            category = "Unhealthy"
            recommendation = "Everyone may begin to experience health effects; sensitive groups may face more serious effects."
        elif prediction <= 300:
            category = "Very Unhealthy"
            recommendation = "Health alert: everyone may experience serious health effects."
        else:
            category = "Hazardous"
            recommendation = "Health warnings of emergency conditions. The entire population is likely to be affected."

        # Pass values to the result page
        return render_template(
            'result.html',
            aqi=round(prediction, 2),
            category=category,
            recommendation=recommendation
        )

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {e}")
        return f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)
