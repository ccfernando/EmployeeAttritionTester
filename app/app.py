import xgboost as xgb
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


model = xgb.Booster()
model.load_model('model/employee_attrition_model.json')

# Load the scaler
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the request
        form_data = request.form.to_dict()

        # Convert form values to correct data types (integers for most columns)
        for key in form_data:
            if key in ['BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
                       'Department_Sales', 'EducationField_Life Sciences', 'EducationField_Marketing',
                       'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
                       'Gender_Male', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
                       'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
                       'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
                       'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes']:
                form_data[key] = int(form_data[key])  # Convert boolean-like fields to integer (0 or 1)
            else:
                form_data[key] = int(form_data[key])  # Convert to int for other columns (e.g., Age, MonthlyIncome)

        # Convert form data to DataFrame (since model expects DataFrame)
        input_df = pd.DataFrame([form_data])

        # Scale the features using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Convert the scaled input to DMatrix for XGBoost
        input_dmatrix = xgb.DMatrix(input_scaled)

        # Make prediction using the loaded model
        prediction = model.predict(input_dmatrix)  # Get model's predicted probabilities (for binary classification)
        probability = prediction[0]  # Assuming it returns an array-like object, get the probability

        # Map prediction to readable output
        result = "Yes" if probability >= 0.5 else "No"  # If probability >= 0.5, classify as 'Yes'
        confidence = probability * 100  # Confidence in percentage

        # Return JSON response with the prediction result and confidence percentage
        # return jsonify({
        #     'Attrition Prediction': result,
        #     'Confidence Percentage': f"{confidence:.2f}%",  # Format the confidence to two decimal places,
        #
        # })

        return render_template('results.html',prediction=result,confidence=confidence)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
