import xgboost as xgb
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from waitress import serve

app = Flask(__name__)

# Load the pre-trained model
model = xgb.Booster()
model.load_model('model/xgboost/employee_attrition_model.json')

# Load the scaler
scaler = joblib.load('model/xgboost/scaler.pkl')

# Ensure the same feature columns used in training
expected_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales', 'EducationField_Life Sciences',
    'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
    'Gender_Male', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'OverTime_Yes'
]


@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the 'templates' folder


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the request
        form_data = request.form.to_dict()

        # Define all possible job roles
        job_roles = [
            'JobRole_Human Resources',
            'JobRole_Laboratory Technician',
            'JobRole_Manager',
            'JobRole_Manufacturing Director',
            'JobRole_Research Director',
            'JobRole_Research Scientist',
            'JobRole_Sales Executive',
            'JobRole_Sales Representative'
        ]

        # Handle education fields
        education_fields = [
            'EducationField_Life Sciences',
            'EducationField_Marketing',
            'EducationField_Medical',
            'EducationField_Technical Degree',
            'EducationField_Other'
        ]

        # Handle marital status fields
        marital_status_fields = [
            'MaritalStatus_Married',
            'MaritalStatus_Single'
        ]

        # Handle department fields
        department_fields = [
            'Department_Research & Development',
            'Department_Sales'
        ]

        # Handle travel frequency
        travel_frequency_fields = [
            'BusinessTravel_Travel_Frequently',
            'BusinessTravel_Travel_Rarely'
        ]

        # Initialize to 0
        for field in job_roles + education_fields + marital_status_fields + department_fields + travel_frequency_fields:
            form_data[field] = 0

        selected_fields = [
            ('JobRole', job_roles),
            ('EducationField', education_fields),
            ('MaritalStatus', marital_status_fields),
            ('Department', department_fields),
            ('Business_Travels', travel_frequency_fields)
        ]

        for field_key, field_list in selected_fields:
            selected_value = form_data.get(field_key)
            if selected_value in field_list:
                form_data[selected_value] = 1

        for key in ['JobRole', 'EducationField', 'MaritalStatus', 'Department', 'Business_Travels']:
            form_data.pop(key, None)

        # Convert form values to integers where necessary
        for key in form_data:
            form_data[key] = int(form_data[key]) if form_data[key] in ['0', '1'] else form_data[key]

        # Print the form_data to debug and see the output
        for key, value in form_data.items():
            print(f"{key}: {value}")

        # Ensure the form data has all the expected columns and add missing ones with default value of 0
        missing_cols = set(expected_columns) - set(form_data.keys())
        for col in missing_cols:
            form_data[col] = 0  # Default value for missing columns

        # Convert form data to DataFrame
        input_df = pd.DataFrame([form_data])

        # Reorder columns to match the training dataset order
        input_df = input_df[expected_columns]

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        input_dmatrix = xgb.DMatrix(input_scaled)
        prediction = model.predict(input_dmatrix)
        probability = prediction[0]

        # Map result to 'Yes' or 'No'
        result = "Yes" if probability >= 0.5 else "No"
        confidence = probability * 100

        return render_template('results.html', prediction=result, confidence=confidence)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    port = 5000
    print(f"Server is running on http://127.0.0.1:{port}")
    serve(app, host="0.0.0.0", port=port)
