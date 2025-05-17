from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("churn_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['Age'])
    tenure = float(request.form['Tenure'])
    usage_frequency = float(request.form['Usage Frequency'])
    support_calls = float(request.form['Support Calls'])
    payment_delay = float(request.form['Payment Delay'])
    total_spend = float(request.form['Total Spend'])
    last_interaction = float(request.form['Last Interaction'])

    gender = request.form['Gender']
    subscription_type = request.form['Subscription Type']
    contract_length = request.form['Contract Length']

    gender_male = 1 if gender == 'Male' else 0


    subscription_premium = 1 if subscription_type == 'Premium' else 0
    subscription_standard = 1 if subscription_type == 'Standard' else 0

    contract_monthly = 1 if contract_length == 'Monthly' else 0
    contract_quarterly = 1 if contract_length == 'Quarterly' else 0

    input_features = [age, tenure, usage_frequency, support_calls, payment_delay,
                      total_spend, last_interaction,
                      gender_male,
                      subscription_premium, subscription_standard,
                      contract_monthly, contract_quarterly]

    prediction = model.predict([input_features])[0]
    result = "Yes" if prediction == 1 else "No"

    return render_template('index.html', prediction_text=f"Churn: {result}")

if __name__ == '__main__':
    app.run(debug=True)