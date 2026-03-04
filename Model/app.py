from flask import Flask, render_template, request
import numpy as np
import pickle

# Load saved model and scaler
model = pickle.load(open("dtc_model.pkl", "rb"))
scaler = pickle.load(open("std_scaler.pkl", "rb"))

# Create Flask App
app = Flask(__name__)


# Home Page
@app.route('/')
def welcome():
    return render_template("index.html")


# Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    months_as_customer = float(request.form["months_as_customer"])
    policy_number = float(request.form["policy_number"])
    policy_bind_date = float(request.form["policy_bind_date"])
    policy_state = float(request.form["policy_state"])
    policy_csl = float(request.form["policy_csl"])
    policy_deductable = float(request.form["policy_deductable"])
    policy_annual_premium = float(request.form["policy_annual_premium"])
    insured_zip = float(request.form["insured_zip"])
    insured_sex = float(request.form["insured_sex"])
    insured_occupation = float(request.form["insured_occupation"])
    insured_hobbies = float(request.form["insured_hobbies"])
    insured_relationship = float(request.form["insured_relationship"])
    capital_gains = float(request.form["capital_gains"])
    capital_loss = float(request.form["capital_loss"])
    incident_date = float(request.form["incident_date"])
    incident_type = float(request.form["incident_type"])
    collision_type = float(request.form["collision_type"])
    incident_severity = float(request.form["incident_severity"])
    authorities_contacted = float(request.form["authorities_contacted"])
    incident_location = float(request.form["incident_location"])
    incident_hour_of_the_day = float(request.form["incident_hour_of_the_day"])
    number_of_vehicles_involved = float(request.form["number_of_vehicles_involved"])
    property_damage = float(request.form["property_damage"])
    bodily_injuries = float(request.form["bodily_injuries"])
    witnesses = float(request.form["witnesses"])
    police_report_available = float(request.form["police_report_available"])
    total_claim_amount = float(request.form["total_claim_amount"])
    auto_make = float(request.form["auto_make"])
    auto_model = float(request.form["auto_model"])
    auto_year = float(request.form["auto_year"])

    # Store values in list
    data = [[months_as_customer, policy_number, policy_bind_date, policy_state,
             policy_csl, policy_deductable, policy_annual_premium, insured_zip,
             insured_sex, insured_occupation, insured_hobbies, insured_relationship,
             capital_gains, capital_loss, incident_date, incident_type,
             collision_type, incident_severity, authorities_contacted,
             incident_location, incident_hour_of_the_day,
             number_of_vehicles_involved, property_damage,
             bodily_injuries, witnesses, police_report_available,
             total_claim_amount, auto_make, auto_model, auto_year]]

    # Scale data
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    if prediction[0] == 0:
        result = "Legal Insurance Claim"
    else:
        result = "Fraud Insurance Claim"

    return render_template("index.html", prediction_text=result)


# Main Function
if __name__ == "__main__":
    app.run(debug=True)