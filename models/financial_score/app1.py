from flask import Flask, redirect, render_template, request, Blueprint, url_for
import pickle
import numpy as np
import random

# Define the blueprint for financial score
financial_score_blueprint = Blueprint('financial_score', __name__, template_folder='templates')
social_media_score_blueprint = Blueprint('social_media_score', __name__, template_folder='templates')
# Load the trained Random Forest model from the pickle file
with open('models/financial_score/financial_model.pkl', 'rb') as model_file:
    rfc_model = pickle.load(model_file)

# Define class mapping for predictions
class_mapping = {0: 'Bad', 1: 'Neutral', 2: 'Good'}

# Define route for the financial form in the blueprint
@financial_score_blueprint.route('/financial_form')
def financial_form():
    return render_template('financial_template.html')

# Define the route for prediction inside the blueprint
@financial_score_blueprint.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form (14 features)
    age = float(request.form['age'])
    annual_income = float(request.form['annual_income'])
    monthly_salary = float(request.form['monthly_salary'])
    num_bank_accounts = int(request.form['num_bank_accounts'])
    num_credit_cards = int(request.form['num_credit_cards'])
    interest_rate = float(request.form['interest_rate'])
    num_loans = int(request.form['num_loans'])
    delay_date = int(request.form['delay_date'])
    delayed_payments = int(request.form['delayed_payments'])
    changed_limit = float(request.form['changed_limit'])
    credit_inquiries = int(request.form['credit_inquiries'])
    outstanding_debt = float(request.form['outstanding_debt'])
    credit_utilization = float(request.form['credit_utilization'])
    credit_history_age = int(request.form['credit_history_age'])
    emi_per_month = float(request.form['emi_per_month'])
    amt_month = float(request.form['amt_month'])
    monthly_balance = float(request.form['monthly_balance'])

    # Prepare the input data in the correct format for prediction
    features = np.array([[ 
        age, annual_income, monthly_salary, num_bank_accounts,
        num_credit_cards, interest_rate, num_loans, delay_date, delayed_payments, 
        changed_limit, credit_inquiries, outstanding_debt, credit_utilization,
        credit_history_age, emi_per_month, amt_month, monthly_balance
    ]])

    # Predict the category using the trained model (without scaling)
    prediction = rfc_model.predict(features)

    # Get the predicted category (Bad, Neutral, or Good)
    predicted_category = class_mapping[prediction[0]]

    # Return the result to the frontend
    return redirect(url_for('social_media_score.social_media_form', predicted_value=predicted_category))
@social_media_score_blueprint.route('/social_media_form')
def social_media_form():
    # Get predicted value from query parameter
    predicted_value = request.args.get('predicted_value', 'Neutral')  
    return render_template('social_media_template.html', predicted_value=predicted_value)

# Initialize the Flask app
app = Flask(__name__)

# Register the blueprint for the financial score with the app
app.register_blueprint(financial_score_blueprint, url_prefix='/financial_score')
app.register_blueprint(social_media_score_blueprint, url_prefix='/social_media_score')
# Define a route for the home page (optional)
@app.route('/')
def home():
    return render_template('financial_template.html')

if __name__ == '__main__':
    app.run(debug=True)
