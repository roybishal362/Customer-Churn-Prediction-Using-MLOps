from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

# Load the model and preprocessor
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form submission
        cons_12m = request.form['cons_12m']
        cons_gas_12m = request.form['cons_gas_12m']
        cons_last_month = request.form['cons_last_month']
        date_activ = request.form['date_activ']
        date_end = request.form['date_end']
        pow_max = request.form['pow_max']
        date_modif_prod = request.form['date_modif_prod']
        date_renewal = request.form['date_renewal']
        forecast_cons_12m = request.form['forecast_cons_12m']
        forecast_cons_year = request.form['forecast_cons_year']
        forecast_discount_energy = request.form['forecast_discount_energy']
        forecast_meter_rent_12m = request.form['forecast_meter_rent_12m']
        forecast_price_energy_off_peak = request.form['forecast_price_energy_off_peak']
        forecast_price_energy_peak = request.form['forecast_price_energy_peak']
        forecast_price_pow_off_peak = request.form['forecast_price_pow_off_peak']
        has_gas = request.form['has_gas']
        imp_cons = request.form['imp_cons']
        margin_gross_pow_ele = request.form['margin_gross_pow_ele']
        margin_net_pow_ele = request.form['margin_net_pow_ele']
        nb_prod_act = request.form['nb_prod_act']
        net_margin = request.form['net_margin']
        num_years_antig = request.form['num_years_antig']
        
        # Convert date inputs to datetime objects
        date_activ = datetime.strptime(date_activ, '%Y-%m-%d')
        date_end = datetime.strptime(date_end, '%Y-%m-%d')
        date_modif_prod = datetime.strptime(date_modif_prod, '%Y-%m-%d')
        date_renewal = datetime.strptime(date_renewal, '%Y-%m-%d')

        # Create a CustomData object for feature inputs
        customer_data = CustomData(
            cons_12m=int(cons_12m),
            cons_gas_12m=int(cons_gas_12m),
            cons_last_month=int(cons_last_month),
            date_activ=date_activ,
            date_end=date_end,
            pow_max=float(pow_max),
            date_modif_prod=date_modif_prod,
            date_renewal=date_renewal,
            forecast_cons_12m=int(forecast_cons_12m),
            forecast_cons_year=int(forecast_cons_year),
            forecast_discount_energy=float(forecast_discount_energy),
            forecast_meter_rent_12m=float(forecast_meter_rent_12m),
            forecast_price_energy_off_peak=float(forecast_price_energy_off_peak),
            forecast_price_energy_peak=float(forecast_price_energy_peak),
            forecast_price_pow_off_peak=float(forecast_price_pow_off_peak),
            has_gas=int(has_gas),
            imp_cons=int(imp_cons),
            margin_gross_pow_ele=float(margin_gross_pow_ele),
            margin_net_pow_ele=float(margin_net_pow_ele),
            nb_prod_act=int(nb_prod_act),
            net_margin=float(net_margin),
            num_years_antig=int(num_years_antig)
        )
        
        # Convert the custom data into a pandas DataFrame
        input_features = customer_data.get_data_as_data_frame()

        # Get the churn prediction from the pipeline
        prediction = predict_pipeline.predict(input_features)

        # Prepare the result
        result = 'Churn' if prediction[0] == 1 else 'Not Churn'
        
        return jsonify({'prediction': result})

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
