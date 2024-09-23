import sys
import os
import pandas as pd
from src.exception import CustomException
import datetime

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, 
    cons_12m :  int,  
    cons_gas_12m :  int,
    cons_last_month :int,  
    forecast_cons_12m:float,
    forecast_cons_year:int,
    forecast_discount_energy:float,
    forecast_meter_rent_12m:float,
    forecast_price_energy_off_peak:float,
    forecast_price_energy_peak:float,
    forecast_price_pow_off_peak:float,
    imp_cons:float,
    margin_gross_pow_ele:float,
    margin_net_pow_ele:float,
    nb_prod_act:int,
    net_margin:float,
    num_years_antig:int, 
    pow_max: float,
    date_activ_day:int,
    date_activ_month:int,
    date_activ_year:int, 
    date_end_day:int, 
    date_end_month:int,
    date_end_year:int, 
    date_modif_prod_day:int, 
    date_modif_prod_month:int,
    date_modif_prod_year:int, 
    date_renewal_day:int, 
    date_renewal_month:int,
    date_renewal_year:int):
                
        self.cons_12m = cons_12m
        self.cons_gas_12m = cons_gas_12m
        self.cons_last_month = cons_last_month
        self.forecast_cons_12m = forecast_cons_12m
        self.forecast_cons_year = forecast_cons_year
        self.forecast_discount_energy = forecast_discount_energy
        self.forecast_meter_rent_12m = forecast_meter_rent_12m
        self.forecast_price_energy_off_peak =forecast_price_energy_off_peak
        self.forecast_price_energy_peak = forecast_price_energy_peak
        self.forecast_price_pow_off_peak = forecast_price_pow_off_peak
        self.imp_cons = imp_cons
        self.margin_gross_pow_ele = margin_gross_pow_ele
        self.margin_net_pow_ele = margin_net_pow_ele
        self.nb_prod_act = nb_prod_act
        self.net_margin = net_margin
        self.num_years_antig = num_years_antig
        self.pow_max =pow_max
        self.date_activ_day = date_activ_day
        self.date_activ_month =date_activ_month
        self.date_activ_year = date_activ_year
        self.date_end_day =  date_end_day
        self.date_end_month = date_end_month
        self.date_end_year  = date_end_year 
        self.date_modif_prod_day = date_modif_prod_day
        self.date_modif_prod_month = date_modif_prod_month
        self.date_modif_prod_year  = date_modif_prod_year
        self.date_renewal_day  = date_renewal_day
        self.date_renewal_month = date_renewal_month
        self.date_renewal_year = date_renewal_year

    def get_data_as_data_frame(self):
                try:
                    custom_data_input_dict = {
                        "cons_12m":[self.cons_12m],
                        "cons_gas_12m ":[self.cons_gas_12m],
                        "cons_last_month":[self.cons_last_month],
                        "forecast_cons_12m":[self.forecast_cons_12m],
                        "forecast_cons_year":[self.forecast_cons_year],
                        "forecast_discount_energy":[self.forecast_discount_energy],
                        "forecast_meter_rent_12m":[self.forecast_meter_rent_12m],
                        "forecast_price_energy_off_peak":[self.forecast_price_energy_off_peak],
                        "forecast_price_energy_peak":[self.forecast_price_energy_peak],
                        "forecast_price_pow_off_peak":[self.forecast_price_pow_off_peak],
                        "imp_cons":[self.imp_cons],
                        "margin_gross_pow_ele":[self.margin_gross_pow_ele],
                        "margin_net_pow_ele":[self.margin_net_pow_ele],
                        "nb_prod_act":[self.nb_prod_act],
                        "net_margin":[self.net_margin],
                        "num_years_antig":[self.num_years_antig],        
                        "pow_max":[self.pow_max],
                        "date_activ_day": [self.date_activ_day],
                        "date_activ_month": [self.date_activ_month],
                        "date_activ_year": [self.date_activ_year],
                        "date_end_day" :   [self.date_end_day],
                        "date_end_month": [self.date_end_month],
                        "date_end_year" : [self.date_end_year],
                        "date_modif_prod_day" : [self.date_modif_prod_day],
                        "date_modif_prod_month" :  [self.date_modif_prod_month],
                        "date_modif_prod_year"  :[self.date_modif_prod_year],
                        "date_renewal_day"  : [self.date_renewal_day],
                        "date_renewal_month"  : [self.date_renewal_month],
                        "date_renewal_year" :  [self.date_renewal_year],
                        }
                    return pd.DataFrame(custom_data_input_dict)

                except Exception as e:
                    raise CustomException(e,sys)