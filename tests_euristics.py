import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from views_competition import config, evaluate, datautils, evallib

#array([490, 491, 492, 493, 494, 495], dtype=int64)
with open('data/pickled/cm_t1.pkl', 'rb') as f:
    data = pickle.load(f)

data = data.reset_index()
##observation are in ["ln_ged_best_sb"]

data["ged_best_sb_T0"] = np.exp(np.log(data.ged_best_sb + 1) - data.d_ln_ged_best_sb) -1


##Mjlogchabge = np.log(Maj + 1) - np.log(T0 + 1)

#select a country
data["Eur_mean_12_log_change"] = 0

#This loop creates an average of the last 12 month
for country_id_selected in data.country_id.unique():

    stop_learnind_month_id = 488
    ged_cm_postpatch=pd.read_parquet('data/ged_cm_postpatch.parquet').reset_index()

    #subset for the specific country
    country_selected_past_data = ged_cm_postpatch.copy()
    country_selected_past_data = country_selected_past_data.loc[country_selected_past_data.country_id == country_id_selected]
    country_selected_past_data_up_to_488 = country_selected_past_data[country_selected_past_data.month_id<=stop_learnind_month_id]
    country_selected_past_data_up_to_488 = country_selected_past_data_up_to_488.loc[country_selected_past_data.month_id>488-12]

    myStats = country_selected_past_data_up_to_488.ged_best_sb.mean()
    ged_best_sb_T0 = ged_cm_postpatch.loc[(ged_cm_postpatch.month_id==488) & (ged_cm_postpatch.country_id==country_id_selected),"ged_best_sb"]
    Eur_mean_log_change = np.log(myStats+1) - np.log(ged_best_sb_T0+1)
    data.loc[data.country_id == country_id_selected,"Eur_mean_12_log_change"] = Eur_mean_log_change.values[0]



data["Eur_max_12_log_change"] = 0

#This loop creates the max of the last 12 month
for country_id_selected in data.country_id.unique():

    stop_learnind_month_id = 488
    ged_cm_postpatch=pd.read_parquet('data/ged_cm_postpatch.parquet').reset_index()

    #subset for the specific country
    country_selected_past_data = ged_cm_postpatch.copy()
    country_selected_past_data = country_selected_past_data.loc[country_selected_past_data.country_id == country_id_selected]
    country_selected_past_data_up_to_488 = country_selected_past_data[country_selected_past_data.month_id<=stop_learnind_month_id]
    country_selected_past_data_up_to_488 = country_selected_past_data_up_to_488.loc[country_selected_past_data.month_id>488-12]

    myStats = country_selected_past_data_up_to_488.ged_best_sb.max()
    ged_best_sb_T0 = ged_cm_postpatch.loc[(ged_cm_postpatch.month_id==488) & (ged_cm_postpatch.country_id==country_id_selected),"ged_best_sb"]
    Eur_mean_log_change = np.log(myStats+1) - np.log(ged_best_sb_T0+1)
    data.loc[data.country_id == country_id_selected,"Eur_max_12_log_change"] = Eur_mean_log_change.values[0]




data["Eur_mean_6_log_change"] = 0

#This loop creates an average of the last 12 month
for country_id_selected in data.country_id.unique():

    stop_learnind_month_id = 488
    ged_cm_postpatch=pd.read_parquet('data/ged_cm_postpatch.parquet').reset_index()

    #subset for the specific country
    country_selected_past_data = ged_cm_postpatch.copy()
    country_selected_past_data = country_selected_past_data.loc[country_selected_past_data.country_id == country_id_selected]
    country_selected_past_data_up_to_488 = country_selected_past_data[country_selected_past_data.month_id<=stop_learnind_month_id]
    country_selected_past_data_up_to_488 = country_selected_past_data_up_to_488.loc[country_selected_past_data.month_id>488-6]

    myStats = country_selected_past_data_up_to_488.ged_best_sb.mean()
    ged_best_sb_T0 = ged_cm_postpatch.loc[(ged_cm_postpatch.month_id==488) & (ged_cm_postpatch.country_id==country_id_selected),"ged_best_sb"]
    Eur_mean_log_change = np.log(myStats+1) - np.log(ged_best_sb_T0+1)
    data.loc[data.country_id == country_id_selected,"Eur_mean_6_log_change"] = Eur_mean_log_change.values[0]



data["Eur_max_6_log_change"] = 0

#This loop creates the max of the last 12 month
for country_id_selected in data.country_id.unique():

    stop_learnind_month_id = 488
    ged_cm_postpatch=pd.read_parquet('data/ged_cm_postpatch.parquet').reset_index()

    #subset for the specific country
    country_selected_past_data = ged_cm_postpatch.copy()
    country_selected_past_data = country_selected_past_data.loc[country_selected_past_data.country_id == country_id_selected]
    country_selected_past_data_up_to_488 = country_selected_past_data[country_selected_past_data.month_id<=stop_learnind_month_id]
    country_selected_past_data_up_to_488 = country_selected_past_data_up_to_488.loc[country_selected_past_data.month_id>488-6]

    myStats = country_selected_past_data_up_to_488.ged_best_sb.max()
    ged_best_sb_T0 = ged_cm_postpatch.loc[(ged_cm_postpatch.month_id==488) & (ged_cm_postpatch.country_id==country_id_selected),"ged_best_sb"]
    Eur_mean_log_change = np.log(myStats+1) - np.log(ged_best_sb_T0+1)
    data.loc[data.country_id == country_id_selected,"Eur_max_6_log_change"] = Eur_mean_log_change.values[0]

data["LOCF_1"] = 0

# This loop creates the max of the last 12 month
for country_id_selected in data.country_id.unique():
    stop_learnind_month_id = 488

    # subset for the specific country
    country_selected_past_data = ged_cm_postpatch.copy()
    country_selected_past_data = country_selected_past_data.loc[
        country_selected_past_data.country_id == country_id_selected]

    country_selected_past_data_up_to_488 = country_selected_past_data[country_selected_past_data.month_id == 487]

    myStats = country_selected_past_data_up_to_488.ged_best_sb

    ged_best_sb_T0 = ged_cm_postpatch.loc[
        (ged_cm_postpatch.month_id == 488) & (ged_cm_postpatch.country_id == country_id_selected), "ged_best_sb"]
    Eur_mean_log_change = np.log(myStats + 1) - np.log(ged_best_sb_T0 + 1)
    data.loc[data.country_id == country_id_selected, "LOCF_1"] = Eur_mean_log_change.values[0]


obs = data['d_ln_ged_best_sb']
for name in ['Eur_max_12_log_change', 'Eur_mean_12_log_change','Eur_max_6_log_change', 'Eur_mean_6_log_change','ettensperger', 'no_change', 'mueller',
             'randahl_vmm_weighted','chadefaux', 'randahl_hmm_weighted', 'malone', 'randahl_hhmm_weighted','oswald']:
    pred = data[name]



    print(f"{name}: {metrics.mean_absolute_error(obs,pred)}")

    #print(f"TADDA_1 {name} {evallib.tadda_score(obs, pred, 1)}")
    #"TADDA_1_nonzero": lambda obs, pred: evallib.tadda_score(obs, pred, 1),
    #"TADDA_2": lambda obs, pred: evallib.tadda_score(obs, pred, 2),
    #"TADDA_2_nonzero": lambda obs, pred: evallib.tadda_score(obs, pred, 2),





