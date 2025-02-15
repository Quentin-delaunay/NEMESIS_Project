import pandas as pd
import numpy as np
import pickle
from prophet import Prophet
import matplotlib.pyplot as plt

###############################################
# Préparation des séries temporelles
###############################################

def prepare_daily_time_series(data):
    """
    Prépare la série quotidienne à partir des données scaleées.
    Les données doivent contenir 'Timestamp' et 'Scaled Demand (MWh)'.
    Agrège par jour et divise par 24 pour obtenir la demande moyenne en MW.
    Retourne un DataFrame avec colonnes 'ds' et 'y'.
    """
    df = data.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').resample('D').sum().reset_index()
    df['y'] = df['Scaled Demand (MWh)'] / 24.0 - 3000 
    return df.rename(columns={'Timestamp': 'ds'})[['ds', 'y']]

def prepare_hourly_time_series(data):
    """
    Prépare la série horaire à partir des données scaleées.
    1 MWh/h correspond à 1 MW.
    Retourne un DataFrame avec colonnes 'ds' et 'y'.
    """
    df = data.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['y'] = df['Scaled Demand (MWh)']
    return df.rename(columns={'Timestamp': 'ds'})[['ds', 'y']]

###############################################
# Entraînement et sauvegarde du modèle Daily
###############################################

def train_daily_model(data, model_file='daily_model.pkl', error_file='daily_error.txt'):
    """
    Entraîne un modèle Prophet sur des données quotidiennes, calcule la distribution
    des résidus, sauvegarde le modèle et enregistre la distribution dans un fichier texte.
    Retourne le modèle entraîné, la moyenne (mu) et l'écart-type (sigma) des résidus.
    """
    ts = prepare_daily_time_series(data)
    
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(ts)
    
    # Calculer les résidus sur les données historiques
    ts_fitted = model.predict(ts)
    residuals = ts['y'] - ts_fitted['yhat']
    mu = residuals.mean()
    sigma = residuals.std()
    
    with open(error_file, 'w') as f:
        f.write(f"Moyenne des résidus: {mu:.4f}\n")
        f.write(f"Ecart-type des résidus: {sigma:.4f}\n")
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modèle daily sauvegardé dans {model_file}")
    print(f"Distribution des erreurs sauvegardée dans {error_file}")
    return model, mu, sigma

###############################################
# Entraînement et sauvegarde du modèle Hourly
###############################################

def train_hourly_model(data, model_file='hourly_model.pkl', error_file='hourly_error.txt'):
    """
    Entraîne un modèle Prophet sur des données horaires, calcule la distribution
    des résidus, sauvegarde le modèle et enregistre la distribution dans un fichier texte.
    Retourne le modèle entraîné.
    (Pour hourly, nous n'ajoutons pas de bruit dans les prédictions.)
    """
    ts = prepare_hourly_time_series(data)
    
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
                    changepoint_prior_scale=0.2, seasonality_prior_scale=10.0)
    model.fit(ts)
    
    ts_fitted = model.predict(ts)
    residuals = ts['y'] - ts_fitted['yhat']
    mu = residuals.mean()
    sigma = residuals.std()
    
    with open(error_file, 'w') as f:
        f.write(f"Moyenne des résidus: {mu:.4f}\n")
        f.write(f"Ecart-type des résidus: {sigma:.4f}\n")
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modèle hourly sauvegardé dans {model_file}")
    print(f"Distribution des erreurs sauvegardée dans {error_file}")
    return model

###############################################
# Prédiction avec modèle (Daily avec bruit, Hourly sans bruit)
###############################################

def predict_daily(model, start_date, end_date, add_noise=True, mu=0, sigma=0):
    """
    Utilise le modèle daily pour prédire la demande entre start_date et end_date.
    Si add_noise est True, ajoute un bruit aléatoire tiré de N(mu, sigma) à la prédiction.
    Retourne un DataFrame avec 'ds', 'yhat' et, si add_noise, 'yhat_noisy'.
    """
    future = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame({'ds': future})
    
    forecast = model.predict(future_df)
    prediction = forecast[['ds', 'yhat']].copy()
    if add_noise:
        noise = np.random.normal(mu, sigma, size=len(prediction))
        prediction['yhat_noisy'] = prediction['yhat'] + noise 
    return prediction

def predict_hourly(model, start_date, end_date):
    """
    Utilise le modèle hourly pour prédire la demande entre start_date et end_date.
    Aucune composante aléatoire n'est ajoutée.
    Retourne un DataFrame avec 'ds' et 'yhat'.
    """
    future = pd.date_range(start=start_date, end=end_date, freq='H')
    future_df = pd.DataFrame({'ds': future})
    
    forecast = model.predict(future_df)
    prediction = forecast[['ds', 'yhat']]
    return prediction

###############################################
# Exemple d'utilisation
###############################################

# Modèle Daily
df_daily = pd.read_csv("Demand_DATA_scaled_daily.csv")
daily_model, daily_mu, daily_sigma = train_daily_model(df_daily, model_file='daily_model.pkl', error_file='daily_error.txt')

# Prédiction Daily sur une période (par exemple, 2019)
pred_daily = predict_daily(daily_model, start_date="2019-01-01", end_date="2025-12-31", add_noise=True, mu=daily_mu, sigma=daily_sigma)

plt.figure(figsize=(12,6))
plt.bar(pred_daily['ds'], pred_daily['yhat_noisy'], width=0.8, color='red', alpha=0.7, label='Prédiction avec bruit (Daily)')
plt.bar(prepare_daily_time_series(df_daily)['ds'], prepare_daily_time_series(df_daily)['y'], width=0.8, color='blue', alpha=0.7, label='Données Historiques (Daily)')
plt.xlabel('Date')
plt.ylabel('Demande (MW)')
plt.title('Prédiction Journalière de la Demande avec Bruit')
plt.legend()
plt.show()

# # Modèle Hourly (sans bruit)
# df_hourly = pd.read_csv("Demand_DATA_scaled_hourly.csv")
# hourly_model = train_hourly_model(df_hourly, model_file='hourly_model.pkl', error_file='hourly_error.txt')

# pred_hourly = predict_hourly(hourly_model, start_date="2024-01-01 00:00:00", end_date="2024-12-31 23:00:00")

# plt.figure(figsize=(12,6))
# plt.bar(pred_hourly['ds'], pred_hourly['yhat'], width=0.03, color='red', alpha=0.7, label='Prédiction (Hourly)')
# plt.bar(prepare_hourly_time_series(df_hourly)['ds'], prepare_hourly_time_series(df_hourly)['y'], width=0.03, color='blue', alpha=0.7, label='Données Historiques (Hourly)')
# plt.xlabel('Date et Heure')
# plt.ylabel('Demande (MW)')
# plt.title('Prédiction Horaire de la Demande (sans bruit)')
# plt.legend()
# plt.show()
