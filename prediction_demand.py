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
    # On divise par 24 pour obtenir la puissance moyenne en MW
    df['y'] = df['Scaled Demand (MWh)'] / 24.0
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
# Entraînement et sauvegarde du modèle Daily Lissé
###############################################

def train_daily_model_smooth(data, model_file='daily_model_smooth.pkl', error_file='daily_error_smooth.txt'):
    """
    Entraîne un modèle Prophet sur des données quotidiennes avec une tendance lissée.
    Utilise des hyperparamètres faibles pour limiter la réactivité aux changements brusques.
    Calcule la distribution des résidus, sauvegarde le modèle et enregistre la distribution dans un fichier texte.
    Retourne le modèle entraîné, la moyenne (mu), l'écart-type (sigma) et le résidu minimal (min_resid) des résidus.
    """
    ts = prepare_daily_time_series(data)
    
    model = Prophet(
        daily_seasonality=False, 
        weekly_seasonality=False, 
        yearly_seasonality=True,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=1.0
    )
    model.fit(ts)
    
    ts_fitted = model.predict(ts)
    residuals = ts['y'] - ts_fitted['yhat']
    mu = residuals.mean()
    sigma = residuals.std()
    min_resid = residuals.min()  # Le plus grand écart négatif
    
    with open(error_file, 'w') as f:
        f.write(f"Moyenne des résidus: {mu:.4f}\n")
        f.write(f"Ecart-type des résidus: {sigma:.4f}\n")
        f.write(f"Résidu minimal (cap négatif): {min_resid:.4f}\n")
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modèle daily lissé sauvegardé dans {model_file}")
    print(f"Distribution des erreurs sauvegardée dans {error_file}")
    return model, mu, sigma, min_resid

###############################################
# Entraînement et sauvegarde du modèle Hourly Lissé
###############################################

def train_hourly_model_smooth(data, model_file='hourly_model_smooth.pkl', error_file='hourly_error_smooth.txt'):
    """
    Entraîne un modèle Prophet sur des données horaires en visant une courbe lissée.
    Utilise des hyperparamètres faibles et une saisonnalité horaire avec un fourier_order réduit.
    Calcule la distribution des résidus, sauvegarde le modèle et enregistre la distribution dans un fichier texte.
    Retourne le modèle entraîné, la moyenne (mu), l'écart-type (sigma) et le résidu minimal (min_resid) des résidus.
    """
    ts = prepare_hourly_time_series(data)
    
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=1.0
    )
    model.add_seasonality(name='hourly', period=24, fourier_order=3)
    
    model.fit(ts)
    
    ts_fitted = model.predict(ts)
    residuals = ts['y'] - ts_fitted['yhat']
    mu = residuals.mean()
    sigma = residuals.std()
    min_resid = residuals.min()
    
    with open(error_file, 'w') as f:
        f.write(f"Moyenne des résidus: {mu:.4f}\n")
        f.write(f"Ecart-type des résidus: {sigma:.4f}\n")
        f.write(f"Résidu minimal (cap négatif): {min_resid:.4f}\n")
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modèle hourly lissé sauvegardé dans {model_file}")
    print(f"Distribution des erreurs sauvegardée dans {error_file}")
    return model, mu, sigma, min_resid

###############################################
# Prédiction avec modèle (ajout de bruit)
###############################################

def predict_with_model(model, start_date, end_date, granularity='daily', add_noise=False, mu=0, sigma=0, min_resid=0):
    """
    Prédit la demande en MW entre start_date et end_date en utilisant le modèle Prophet.
    La fréquence est 'D' pour daily et 'H' pour hourly.
    Si add_noise est True, ajoute un bruit aléatoire tiré de N(mu, sigma) aux prédictions.
    Toutefois, pour tout bruit négatif, on le plafonne à min_resid (c'est-à-dire, le bruit négatif ne pourra être inférieur à min_resid).
    Retourne un DataFrame avec 'ds', 'yhat' et, si add_noise, 'yhat_noisy'.
    """
    freq = 'D' if granularity=='daily' else 'H'
    future = pd.date_range(start=start_date, end=end_date, freq=freq)
    future_df = pd.DataFrame({'ds': future})
    
    forecast = model.predict(future_df)
    prediction = forecast[['ds', 'yhat']].copy()
    
    if add_noise:
        noise = np.random.normal(mu, sigma, size=len(prediction))
        # Pour les valeurs négatives, on plafonne le bruit à min_resid (qui est négatif ou 0)
        noise = np.where(noise < 0, np.maximum(noise, min_resid), noise)
        prediction['yhat_noisy'] = prediction['yhat'] + noise
    return prediction

###############################################
# Exemple d'utilisation
###############################################

# Modèle Daily Lissé avec bruit
df_daily = pd.read_csv("Demand_DATA_scaled_daily.csv")
daily_model, daily_mu, daily_sigma, daily_min = train_daily_model_smooth(
    df_daily, model_file='daily_model_smooth.pkl', error_file='daily_error_smooth.txt'
)

start_daily = "2019-01-01"
end_daily = "2019-12-31"
pred_daily = predict_with_model(
    daily_model, 
    start_date=start_daily, 
    end_date=end_daily, 
    granularity='daily', 
    add_noise=True, 
    mu=daily_mu, 
    sigma=daily_sigma,
    min_resid=daily_min
)

# Filtrer les données historiques pour la plage de simulation
ts_daily = prepare_daily_time_series(df_daily)
ts_daily_filtered = ts_daily[(ts_daily['ds'] >= start_daily) & (ts_daily['ds'] <= end_daily)]

plt.figure(figsize=(12,6))
plt.bar(pred_daily['ds'], pred_daily['yhat_noisy'], width=0.8, color='red', alpha=0.7, label='Prédiction avec bruit (Daily)')
plt.bar(ts_daily_filtered['ds'], ts_daily_filtered['y'], width=0.8, color='blue', alpha=0.7, label='Données Historiques (Daily)')
plt.xlabel('Date')
plt.ylabel('Demande (MW)')
plt.title('Prédiction Journalière de la Demande (Daily Lissé) avec Bruit')
plt.legend()
plt.show()

# Modèle Hourly Lissé avec bruit
df_hourly = pd.read_csv("Demand_DATA_scaled_hourly.csv")
hourly_model, hourly_mu, hourly_sigma, hourly_min = train_hourly_model_smooth(
    df_hourly, model_file='hourly_model_smooth.pkl', error_file='hourly_error_smooth.txt'
)

start_hourly = "2024-01-01 00:00:00"
end_hourly = "2024-12-31 23:00:00"
pred_hourly = predict_with_model(
    hourly_model, 
    start_date=start_hourly, 
    end_date=end_hourly, 
    granularity='hourly', 
    add_noise=True, 
    mu=hourly_mu, 
    sigma=hourly_sigma,
    min_resid=hourly_min
)

ts_hourly = prepare_hourly_time_series(df_hourly)
ts_hourly_filtered = ts_hourly[(ts_hourly['ds'] >= start_hourly) & (ts_hourly['ds'] <= end_hourly)]

plt.figure(figsize=(12,6))
plt.bar(pred_hourly['ds'], pred_hourly['yhat_noisy'], width=0.03, color='red', alpha=0.7, label='Prédiction avec bruit (Hourly)')
plt.bar(ts_hourly_filtered['ds'], ts_hourly_filtered['y'], width=0.03, color='blue', alpha=0.7, label='Données Historiques (Hourly)')
plt.xlabel('Date et Heure')
plt.ylabel('Demande (MW)')
plt.title('Prédiction Horaire de la Demande (Hourly Lissé) avec Bruit')
plt.legend()
plt.show()
