import pandas as pd
import glob
import os

# --- 1. Chargement et préparation des données mensuelles de Géorgie ---

georgia_file = r"Demand DATA\MM_GEORGIA_GEN.csv"
df_geo = pd.read_csv(georgia_file)
df_geo['Date'] = pd.to_datetime(df_geo['category'], format='%a %b %d %Y', errors='coerce')
df_geo['Month'] = df_geo['Date'].dt.to_period('M')
geo_monthly = df_geo.set_index('Month')['Net Generation (MWh)']
print("Données géorgiennes mensuelles:")
print(geo_monthly)

# --- Variables globales pour stocker les scaling factors issus des fichiers daily ---
scaling_records = []  # Chaque enregistrement : {'Year': ..., 'Month': ..., 'Scaling': ...}

# --- Fonctions utilitaires ---

def get_timestamp_column(df):
    """Retourne le nom de la colonne contenant 'timestamp' (insensible à la casse)."""
    for col in df.columns:
        if 'timestamp' in col.lower():
            return col
    raise KeyError("Aucune colonne de timestamp trouvée dans le fichier.")

def get_demand_column(df):
    """Retourne le nom de la colonne contenant 'demand' mais pas 'forecast' (insensible à la casse)."""
    for col in df.columns:
        if "demand" in col.lower() and "forecast" not in col.lower():
            return col
    raise KeyError("Aucune colonne de demande trouvée dans le fichier.")

# --- Fonction de traitement pour les fichiers daily ---
def process_daily_file(file_path):
    df = pd.read_csv(file_path)
    ts_col = get_timestamp_column(df)
    ts = df[ts_col].str.replace(", Central Time", "", regex=False).str.strip()
    df['Timestamp'] = pd.to_datetime(ts, format='%m/%d/%Y', errors='coerce')
    
    if df['Timestamp'].isnull().any():
        print(f"ATTENTION : certaines dates n'ont pas pu être converties dans {file_path}")
    
    df['Month'] = df['Timestamp'].dt.to_period('M')
    demand_col = get_demand_column(df)
    # Calcul de la demande mensuelle et du total de Net Generation (MWh) dans le fichier SW
    netgen_col = "Net Generation (MWh)"  # On suppose que ce nom est constant
    monthly_netgen = df.groupby('Month')[netgen_col].sum()
    
    # Filtrer pour ne traiter que les mois correspondant à l'année du fichier
    basename = os.path.basename(file_path)
    try:
        file_year = int(basename.split('_')[0])
    except Exception as e:
        raise ValueError(f"Impossible d'extraire l'année du nom du fichier {basename}: {e}")
    monthly_netgen = monthly_netgen[monthly_netgen.index.year == file_year]
    
    scaling_factors = geo_monthly.reindex(monthly_netgen.index) / monthly_netgen
    print(f"\nFacteurs de scaling pour {basename}:")
    print(scaling_factors)
    
    # Enregistrer les scaling factors pour ce fichier daily
    for period, factor in scaling_factors.items():
        scaling_records.append({
            'Year': period.year,
            'Month': period.month,
            'Scaling': factor
        })
    
    # On peut également appliquer ces scaling factors au fichier daily si nécessaire
    df['Scaling Factor'] = df['Month'].map(scaling_factors)
    df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
    df['Scaled Demand (MWh)'] = df[demand_col] * df['Scaling Factor']
    return df

# --- Traitement des fichiers daily ---
daily_files = glob.glob(os.path.join(r"Demand DATA\YYYY_D_SW", "*_D_SW.csv"))
if daily_files:
    list_dfs_daily = []
    for file in daily_files:
        print(f"\nTraitement du fichier journalier: {file}")
        try:
            df_daily = process_daily_file(file)
            list_dfs_daily.append(df_daily)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file}: {e}")
    if list_dfs_daily:
        df_daily_all = pd.concat(list_dfs_daily, ignore_index=True)
        df_daily_all.to_csv("Demand_DATA_scaled_daily.csv", index=False)
        print("\nFichier 'Demand_DATA_scaled_daily.csv' enregistré.")
else:
    print("Aucun fichier journalier trouvé.")

# --- Calcul de la moyenne des scaling factors par mois (basé sur les fichiers daily) ---
if scaling_records:
    df_scaling = pd.DataFrame(scaling_records)
    # Grouper par numéro de mois (indépendamment de l'année)
    avg_scaling = df_scaling.groupby(df_scaling['Month'].apply(lambda x: x))['Scaling'].mean().to_dict()
    # S'assurer que les clés soient des entiers (1 à 12)
    avg_scaling = {int(k): v for k, v in avg_scaling.items()}
    print("\nMoyenne des scaling factors par mois (calculée à partir des fichiers daily) :")
    print(avg_scaling)
else:
    avg_scaling = {}
    print("Aucun scaling n'a été enregistré à partir des fichiers daily.")

# --- Fonction de traitement pour les fichiers horaires ---
def process_hourly_file(file_path, avg_scaling):
    df = pd.read_csv(file_path)
    ts_col = get_timestamp_column(df)
    ts = df[ts_col].str.replace(' CST', '', regex=False)
    ts = ts.str.replace(' CDT', '', regex=False)
    ts = ts.str.replace(' a.m.', ' AM', regex=False).str.replace(' p.m.', ' PM', regex=False)
    df['Timestamp'] = pd.to_datetime(ts, errors='coerce')
    
    if df['Timestamp'].isnull().any():
        print(f"ATTENTION : certaines dates n'ont pas pu être converties dans {file_path}")
    
    df['Month'] = df['Timestamp'].dt.to_period('M')
    # Ici, on n'utilise que la moyenne des scaling factors calculée à partir des fichiers daily.
    df['Scaling Factor'] = df['Timestamp'].dt.month.map(lambda m: avg_scaling.get(m, 1))
    demand_col = get_demand_column(df)
    df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
    df['Scaled Demand (MWh)'] = df[demand_col] * df['Scaling Factor']
    return df

# --- Traitement des fichiers horaires ---
hourly_files = glob.glob(os.path.join(r"Demand DATA/2024_MM_H_SW", "*_H_SW.csv"))
if hourly_files:
    list_dfs_hourly = []
    for file in hourly_files:
        print(f"\nTraitement du fichier horaire: {file}")
        try:
            df_hourly = process_hourly_file(file, avg_scaling)
            list_dfs_hourly.append(df_hourly)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file}: {e}")
    if list_dfs_hourly:
        df_hourly_all = pd.concat(list_dfs_hourly, ignore_index=True)
        df_hourly_all.to_csv("Demand_DATA_scaled_hourly.csv", index=False)
        print("\nFichier 'Demand_DATA_scaled_hourly.csv' enregistré.")
else:
    print("Aucun fichier horaire trouvé.")

print("\nTraitement terminé.")
