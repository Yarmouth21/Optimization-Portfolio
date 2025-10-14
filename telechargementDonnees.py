import yfinance as yf
import pandas as pd


def appelAPI():
    start_date = "2019-06-01"
    end_date = "2023-08-31"
    interval = "1d"

    # --- 3) Télécharger les données ticker par ticker ---
    all_data = pd.DataFrame()

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
            # Garder uniquement la colonne 'Adj Close'
            df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
            all_data = pd.concat([all_data, df], axis=1)
        except Exception as e:
            print(f"Erreur pour {ticker}: {e}")

    # --- 4) Nettoyer les lignes avec NaN (mois incomplets) ---
    all_data = all_data.dropna(how='any')

    # --- 5) Afficher les premières lignes ---
    print(all_data.head())

    # --- 6) Sauvegarder en CSV ---
    all_data.to_csv("CAC40_20actions_50mois.csv", sep=';')
    print("Données sauvegardées dans CAC40_20actions_50mois.csv")

tickers = [
    "AIR.PA", "AI.PA", "BN.PA", "CAP.PA", "ACA.PA", "BNP.PA", "EN.PA",
    "EL.PA", "RMS.PA", "OR.PA", "MC.PA", "RI.PA", "SGO.PA", "GLE.PA",
    "KER.PA", "LR.PA", "SAN.PA", "SU.PA", "VIE.PA", "VIV.PA"
]
appelAPI()