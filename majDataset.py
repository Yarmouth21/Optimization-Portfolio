import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1️⃣ Chargement et préparation des données ---
data = pd.read_csv("CAC40_20actions_50mois.csv", sep=';', header=0)
data.rename(columns={data.columns[0]: 'Date'}, inplace=True)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Convertir toutes les colonnes sauf Date en float
for col in data.columns:
    if col != 'Date':
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(how='any')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# --- 2️⃣ Calcul des rendements trimestriels ---
returns = data.pct_change(90).dropna(how='any')

# --- 3️⃣ Création des features et de la cible ---
X = returns.iloc[:-1]           # rendements du trimestre courant
y = returns.shift(-1).iloc[:-1] # rendements du trimestre suivant

# --- 4️⃣ Découpage train/test ---
test_size = 200
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# --- 5️⃣ Entraînement du modèle ---
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# --- 6️⃣ Prédiction ---
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

# --- 7️⃣ Évaluation ---
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ Score R² :", round(r2, 4))
print("✅ RMSE :", round(rmse, 5))

# --- 8️⃣ Visualisation ---
actions = ['AIR.PA', 'AI.PA', 'BN.PA', 'CAP.PA', 'ACA.PA']
fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
for i, action in enumerate(actions):
    axes[i].plot(y_test.index, y_test[action], label='Rendement réel', marker='o')
    axes[i].plot(y_test.index, y_pred_df[action], label='Rendement prédit', marker='x')
    axes[i].set_title(f'Prédiction vs Réel pour {action}')
    axes[i].legend()
    axes[i].grid(True)
plt.tight_layout()
plt.show()

# --- 9️⃣ Sélection des meilleures actions ---
dernier_pred = y_pred_df.iloc[-1]
top5_actions = dernier_pred.sort_values(ascending=False).head(5)
rendements = top5_actions.values

# --- 🔟 Calcul de la matrice de covariance (sur l'historique récent) ---
# Ici, on prend les 250 derniers jours des rendements réels
cov_matrix = returns[top5_actions.index].iloc[-250:].cov().values

# --- 1️⃣1️⃣ Optimisation moyenne-variance ---
# Paramètre de risque (plus grand = plus aversion au risque)
lambda_risque = 5.0  

# Fonction objectif : - (rendement attendu - λ * risque)
def objectif(poids):
    rendement_portefeuille = np.dot(poids, rendements)
    variance_portefeuille = np.dot(poids.T, np.dot(cov_matrix, poids))
    return -(rendement_portefeuille - lambda_risque * variance_portefeuille)

# Contraintes et bornes
contraintes = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bornes = [(0.05, 0.30) for _ in range(len(top5_actions))]
x0 = np.array([1/len(top5_actions)] * len(top5_actions))

# Optimisation
result = minimize(objectif, x0, bounds=bornes, constraints=contraintes)
poids_opt = result.x

# --- 1️⃣2️⃣ Résultats ---
portefeuille_opt = pd.Series(poids_opt, index=top5_actions.index)
rendement_portefeuille = np.dot(poids_opt, rendements)
risque_portefeuille = np.sqrt(np.dot(poids_opt.T, np.dot(cov_matrix, poids_opt)))

print("\n📈 Portefeuille optimisé (moyenne-variance) :")
print(portefeuille_opt)
print("\n🔹 Rendement attendu :", round(rendement_portefeuille, 5))
print("🔹 Risque (écart-type du portefeuille) :", round(risque_portefeuille, 5))
