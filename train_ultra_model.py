import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ----- Load Dataset -----
df = pd.read_csv("fifa_players.csv")

# Drop duplicates & handle missing
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

# ----- Encode Categorical Features -----
categorical_cols = [
    'positions', 'nationality', 'preferred_foot', 'body_type', 'national_team', 'national_team_position'
]
le_dict = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# Convert birth_date to age feature (optional enhancement)
if 'birth_date' in df.columns:
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    df['age_from_birth'] = 2025 - df['birth_date'].dt.year
    df['age_from_birth'] = df['age_from_birth'].fillna(df['age'])

# ----- Define Target Variables -----
# We'll train three models: Value, Potential, and Overall Rating
target_value = 'value_euro'
target_potential = 'potential'
target_overall = 'overall_rating'

# Define features: use ALL numerical columns except IDs, names, and targets
exclude_cols = [
    'name', 'full_name', 'birth_date', target_value, target_potential, target_overall,
    'national_team', 'national_team_position', 'national_jersey_number'
]
features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

X = df[features]
y_value = df[target_value]
y_potential = df[target_potential]
y_overall = df[target_overall]

# ----- Split Data (single split for consistency across targets) -----
X_train, X_test, y_train_multi, y_test_multi = train_test_split(
    X,
    pd.DataFrame({
        'value': y_value,
        'potential': y_potential,
        'overall': y_overall,
    }),
    test_size=0.2,
    random_state=42,
)

y_train_value = y_train_multi['value']
y_test_value = y_test_multi['value']
y_train_potential = y_train_multi['potential']
y_test_potential = y_test_multi['potential']
y_train_overall = y_train_multi['overall']
y_test_overall = y_test_multi['overall']

# ----- Scale Data -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Train Ensemble Models -----
print("Training Value Prediction Model...")
value_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
value_model.fit(X_train_scaled, y_train_value)

print("Training Potential Prediction Model...")
potential_model = RandomForestRegressor(n_estimators=200, random_state=42)
potential_model.fit(X_train_scaled, y_train_potential)

print("Training Overall Rating Prediction Model...")
overall_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
overall_model.fit(X_train_scaled, y_train_overall)

# ----- Evaluate -----
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} → MAE: {mae:,.2f} | R²: {r2:.4f}")
    return preds

y_pred_value = evaluate(value_model, X_test_scaled, y_test_value, "Value Model")
y_pred_potential = evaluate(potential_model, X_test_scaled, y_test_potential, "Potential Model")
y_pred_overall = evaluate(overall_model, X_test_scaled, y_test_overall, "Overall Model")

# ----- Save Models -----
joblib.dump(value_model, "value_model.pkl")
joblib.dump(potential_model, "potential_model.pkl")
joblib.dump(overall_model, "overall_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "label_encoders.pkl")

# Persist feature names and feature means for inference defaulting
feature_artifacts = {
    'features': features,
    'feature_means': X_train.mean(numeric_only=True).to_dict(),
    'categorical_cols': [c for c in categorical_cols if c in df.columns],
}
with open('features.json', 'w', encoding='utf-8') as f:
    json.dump(feature_artifacts, f, ensure_ascii=False, indent=2)

print("\n✅ Models saved: value_model.pkl, potential_model.pkl, overall_model.pkl, scaler.pkl, label_encoders.pkl, features.json")

# ----- Visual Evaluation -----
plt.figure(figsize=(10,5))
sns.scatterplot(x=y_test_value, y=y_pred_value)
plt.xlabel("Actual Value (€)")
plt.ylabel("Predicted Value (€)")
plt.title("Player Market Value Prediction Performance")
plt.tight_layout()
plt.savefig("value_model_eval.png", dpi=150)
plt.close()
