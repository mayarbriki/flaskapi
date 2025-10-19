from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("fifa_players.csv")

# Keep copies of original fields for display
if 'positions' in df.columns:
    df['positions_original'] = df['positions'].astype(str)
if 'nationality' in df.columns:
    df['nationality_original'] = df['nationality'].astype(str)
if 'overall_rating' in df.columns:
    df['overall_rating_original'] = df['overall_rating']
if 'value_euro' in df.columns:
    df['value_euro_original'] = df['value_euro']

# ----- Data Preprocessing -----
# Encode categorical columns (except original positions)
categorical_cols = ['positions', 'nationality', 'preferred_foot', 'body_type']
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle missing values
# Use only real numeric feature columns (exclude encoded categoricals and *_original copies)
numeric_cols_all = df.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = [
    c for c in numeric_cols_all
    if not c.endswith('_original') and c not in categorical_cols
]
if numeric_cols:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Normalize numeric data
if numeric_cols:
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df = df.dropna(subset=numeric_cols)

# ----- Lazy-loaded ML Artifacts for Overall Prediction -----
_overall_model = None
_overall_scaler = None
_label_encoders = None
_feature_artifacts = None

def _load_overall_artifacts():
    global _overall_model, _overall_scaler, _label_encoders, _feature_artifacts
    if _overall_model is None:
        if not os.path.exists('overall_model.pkl'):
            raise FileNotFoundError('overall_model.pkl not found. Please run train_ultra_model.py first.')
        _overall_model = joblib.load('overall_model.pkl')
    if _overall_scaler is None:
        if not os.path.exists('scaler.pkl'):
            raise FileNotFoundError('scaler.pkl not found. Please run train_ultra_model.py first.')
        _overall_scaler = joblib.load('scaler.pkl')
    if _label_encoders is None:
        if not os.path.exists('label_encoders.pkl'):
            raise FileNotFoundError('label_encoders.pkl not found. Please run train_ultra_model.py first.')
        _label_encoders = joblib.load('label_encoders.pkl')
    if _feature_artifacts is None:
        if not os.path.exists('features.json'):
            raise FileNotFoundError('features.json not found. Please run train_ultra_model.py first.')
        with open('features.json', 'r', encoding='utf-8') as f:
            _feature_artifacts = json.load(f)

def _encode_with_fallback(le: LabelEncoder, value, feat_name: str | None = None):
    s = str(value) if value is not None else ''
    # Try exact match first (case-insensitive)
    try:
        classes = list(le.classes_)
        for c in classes:
            if c.lower() == s.lower():
                return int(le.transform([c])[0])
        # Try direct transform (may work if exact string)
        return int(le.transform([s])[0])
    except Exception:
        pass

    # Heuristic specifically for positions: allow matching tokens inside multi-position strings
    if feat_name == 'positions' and isinstance(value, str) and hasattr(le, 'classes_'):
        token = s.upper().strip()
        for c in le.classes_:
            parts = [p.strip().upper() for p in str(c).split(',')]
            if token in parts or token in str(c).upper():
                try:
                    return int(le.transform([c])[0])
                except Exception:
                    continue

    # Fallback to first known class to keep value in-range
    return int(le.transform([le.classes_[0]])[0])

def _prepare_input_row(payload: dict):
    """Prepare a single-row DataFrame matching training features and scaling.

    - Applies label encoders to categorical columns with fallback.
    - Fills missing numeric features with training means.
    - Orders columns per training feature list.
    """
    _load_overall_artifacts()
    features = _feature_artifacts.get('features', [])
    feature_means = _feature_artifacts.get('feature_means', {})
    cat_cols = set(_feature_artifacts.get('categorical_cols', []))

    row = {}
    for feat in features:
        val = payload.get(feat, None)
        if feat in _label_encoders and feat in cat_cols:
            if val is None or str(val).lower() in ('', 'nan', 'none'):
                # use first class as default
                row[feat] = _encode_with_fallback(_label_encoders[feat], _label_encoders[feat].classes_[0], feat)
            else:
                row[feat] = _encode_with_fallback(_label_encoders[feat], val, feat)
        else:
            # numeric path
            if val is None:
                val = feature_means.get(feat, 0.0)
            try:
                row[feat] = float(val)
            except Exception:
                row[feat] = float(feature_means.get(feat, 0.0))
    X = pd.DataFrame([row])[features]
    Xs = _overall_scaler.transform(X)
    return Xs

# ----- API Endpoint -----
@app.route('/', methods=['GET'])
def index():
    """Serve a simple frontend UI."""
    return render_template('index.html')

@app.route('/players', methods=['GET'])
def players():
    """Lightweight player name suggestions with optional query filtering.

    Query params:
    - q: substring filter (case-insensitive)
    - limit: max number of results (default 20, caps at 100)
    """
    q = (request.args.get('q') or '').strip().lower()
    try:
        limit = int(request.args.get('limit', 100))
    except ValueError:
        limit = 100
    limit = max(1, min(limit, 100))

    names_series = df['name'].dropna().astype(str)
    unique_names = names_series.unique().tolist()
    if q:
        filtered = [n for n in unique_names if q in n.lower()]
    else:
        filtered = unique_names
    filtered = sorted(filtered)[:limit]
    return jsonify(filtered)

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Example usage:
    http://127.0.0.1:5000/recommend?player=Cristiano%20Ronaldo
    Optional filters:
    &position=ST  or  &position=LW  or  &position=RW
    """
    player_name = request.args.get('player')
    desired_position = request.args.get('position')

    # Validate player
    if player_name not in df['name'].values:
        return jsonify({'error': f'Player \"{player_name}\" not found.'}), 404

    # Get player index
    player_index = df[df['name'] == player_name].index[0]

    # Use original text positions for filtering
    if desired_position:
        df_filtered = df[df['positions_original'].str.upper().str.contains(desired_position.upper(), na=False)]
        if df_filtered.empty:
            return jsonify({'error': f'No players found for position \"{desired_position}\".'}), 404
    else:
        df_filtered = df

    # Compute similarity ONLY for this player
    player_vector = df.loc[[player_index], numeric_cols]
    all_vectors = df_filtered[numeric_cols]

    similarities = cosine_similarity(player_vector, all_vectors)[0]

    # Get top 5 most similar players
    top_indices = np.argsort(similarities)[::-1][1:6]
    cols_for_output = []
    for c in ['name', 'nationality_original', 'overall_rating_original', 'value_euro_original', 'positions_original']:
        if c in df_filtered.columns:
            cols_for_output.append(c)

    recommendations = df_filtered.iloc[top_indices][cols_for_output]
    # Rename originals back to expected names for the frontend
    recommendations = recommendations.rename(columns={
        'nationality_original': 'nationality',
        'overall_rating_original': 'overall_rating',
        'value_euro_original': 'value_euro',
    })

    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/metadata', methods=['GET'])
def metadata():
    """Return options and hints for frontend forms."""
    # Positions: extract atomic tokens from original column
    positions_set = set()
    if 'positions_original' in df.columns:
        ser = df['positions_original'].dropna().astype(str)
        for s in ser:
            for tok in s.split(','):
                t = tok.strip().upper()
                if t:
                    positions_set.add(t)
    positions = sorted(list(positions_set))

    # Preferred foot options
    feet = ['Right', 'Left']
    if _label_encoders is not None and 'preferred_foot' in _label_encoders:
        feet = sorted(list(_label_encoders['preferred_foot'].classes_))

    # Body types
    body_types = []
    if 'body_type' in df.columns:
        body_types = sorted(set(df['body_type'].dropna().astype(str)))

    # Nationalities: top 100 by frequency for datalist
    nat_top = []
    if 'nationality_original' in df.columns:
        nat_top = (
            df['nationality_original']
            .dropna()
            .astype(str)
            .value_counts()
            .head(100)
            .index
            .tolist()
        )
    elif 'nationality' in df.columns:
        nat_top = sorted(set(df['nationality'].dropna().astype(str)))[:100]

    return jsonify({
        'positions': positions,
        'preferred_foot': feet,
        'body_type': body_types,
        'nationality_top': nat_top,
    })

@app.route('/predict/overall', methods=['POST'])
def predict_overall():
    """Predict current overall rating from player attributes.

    Request JSON body: keys should correspond to training feature names.
    Missing numeric features are filled with training means; missing categoricals default to a valid known class.
    Response: { "overall": float, "overall_rounded": int }
    """
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({ 'error': 'Invalid JSON: expected an object with feature keys.' }), 400

        Xs = _prepare_input_row(payload)
        pred = float(_overall_model.predict(Xs)[0])
        return jsonify({ 'overall': pred, 'overall_rounded': int(round(pred)) })
    except FileNotFoundError as e:
        return jsonify({ 'error': str(e) }), 500
    except Exception as e:
        return jsonify({ 'error': f'Prediction failed: {e}' }), 500

# ----- Run the App -----
if __name__ == '__main__':
    app.run(debug=True)
