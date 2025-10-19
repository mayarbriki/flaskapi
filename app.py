from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# ----- Run the App -----
if __name__ == '__main__':
    app.run(debug=True)
