from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load dataset
df = pd.read_csv("fifa_players.csv")

# Keep a copy of the original positions for filtering
df['positions_original'] = df['positions'].astype(str)

# ----- Data Preprocessing -----
# Encode categorical columns (except original positions)
categorical_cols = ['positions', 'nationality', 'preferred_foot', 'body_type']
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Normalize numeric data
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df = df.dropna(subset=numeric_cols)

# ----- API Endpoint -----
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
    recommendations = df_filtered.iloc[top_indices][[
        'name', 'nationality', 'overall_rating', 'value_euro', 'positions_original'
    ]]

    return jsonify(recommendations.to_dict(orient='records'))

# ----- Run the App -----
if __name__ == '__main__':
    app.run(debug=True)
