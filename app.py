from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import json
import io
import os
import urllib.request
import urllib.parse
import urllib.error
import html

app = Flask(__name__)

# Load dataset
raw_df = pd.read_csv("fifa_players.csv")
df = raw_df.copy()

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

def generate_ai_scout_report(player_data):
    """Simule un appel d'API IA pour générer un rapport de scoutage.

    Entrée attendue (dict):
    - name: str
    - positions/position: str
    - overall ou overall_rating: number
    - potential: number
    - top_stats: list[(stat_name, value)] (jusqu'à 10)
    """
    name = str(player_data.get('name', 'Joueur')).strip()
    position = str(player_data.get('position') or player_data.get('positions') or '').strip()
    primary_position = position.split(',')[0].strip() if position else ''

    overall = player_data.get('overall')
    if overall is None:
        overall = player_data.get('overall_rating')
    potential = player_data.get('potential')
    top_stats = player_data.get('top_stats') or []

    def norm_val(v):
        try:
            if pd.isna(v):
                return None
            v = float(v)
            return int(v) if v.is_integer() else round(v, 1)
        except Exception:
            return None

    o = norm_val(overall)
    p = norm_val(potential)

    # Synthèse (3 phrases basées sur Overall/Potential)
    synth_parts = []
    if primary_position:
        synth_parts.append(f"Profil de recrutement pour {name} ({primary_position}).")
    else:
        synth_parts.append(f"Profil de recrutement pour {name}.")
    if o is not None and p is not None:
        margin = p - o
        if margin >= 6:
            marge_txt = "une marge de progression importante"
        elif margin >= 3:
            marge_txt = "une marge de progression modérée"
        else:
            marge_txt = "une progression mesurée"
        synth_parts.append(f"Évalué à {o} d'overall avec un potentiel projeté à {p}, indiquant {marge_txt}.")
    elif o is not None:
        synth_parts.append(f"Évalué à {o} d'overall.")
    elif p is not None:
        synth_parts.append(f"Potentiel projeté à {p}.")
    else:
        synth_parts.append("Indicateurs globaux non disponibles.")
    synth_parts.append("Projection positive à court/moyen terme au regard des indicateurs clés.")

    # Points forts (3 à 5)
    strengths = []
    k = min(5, len(top_stats))
    for i, item in enumerate(top_stats[:k]):
        try:
            stat_name, stat_val = item
        except Exception:
            if isinstance(item, dict):
                stat_name = item.get('stat') or item.get('name') or 'Attribut'
                stat_val = item.get('value')
            else:
                stat_name = 'Attribut'
                stat_val = None
        pretty = str(stat_name).replace('_', ' ').title()
        v = norm_val(stat_val)
        v_str = "N/A" if v is None else str(v)
        if i == 0:
            analysis = "Impact déterminant dans les situations clés."
        elif i == 1:
            analysis = "Atout fiable et répétable au haut niveau."
        elif i == 2:
            analysis = "Contribution tangible à la structure collective."
        elif i == 3:
            analysis = "Renforce la polyvalence du profil."
        else:
            analysis = "Indicateur de constance sur la durée."
        strengths.append(f"- {pretty} {v_str}: {analysis}")

    # Conclusion
    if o is not None and p is not None:
        growth = p - o
        if growth >= 6:
            reco = "Investissement prioritaire recommandé, avec plan de développement individualisé."
        elif growth >= 3:
            reco = "Candidat sérieux à suivre de près; engagement opportuniste selon contexte budgétaire."
        else:
            reco = "Profil mature apte à contribuer rapidement selon besoin immédiat."
    else:
        reco = "Recommandation sous réserve d'observation complémentaire."

    # Assemblage du rapport
    title = f"Rapport de Scout IA - {name}"
    sep = "-" * len(title)
    report = []
    report.append(title)
    report.append(sep)
    report.append(f"Synthèse : {' '.join(synth_parts)}")
    report.append("Points Forts Clés :")
    if strengths:
        report.extend(strengths[:max(3, min(5, len(strengths)))])
    else:
        report.append("- Données insuffisantes pour extraire des points forts.")
    report.append(f"Conclusion et Potentiel : {reco}")

    return "\n".join(report)

def _chunk_text(text: str, max_chars: int = 450) -> list[str]:
    """Split text into chunks under max_chars, preferring newline/space boundaries."""
    if text is None:
        return ['']
    s = str(text)
    if len(s) <= max_chars:
        return [s]
    chunks = []
    current = []
    current_len = 0
    # Prefer splitting on lines first
    for line in s.split('\n'):
        line = line.rstrip('\r')
        if not line:
            if current_len + 1 <= max_chars:
                current.append('')
                current_len += 1
            else:
                chunks.append('\n'.join(current))
                current = ['']
                current_len = 1
            continue
        if current_len + len(line) + (1 if current else 0) <= max_chars:
            current.append(line)
            current_len += len(line) + (1 if len(current) > 1 else 0)
        else:
            # If line itself is too long, split on spaces
            if len(line) > max_chars:
                words = line.split(' ')
                buf = ''
                for w in words:
                    if not buf:
                        buf = w
                    elif len(buf) + 1 + len(w) <= max_chars:
                        buf += ' ' + w
                    else:
                        # flush buf
                        if current_len + len(buf) + (1 if current else 0) <= max_chars:
                            current.append(buf)
                            current_len += len(buf) + (1 if len(current) > 1 else 0)
                        else:
                            chunks.append('\n'.join(current))
                            current = [buf]
                            current_len = len(buf)
                        buf = w
                if buf:
                    if current_len + len(buf) + (1 if current else 0) <= max_chars:
                        current.append(buf)
                        current_len += len(buf) + (1 if len(current) > 1 else 0)
                    else:
                        chunks.append('\n'.join(current))
                        current = [buf]
                        current_len = len(buf)
            else:
                # flush current as a chunk and start a new one with this line
                if current:
                    chunks.append('\n'.join(current))
                current = [line]
                current_len = len(line)
    if current:
        chunks.append('\n'.join(current))
    return chunks

def _translate_try_providers(text_to_translate, target_language):
    base = '' if text_to_translate is None else str(text_to_translate)
    lang_norm = str(target_language or '').strip().lower()
    if lang_norm in ('french', 'français', 'francais', 'fr'):
        return base, 'passthrough'
    lang_map = {
        'english': 'en', 'en': 'en',
        'spanish': 'es', 'es': 'es', 'español': 'es',
        'german': 'de', 'de': 'de', 'deutsch': 'de',
        'italian': 'it', 'it': 'it', 'italiano': 'it',
        'portuguese': 'pt', 'pt': 'pt', 'português': 'pt',
        'dutch': 'nl', 'nl': 'nl', 'nederlands': 'nl',
        'turkish': 'tr', 'tr': 'tr', 'türkçe': 'tr',
        'arabic': 'ar', 'ar': 'ar', 'العربية': 'ar',
        'french': 'fr', 'français': 'fr', 'francais': 'fr', 'fr': 'fr',
    }
    target_code = lang_map.get(lang_norm)
    if not target_code:
        return f"[Traduit en {target_language}] - " + base, 'simulation'
    if target_code == 'fr':
        return base, 'passthrough'
    # LibreTranslate attempts (chunked)
    candidates = []
    env_url = os.environ.get('LIBRE_TRANSLATE_URL', '').strip()
    if env_url:
        candidates.append(env_url)
    candidates.extend([
        'https://translate.argosopentech.com',
        'https://libretranslate.de',
        'https://libretranslate.com',
    ])
    for base_url in candidates:
        try:
            libre_url = base_url.rstrip('/') + '/translate'
            parts = _chunk_text(base, max_chars=1800)
            translated_parts = []
            ok = True
            for part in parts:
                libre_payload = {
                    'q': part,
                    'source': 'fr',
                    'target': target_code,
                    'format': 'text',
                }
                libre_api_key = os.environ.get('LIBRE_TRANSLATE_API_KEY')
                if libre_api_key:
                    libre_payload['api_key'] = libre_api_key
                req = urllib.request.Request(
                    libre_url,
                    data=json.dumps(libre_payload).encode('utf-8'),
                    headers={ 'Content-Type': 'application/json' }
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = resp.read().decode('utf-8', errors='ignore')
                data_lt = json.loads(body)
                translated = data_lt.get('translatedText') or data_lt.get('translation')
                if isinstance(translated, str) and translated.strip():
                    translated_parts.append(translated)
                else:
                    ok = False
                    break
            if ok and translated_parts:
                return '\n'.join(translated_parts), f'libretranslate:{base_url}'
        except Exception:
            continue
    # MyMemory (chunked, 500 chars limit)
    try:
        parts = _chunk_text(base, max_chars=480)
        out_parts = []
        for part in parts:
            mm_url = 'https://api.mymemory.translated.net/get?'+ urllib.parse.urlencode({
                'q': part,
                'langpair': f'fr|{target_code}',
            })
            with urllib.request.urlopen(mm_url, timeout=10) as resp:
                body = resp.read().decode('utf-8', errors='ignore')
            data_mm = json.loads(body)
            status = int(data_mm.get('responseStatus', 500))
            if status != 200:
                raise RuntimeError(f"MyMemory status {status}")
            translated = (data_mm.get('responseData') or {}).get('translatedText')
            if not isinstance(translated, str) or not translated.strip():
                raise RuntimeError("MyMemory empty chunk")
            out_parts.append(html.unescape(translated))
        if out_parts:
            return '\n'.join(out_parts), 'mymemory'
    except Exception:
        pass
    # Google
    api_key = os.environ.get('GOOGLE_TRANSLATE_API_KEY')
    if api_key:
        try:
            url = "https://translation.googleapis.com/language/translate/v2?key=" + urllib.parse.quote(api_key)
            payload = {
                "q": base[:4500],
                "target": target_code,
                "format": "text",
                "source": "fr",
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={ 'Content-Type': 'application/json' }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode('utf-8', errors='ignore')
            data = json.loads(body)
            translations = data.get('data', {}).get('translations', [])
            if translations:
                translated = translations[0].get('translatedText', '')
                return html.unescape(translated), 'google'
        except Exception:
            pass
    return f"[Traduit en {target_language}] - " + base, 'simulation'

def translate_text_ai(text_to_translate, target_language):
    text, _provider = _translate_try_providers(text_to_translate, target_language)
    return text

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

@app.route('/api/players', methods=['GET'])
def api_players():
    """Return a filtered, paginated list of players from the CSV dataset.

    Query params (all optional):
    - nationality: str (case-insensitive exact match on original column)
    - position: str (token contained in original positions list, case-insensitive)
    - min_overall, max_overall: ints (use original overall_rating)
    - min_age, max_age: ints
    - min_value, max_value: numbers (value_euro original)
    - q: substring on name (case-insensitive)
    - page: 1-based page index (default 1)
    - page_size: items per page (default 20, max 100)
    - sort: one of name, overall_rating, value_euro, age (prefix '-' for descending)
    """
    # Work on a copy for filtering
    data = df.copy()

    # Ensure original display columns exist
    if 'nationality_original' not in data.columns and 'nationality' in data.columns:
        data['nationality_original'] = data['nationality'].astype(str)
    if 'positions_original' not in data.columns and 'positions' in data.columns:
        data['positions_original'] = data['positions'].astype(str)
    if 'overall_rating_original' not in data.columns and 'overall_rating' in data.columns:
        data['overall_rating_original'] = data['overall_rating']
    if 'value_euro_original' not in data.columns and 'value_euro' in data.columns:
        data['value_euro_original'] = data['value_euro']

    # Text query on name
    q = (request.args.get('q') or '').strip().lower()
    if q:
        if 'name' in data.columns:
            data = data[data['name'].astype(str).str.lower().str.contains(q, na=False)]

    # Nationality exact (case-insensitive) on original column if available
    nat = (request.args.get('nationality') or '').strip()
    if nat and 'nationality_original' in data.columns:
        data = data[data['nationality_original'].astype(str).str.lower() == nat.lower()]

    # Position token contains in original positions column
    pos = (request.args.get('position') or '').strip()
    if pos and 'positions_original' in data.columns:
        token = pos.strip().upper()
        data = data[data['positions_original'].astype(str).str.upper().str.contains(token, na=False)]

    # Numeric range filters
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return None
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    min_overall = _to_int(request.args.get('min_overall'))
    max_overall = _to_int(request.args.get('max_overall'))
    if 'overall_rating_original' in data.columns:
        if min_overall is not None:
            data = data[data['overall_rating_original'] >= min_overall]
        if max_overall is not None:
            data = data[data['overall_rating_original'] <= max_overall]

    min_age = _to_int(request.args.get('min_age'))
    max_age = _to_int(request.args.get('max_age'))
    if 'age' in data.columns:
        if min_age is not None:
            data = data[data['age'] >= min_age]
        if max_age is not None:
            data = data[data['age'] <= max_age]

    min_value = _to_float(request.args.get('min_value'))
    max_value = _to_float(request.args.get('max_value'))
    if 'value_euro_original' in data.columns:
        if min_value is not None:
            data = data[data['value_euro_original'] >= min_value]
        if max_value is not None:
            data = data[data['value_euro_original'] <= max_value]

    # Sorting
    sort = (request.args.get('sort') or '').strip()
    sort_map = {
        'name': 'name',
        'overall_rating': 'overall_rating_original',
        'value_euro': 'value_euro_original',
        'age': 'age',
    }
    ascending = True
    if sort.startswith('-'):
        ascending = False
        sort = sort[1:]
    sort_col = sort_map.get(sort)
    if sort_col and sort_col in data.columns:
        try:
            data = data.sort_values(by=sort_col, ascending=ascending, kind='mergesort')
        except Exception:
            pass

    # Pagination
    try:
        page = max(1, int(request.args.get('page', 1)))
    except Exception:
        page = 1
    try:
        page_size = int(request.args.get('page_size', 20))
    except Exception:
        page_size = 20
    page_size = max(1, min(page_size, 100))

    total = int(len(data))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = data.iloc[start:end]

    cols = []
    for c in ['name', 'nationality_original', 'overall_rating_original', 'value_euro_original', 'positions_original', 'age']:
        if c in page_df.columns:
            cols.append(c)
    out = page_df[cols].rename(columns={
        'nationality_original': 'nationality',
        'overall_rating_original': 'overall_rating',
        'value_euro_original': 'value_euro',
    })

    return jsonify({
        'items': out.to_dict(orient='records'),
        'page': page,
        'page_size': page_size,
        'total': total,
        'total_pages': (total + page_size - 1) // page_size
    })

@app.route('/api/players/<string:name>', methods=['GET'])
def api_player_detail(name: str):
    """Return detailed information for a player by name.

    Exact case-insensitive match is attempted first; if not found, falls back to substring contains.
    """
    if 'name' not in raw_df.columns:
        return jsonify({ 'error': 'Column "name" missing in dataset.' }), 500

    names = raw_df['name'].astype(str)
    target = (name or '').strip()
    if not target:
        return jsonify({ 'error': 'Empty player name.' }), 400

    # exact match (case-insensitive)
    exact_mask = names.str.strip().str.lower() == target.lower()
    candidates = raw_df[exact_mask]
    if candidates.empty:
        contains_mask = names.str.lower().str.contains(target.lower(), na=False)
        candidates = raw_df[contains_mask]
    if candidates.empty:
        return jsonify({ 'error': f'Player "{target}" not found.' }), 404

    row = candidates.iloc[0]
    record = {}
    for col in raw_df.columns:
        try:
            val = row[col]
            if pd.isna(val):
                record[col] = None
            else:
                # Convert numpy types to native python
                if hasattr(val, 'item'):
                    record[col] = val.item()
                else:
                    record[col] = val
        except Exception:
            record[col] = None
    return jsonify(record)

@app.route('/api/stats/top', methods=['GET'])
def api_stats_top():
    """Return top-N players by a numeric metric with optional filters.

    Query params:
    - stat: column name (default 'overall_rating')
    - n: number of rows (default 10, max 100)
    - position: filter by token contained in positions (original)
    - nationality: exact case-insensitive match on nationality (original)
    """
    if raw_df is None or raw_df.empty:
        return jsonify({ 'items': [], 'n': 0 }), 200

    stat = (request.args.get('stat') or 'overall_rating').strip()
    try:
        n = int(request.args.get('n', 10))
    except Exception:
        n = 10
    n = max(1, min(n, 100))

    data = raw_df.copy()

    # Filters using original columns if available
    pos = (request.args.get('position') or '').strip()
    if pos and 'positions' in data.columns:
        token = pos.strip().upper()
        data = data[data['positions'].astype(str).str.upper().str.contains(token, na=False)]

    nat = (request.args.get('nationality') or '').strip()
    if nat and 'nationality' in data.columns:
        data = data[data['nationality'].astype(str).str.lower() == nat.lower()]

    # Validate stat column
    if stat not in data.columns:
        return jsonify({ 'error': f"Unknown stat '{stat}'" }), 400

    # Keep rows with numeric values
    try:
        ser = pd.to_numeric(data[stat], errors='coerce')
        data = data.loc[ser.notna()].copy()
        data['_stat_value'] = ser[ser.notna()]
    except Exception as e:
        return jsonify({ 'error': f"Stat '{stat}' is not numeric or cannot be parsed: {e}" }), 400

    # Sort and take top-N
    data = data.sort_values(by='_stat_value', ascending=False)
    top = data.head(n)

    cols = []
    for c in ['name', 'nationality', 'overall_rating', 'value_euro', 'positions']:
        if c in top.columns:
            cols.append(c)
    out = top[cols + ['_stat_value']].rename(columns={'_stat_value': 'stat_value'})

    return jsonify({
        'items': out.to_dict(orient='records'),
        'stat': stat,
        'n': int(len(out))
    })

@app.route('/api/compare', methods=['GET'])
def api_compare():
    """Compare selected players across selected metrics.

    Params:
    - players: comma-separated names (at least 2)
    - metrics: comma-separated column names (defaults to a safe set)
    Returns list of { name, <metric>: value, ... }
    """
    players_param = (request.args.get('players') or '').strip()
    if not players_param:
        return jsonify({ 'error': 'Missing players parameter' }), 400
    names = [p.strip() for p in players_param.split(',') if p.strip()]
    names = list(dict.fromkeys(names))[:10]
    if len(names) < 1:
        return jsonify({ 'error': 'No valid player names' }), 400

    default_metrics = ['overall_rating','potential','sprint_speed','acceleration','dribbling','short_passing','finishing','strength','stamina']
    metrics_param = (request.args.get('metrics') or '').strip()
    metrics = [m.strip() for m in metrics_param.split(',') if m.strip()] if metrics_param else default_metrics
    metrics = [m for m in metrics if m in raw_df.columns][:20]
    if not metrics:
        return jsonify({ 'error': 'No valid metrics' }), 400

    out = []
    for n in names:
        mask = raw_df['name'].astype(str).str.strip().str.lower() == n.lower()
        rows = raw_df[mask]
        if rows.empty:
            # try contains as fallback
            mask = raw_df['name'].astype(str).str.lower().str.contains(n.lower(), na=False)
            rows = raw_df[mask]
        if rows.empty:
            continue
        row = rows.iloc[0]
        rec = { 'name': str(row.get('name', n)) }
        for m in metrics:
            val = row.get(m, None)
            try:
                if pd.isna(val):
                    rec[m] = None
                else:
                    rec[m] = float(val) if isinstance(val, (int, float, np.number)) else val
            except Exception:
                rec[m] = None
        out.append(rec)
    if not out:
        return jsonify({ 'error': 'No players found' }), 404
    return jsonify({ 'items': out, 'metrics': metrics })

@app.route('/api/anomalies', methods=['POST'])
def api_anomalies():
    """Detect anomalies using IsolationForest on numeric columns.

    Body JSON optional keys: { n: int (default 20), position: str, nationality: str }
    Returns items with name, score, and selected context fields.
    """
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}
    n = payload.get('n', 20)
    try:
        n = int(n)
    except Exception:
        n = 20
    n = max(1, min(n, 200))

    data = raw_df.copy()
    pos = str(payload.get('position') or '').strip()
    if pos and 'positions' in data.columns:
        token = pos.upper()
        data = data[data['positions'].astype(str).str.upper().str.contains(token, na=False)]
    nat = str(payload.get('nationality') or '').strip()
    if nat and 'nationality' in data.columns:
        data = data[data['nationality'].astype(str).str.lower() == nat.lower()]

    # numeric columns
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    # drop obvious identifiers
    drop_cols = {'sofifa_id','player_id','id'}
    num_cols = [c for c in num_cols if c not in drop_cols]
    if not num_cols:
        return jsonify({ 'items': [], 'n': 0 })

    X = data[num_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean(numeric_only=True))
    try:
        iso = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
        scores = iso.fit_predict(X)
        # lower score -> more anomalous in score_samples
        score_vals = iso.score_samples(X)
        data = data.assign(_anomaly_score=(-score_vals))
        top = data.sort_values(by='_anomaly_score', ascending=False).head(n)
    except Exception as e:
        return jsonify({ 'error': f'Anomaly detection failed: {e}' }), 500

    cols = []
    for c in ['name','nationality','positions','overall_rating','value_euro','age']:
        if c in top.columns:
            cols.append(c)
    out = top[cols + ['_anomaly_score']].rename(columns={'_anomaly_score': 'anomaly_score'})
    return jsonify({ 'items': out.to_dict(orient='records'), 'n': int(len(out)) })

@app.route('/api/players/export', methods=['POST'])
def api_players_export():
    """Export filtered players as CSV using same filters as /api/players.

    Accepts JSON body with the same filter keys.
    """
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    # Reuse similar logic as api_players but without pagination
    data = df.copy()
    if 'nationality_original' not in data.columns and 'nationality' in data.columns:
        data['nationality_original'] = data['nationality'].astype(str)
    if 'positions_original' not in data.columns and 'positions' in data.columns:
        data['positions_original'] = data['positions'].astype(str)
    if 'overall_rating_original' not in data.columns and 'overall_rating' in data.columns:
        data['overall_rating_original'] = data['overall_rating']
    if 'value_euro_original' not in data.columns and 'value_euro' in data.columns:
        data['value_euro_original'] = data['value_euro']

    q = str(payload.get('q') or '').strip().lower()
    if q and 'name' in data.columns:
        data = data[data['name'].astype(str).str.lower().str.contains(q, na=False)]

    nat = str(payload.get('nationality') or '').strip()
    if nat and 'nationality_original' in data.columns:
        data = data[data['nationality_original'].astype(str).str.lower() == nat.lower()]

    pos = str(payload.get('position') or '').strip()
    if pos and 'positions_original' in data.columns:
        token = pos.upper()
        data = data[data['positions_original'].astype(str).str.upper().str.contains(token, na=False)]

    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return None
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    min_overall = _to_int(payload.get('min_overall'))
    max_overall = _to_int(payload.get('max_overall'))
    if 'overall_rating_original' in data.columns:
        if min_overall is not None:
            data = data[data['overall_rating_original'] >= min_overall]
        if max_overall is not None:
            data = data[data['overall_rating_original'] <= max_overall]

    min_age = _to_int(payload.get('min_age'))
    max_age = _to_int(payload.get('max_age'))
    if 'age' in data.columns:
        if min_age is not None:
            data = data[data['age'] >= min_age]
        if max_age is not None:
            data = data[data['age'] <= max_age]

    min_value = _to_float(payload.get('min_value'))
    max_value = _to_float(payload.get('max_value'))
    if 'value_euro_original' in data.columns:
        if min_value is not None:
            data = data[data['value_euro_original'] >= min_value]
        if max_value is not None:
            data = data[data['value_euro_original'] <= max_value]

    cols = []
    for c in ['name','nationality_original','overall_rating_original','value_euro_original','positions_original','age']:
        if c in data.columns:
            cols.append(c)
    out = data[cols].rename(columns={
        'nationality_original': 'nationality',
        'overall_rating_original': 'overall_rating',
        'value_euro_original': 'value_euro',
    })

    # to CSV
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    csv_text = buf.getvalue()
    resp = app.response_class(csv_text, mimetype='text/csv')
    resp.headers['Content-Disposition'] = 'attachment; filename=players_export.csv'
    return resp

@app.route('/api/stats/histogram', methods=['GET'])
def api_stats_histogram():
    """Histogram for a numeric metric.

    Params: stat, bins (default 20), optional position/nationality filters.
    Returns: { counts: [], bins: [] }
    """
    stat = (request.args.get('stat') or 'overall_rating').strip()
    try:
        bins = int(request.args.get('bins', 20))
    except Exception:
        bins = 20
    bins = max(2, min(bins, 200))

    data = raw_df.copy()
    pos = (request.args.get('position') or '').strip()
    if pos and 'positions' in data.columns:
        token = pos.upper()
        data = data[data['positions'].astype(str).str.upper().str.contains(token, na=False)]
    nat = (request.args.get('nationality') or '').strip()
    if nat and 'nationality' in data.columns:
        data = data[data['nationality'].astype(str).str.lower() == nat.lower()]

    if stat not in data.columns:
        return jsonify({ 'error': f"Unknown stat '{stat}'" }), 400
    ser = pd.to_numeric(data[stat], errors='coerce').dropna()
    if ser.empty:
        return jsonify({ 'counts': [], 'bins': [] })
    counts, bin_edges = np.histogram(ser.values, bins=bins)
    return jsonify({ 'counts': counts.tolist(), 'bins': bin_edges.tolist(), 'stat': stat })

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

@app.route('/api/scout-report', methods=['POST'])
def api_scout_report():
    """Génère un rapport de scoutage IA pour un joueur donné.

    Corps JSON attendu: { "player_name": "L. Messi" }
    Réponse: { "report": "..." }
    """
    try:
        payload = request.get_json(silent=True) or {}
        player_name = (payload.get('player_name') or '').strip()
        if not player_name:
            return jsonify({ 'error': 'Corps JSON invalide. Attendu: {"player_name": "Nom"}'}), 400

        if 'name' not in raw_df.columns:
            return jsonify({ 'error': 'La colonne "name" est absente du dataset.' }), 500

        names = raw_df['name'].astype(str)
        exact_mask = names.str.strip().str.lower() == player_name.lower()
        candidates = raw_df[exact_mask]
        if candidates.empty:
            contains_mask = names.str.lower().str.contains(player_name.lower(), na=False)
            candidates = raw_df[contains_mask]
        if candidates.empty:
            return jsonify({ 'error': f'Joueur "{player_name}" introuvable.' }), 404

        row = candidates.iloc[0]

        # Champs principaux
        name = str(row.get('name', player_name))
        positions_val = row.get('positions', None)
        if positions_val is None and 'position' in raw_df.columns:
            positions_val = row.get('position', None)

        def pick(*keys):
            for k in keys:
                if k in raw_df.columns:
                    return row.get(k, None)
            return None

        overall = pick('overall_rating', 'overall')
        potential = pick('potential', 'potential_rating')

        # Top 10 statistiques numériques
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {
            'overall_rating','overall','potential',
            'value_euro','wage_euro','release_clause_euro',
            'sofifa_id','player_id','id','age','height_cm','weight_kg'
        }
        top_candidates = []
        for c in numeric_cols:
            if c in exclude:
                continue
            try:
                val = row[c]
            except Exception:
                continue
            if pd.isna(val):
                continue
            try:
                top_candidates.append((c, float(val)))
            except Exception:
                continue
        top_candidates.sort(key=lambda x: x[1], reverse=True)
        top_stats = top_candidates[:10]

        player_data = {
            'name': name,
            'positions': positions_val,
            'overall_rating': overall,
            'potential': potential,
            'top_stats': top_stats,
        }

        report = generate_ai_scout_report(player_data)
        return jsonify({ 'report': report }), 200
    except Exception as e:
        return jsonify({ 'error': f"Échec de génération du rapport: {e}" }), 500

@app.route('/api/translate-report', methods=['POST'])
def api_translate_report():
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get('text', None)
        target_lang = payload.get('target_lang', None)
        if text is None or not isinstance(target_lang, str) or not target_lang.strip():
            return jsonify({ 'error': 'Champs manquants: "text" et/ou "target_lang"' }), 400
        translated, provider = _translate_try_providers(text, target_lang)
        return jsonify({ 'translated_text': translated, 'provider': provider }), 200
    except Exception as e:
        return jsonify({ 'error': f"Échec de traduction: {e}" }), 500

# ----- Run the App -----
if __name__ == '__main__':
    app.run(debug=True)
