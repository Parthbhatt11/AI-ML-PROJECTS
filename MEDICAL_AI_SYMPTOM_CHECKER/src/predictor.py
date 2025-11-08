import pandas as pd
import pickle
import numpy as np
from difflib import SequenceMatcher

# ===================================================
# 🔹 Load Trained Assets
# ===================================================
with open('models/custom_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('models/symptom_columns.pkl', 'rb') as f:
    all_symptoms = pickle.load(f)

df = pd.read_csv('data/medical_dataset_cleaned.csv')

# ===================================================
# ⚖️ Symptom Severity Weights
# ===================================================
severity_weight = {
    "cough": 0.9,
    "runny nose": 0.8,
    "sore throat": 0.9,
    "sneezing": 0.8,
    "headache": 1.0,
    "fatigue": 1.0,
    "fever": 1.2,
    "high fever": 1.3,
    "vomiting": 1.3,
    "nausea": 1.2,
    "body ache": 1.1,
    "chest pain": 1.6,
    "shortness of breath": 1.6,
    "severe cough": 1.4,
    "bloody sputum": 1.7,
    "chronic cough": 1.5,
}

# ===================================================
# 🔹 Helper Functions
# ===================================================
def symptom_similarity(s1, s2):
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def get_symptom_vector(symptoms):
    vector = np.zeros(len(all_symptoms))
    for s in symptoms:
        for i, known in enumerate(all_symptoms):
            if symptom_similarity(s, known) > 0.8:
                vector[i] = severity_weight.get(s.lower(), 1.0)
    return np.array([vector])

def classify_confidence(score):
    if score >= 75:
        return "High", "#28a745"
    elif score >= 45:
        return "Moderate", "#ffc107"
    else:
        return "Low", "#dc3545"

# ===================================================
# 🧠 Main Prediction Function
# ===================================================
def predict_disease(symptoms):
    symptoms = [s.strip().lower() for s in symptoms if s.strip()]
    if not symptoms:
        return [{"error": "Please enter at least one symptom."}]

    # Step 1: Model Probability
    vector = get_symptom_vector(symptoms)[0]
    X_test = pd.DataFrame([vector], columns=all_symptoms)
    try:
        model_probs = model.predict_proba(X_test)[0]
    except Exception:
        model_probs = np.ones(len(model.classes_)) / len(model.classes_)
    model_probs = np.clip(model_probs, 0.05, 0.85)
    model_probs = model_probs / model_probs.sum()
    diseases = encoder.inverse_transform(model.classes_)
    results = pd.DataFrame({'Disease': diseases, 'Model_Prob': model_probs})

    # Step 2: Match Logic
    def calc_combined_match(row_symptoms):
        if not isinstance(row_symptoms, str):
            return 0, []
        row_set = {s.strip().lower() for s in row_symptoms.split(';')}
        matched = [s for s in symptoms if any(symptom_similarity(s, rs) > 0.8 for rs in row_set)]
        ratio = len(matched) / len(symptoms)
        penalty = (len(symptoms) - len(matched)) * 0.1
        return round(max(0, ratio - penalty), 3), matched

    match_data = df['Symptoms'].apply(calc_combined_match)
    df['match_ratio'] = [x[0] for x in match_data]
    df['matched_symptoms'] = [x[1] for x in match_data]
    df['matched_symptoms_str'] = df['matched_symptoms'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
    )

    # Step 3: Safe Merge
    merged = results.merge(
        df[['Disease', 'match_ratio', 'matched_symptoms_str']].drop_duplicates(subset='Disease'),
        on='Disease', how='left'
    )

    merged['Final_Score'] = (merged['Model_Prob'] * 0.6) + (merged['match_ratio'].fillna(0) * 0.4)

    # Step 4: Mild vs Severe Logic
    mild = {"cough", "runny nose", "sneezing", "sore throat"}
    severe = {"high fever", "chest pain", "shortness of breath", "bloody sputum"}
    has_mild = any(s in mild for s in symptoms)
    has_severe = any(s in severe for s in symptoms)

    if has_mild and not has_severe:
        merged['Final_Score'] *= merged['Disease'].str.lower().apply(
            lambda d: 1.3 if any(k in d for k in ["common cold", "seasonal allergy", "rhinitis"]) else 0.7
        )
    elif has_severe:
        merged['Final_Score'] *= merged['Disease'].str.lower().apply(
            lambda d: 1.3 if any(k in d for k in ["pneumonia", "covid", "tuberculosis"]) else 1.0
        )

    # Step 5: Normalize
    max_score = merged['Final_Score'].max()
    if max_score > 0:
        merged['Final_Score'] /= max_score

    merged = merged.sort_values('Final_Score', ascending=False).head(5)

    # Step 6: Prepare Output
    predictions = []
    for _, row in merged.iterrows():
        info = df[df['Disease'].str.lower() == row['Disease'].lower()]
        if info.empty:
            continue
        info = info.iloc[0]
        precautions = [p.strip() for p in str(info['Precautions']).split(';') if p.strip()]
        score = round(row['Final_Score'] * 100, 1)
        conf_level, color = classify_confidence(score)
        predictions.append({
            "disease": row['Disease'],
            "score": score,
            "confidence": conf_level,
            "color": color,
            "description": info['Description'],
            "precautions": precautions,
            "matched_symptoms": row.get('matched_symptoms_str', '—')
        })

    if not predictions:
        return [{"error": "No diseases match your symptom pattern. Try adding more details."}]

    print(f"\n🧠 Input Symptoms: {symptoms}")
    for p in predictions:
        print(f"   → {p['disease']} ({p['score']}%) — {p['confidence']}")

    return predictions


# ===================================================
# 🔹 Example Run
# ===================================================
if __name__ == "__main__":
    test_symptoms = ["cough", "runny nose"]
    results = predict_disease(test_symptoms)

    print("\n🩺 Top Predicted Diseases:")
    for i, r in enumerate(results, 1):
        if "error" in r:
            print("❌", r["error"])
            continue
        print(f"{i}. {r['disease']} ({r['score']}%) — {r['confidence']} confidence")
        print(f"   📖 {r['description']}")
        print("   💡 Precautions:")
        for p in r['precautions']:
            print(f"      - {p}")
        print(f"   ✅ Matched Symptoms: {r['matched_symptoms']}")
        print()
