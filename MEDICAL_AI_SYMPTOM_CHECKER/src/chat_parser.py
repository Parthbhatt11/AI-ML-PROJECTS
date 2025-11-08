import re
import pandas as pd

# Load all known symptoms from dataset
df = pd.read_csv('data/medical_dataset_cleaned.csv')
SYMPTOMS = sorted({s.strip().lower() for x in df['Symptoms'].dropna() for s in x.split(';')})

# Mild symptom whitelist (priority symptoms for general users)
MILD_ALLOWED = {
    "cough", "runny_nose", "sore_throat", "headache",
    "fever", "sneezing", "fatigue", "blocked_nose", "body_ache"
}

# Severe or advanced ones to only include if explicitly said
SEVERE_ALLOWED = {
    "chest_pain", "shortness_of_breath", "bloody_sputum",
    "high_fever", "vomiting", "nausea", "severe_cough", "chronic_cough"
}

def extract_symptoms(user_text: str):
    """Extracts clear, calibrated symptoms from natural text."""
    text = re.sub(r'[^\w\s]', '', user_text.lower())
    tokens = text.split()

    found = set()
    for s in SYMPTOMS:
        words = s.split('_')
        # match if all words appear in text
        if all(w in tokens for w in words):
            found.add(s)

    # 🔹 Remove redundant variants
    filtered = set()
    for s in found:
        if any(base in s for base in ["dry_cough", "chronic_cough", "severe_cough"]) and "cough" in found:
            continue
        if any(base in s for base in ["high_fever"]) and "fever" in found:
            continue
        filtered.add(s)

    # 🔹 Only keep mild and truly stated severe symptoms
    final = []
    for s in filtered:
        if s in MILD_ALLOWED:
            final.append(s)
        elif s in SEVERE_ALLOWED and s.replace("_", " ") in text:
            final.append(s)

    return sorted(set(final))
