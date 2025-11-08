import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===================================================
# 🔹 Load Clean Dataset
# ===================================================
df = pd.read_csv('data/medical_dataset_cleaned.csv')

# Remove missing and invalid rows
df = df.dropna(subset=['Disease', 'Symptoms'])
df = df[df['Symptoms'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

# Prepare symptom list
df['Symptoms'] = df['Symptoms'].str.lower()
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip() for s in x.split(';') if s.strip()])

# Generate full symptom set
all_symptoms = sorted(set(sum(df['Symptoms'], [])))

# ===================================================
# 🔹 Encode Disease Labels
# ===================================================
encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Disease'])

# ===================================================
# 🔹 Create Binary Feature Matrix
# ===================================================
X = []
for symptom_list in df['Symptoms']:
    row = [1 if s in symptom_list else 0 for s in all_symptoms]
    X.append(row)
X = pd.DataFrame(X, columns=all_symptoms)

y = df['Label']

# ===================================================
# 🔹 Split and Train Model
# ===================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1
)
model.fit(X_train, y_train)

# ===================================================
# 🔹 Save Model, Encoder, and Feature Columns
# ===================================================
with open('models/custom_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('models/symptom_columns.pkl', 'wb') as f:
    pickle.dump(all_symptoms, f)

# ===================================================
# ✅ Summary
# ===================================================
print("✅ Custom Medical AI model, encoder, and symptom columns saved successfully!")
print(f"🧬 Total Symptoms Learned: {len(all_symptoms)}")
print(f"🦠 Total Diseases Trained: {len(encoder.classes_)}")
