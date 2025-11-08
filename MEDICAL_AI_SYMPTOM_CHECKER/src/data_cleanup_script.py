import pandas as pd

file_path = 'data/medical_dataset_raw.csv'

try:
    # 🧹 1️⃣ Read CSV with smart handling for embedded commas and quotes
    df = pd.read_csv(file_path, sep=',', quotechar='"', engine='python', on_bad_lines='skip')

    # 🧩 2️⃣ Remove all commented section header rows
    df = df[~df['Disease'].astype(str).str.startswith('#')]

    # 🧽 3️⃣ Clean whitespace and normalize casing
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    df['Severity'] = df['Severity'].str.capitalize()

    # 🩸 4️⃣ Drop duplicates and NaNs
    df = df.dropna(subset=['Disease'])
    df = df.drop_duplicates(subset=['Disease'], keep='first')

    # 💉 5️⃣ Remove any rows missing mandatory columns
    required_cols = ['Disease', 'Symptoms', 'Severity', 'Description', 'Precautions']
    df = df[[c for c in required_cols if c in df.columns]]

    # 💾 6️⃣ Save cleaned dataset
    output_path = 'data/medical_dataset_cleaned.csv'
    df.to_csv(output_path, index=False)

    print(f"✅ Clean dataset created successfully: {output_path}")
    print(f"📊 Total unique diseases: {len(df)}")

except Exception as e:
    print(f"❌ Error while cleaning dataset: {e}")
