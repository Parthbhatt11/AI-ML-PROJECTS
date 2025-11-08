AI–ML PROJECTS

Author: Parth Bhatt
Institute: Nanhi Pari Seemant Engineering, Pithoragarh
Year: 2025

----------------------------------------
PROJECT: MEDICAL SYMPTOM CHECKER
----------------------------------------
DESCRIPTION:
This project is an AI-based medical assistant that predicts possible diseases
based on user symptoms. It uses machine learning with fuzzy symptom matching
and severity weighting for realistic, human-normalized predictions.

FEATURES:
- Multi-disease ranking with confidence percentage
- Realistic probability adjustments
- Tkinter GUI with voice output
- Uses hybrid model + dataset combination logic

REQUIREMENTS:
- Python 3.10+
- See requirements.txt for dependencies

HOW TO RUN:
1. Create and activate a virtual environment.
2. Install dependencies using:
   pip install -r requirements.txt
3. Launch the app:
   python src/gui.py

EXAMPLE INPUT:
Symptoms: fever, sore throat, cough
OUTPUT:
Possible Diseases:
1. Common Cold (92.4%)
2. Flu (73.8%)
3. Sinusitis (41.2%)

FOLDER STRUCTURE:
medical_symptom_checker/
 ├── src/
 │   ├── MAIN.py
 │   ├── predictor.py
 │   └── chat_parser.py
 ├── data/
 │   └── medical_dataset_cleaned.csv
 ├── models/
 │   ├── custom_predictor.pkl
 │   ├── label_encoder.pkl
 │   └── symptom_columns.pkl
 ├── requirements.txt
 ├── README.txt
 └── LICENSE.txt

----------------------------------------
CREDITS:
Developed by Parth Bhatt
License: MIT License
GitHub: https://github.com/<parthbhatt11>/AI-ML-Projects
----------------------------------------
