import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import DBSCAN
import os

print("--- Starting Accident Severity & Hotspot Training Pipeline ---")

# 1. Load Dataset (Update filename to match your downloaded dataset in the 'data/' folder)
data_path = 'data/accidents.csv'  # Change this to your actual CSV filename

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows.")
else:
    print(f"Dataset not found at {data_path}. Creating a dummy sample dataset for demonstration...")
    # Creating a mock dataframe if dataset is not yet placed
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'Latitude': np.random.uniform(18.4, 18.7, n_samples),
        'Longitude': np.random.uniform(73.7, 74.0, n_samples),
        'Speed_limit': np.random.choice([30, 40, 50, 60, 80], n_samples),
        'Weather': np.random.choice(['Clear', 'Rain', 'Fog'], n_samples),
        'Road_Surface': np.random.choice(['Dry', 'Wet', 'Frost'], n_samples),
        'Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_samples, p=[0.7, 0.2, 0.1])
    })

# 2. Hotspot Analysis using DBSCAN
print("Running DBSCAN Spatial Clustering for Hotspots...")
coords = df[['Latitude', 'Longitude']].values
# eps=0.01 roughly corresponds to ~1km clusters
dbscan = DBSCAN(eps=0.01, min_samples=5).fit(coords)
df['Cluster_ID'] = dbscan.labels_
print(f"Identified {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)} distinct accident hotspots.")

# 3. Data Preprocessing & Encoding
print("Preprocessing features...")
categorical_cols = ['Weather', 'Road_Surface']
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Select features and target
feature_columns = ['Latitude', 'Longitude', 'Speed_limit']
for col in categorical_cols:
    if col in df.columns:
        feature_columns.append(col)

X = df[[col for col in feature_columns if col in df.columns]]
y = df['Severity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Classifier
print("Training Random Forest Severity Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 5. Save the Trained Model
joblib.dump(model, 'model.pkl')
print("\nSuccess! Model saved as 'model.pkl' in your root directory.")