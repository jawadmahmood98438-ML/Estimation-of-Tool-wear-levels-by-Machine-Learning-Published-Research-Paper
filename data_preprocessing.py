import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    features = df.drop(columns=['label'])
    labels = df['label']

    encoder = OneHotEncoder(sparse=False)
    labels_encoded = encoder.fit_transform(labels.values.reshape(-1, 1))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test