import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # Load dataset
    data = pd.read_csv('bank_data_formatted.csv')
    print("Dataset Preview:\n", data.head())
    
    # Display data summary
    print("\nData Summary:\n", data.describe())
    print("\nData Info:\n", data.info())
    
    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include='object').columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    # Separate features and target variable
    X = data.drop('y', axis=1)  # Features
    y = data['y']  # Target
    
    # Set parameters
    test_size = 0.2
    random_state = 42
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Random Forest model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    

if __name__ == "__main__":
    main()
