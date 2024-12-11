# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load the dataset from a CSV file
def load_data(filepath):
    return pd.read_csv(filepath)  # Reads the CSV file into a pandas DataFrame

# Function to train an XGBoost model
def train_model(X, y):
    model = xgb.XGBClassifier(objective='binary:logistic', verbosity=1, seed=42)  # Initialize the model
    model.fit(X, y)  # Train the model using the training data
    return model  # Return the trained model

# Function to evaluate the model's performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)  # Predict the target values for the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model

    print(f"Accuracy: {accuracy:.4f}")  # Print the accuracy score

# Function to explain the model predictions using SHAP values
def model_explanation(model, X):
    explainer = shap.Explainer(model)  # Initialize the SHAP explainer
    shap_values = explainer(X)  # Calculate SHAP values for the dataset
    shap.summary_plot(shap_values, X)  # Create a summary plot of SHAP values
    plt.show()  # Show the plot


def mean_shap_plot(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['Feature', 'Mean SHAP Value']
    importance_df.sort_values(by='Mean SHAP Value', ascending=True, inplace=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['Feature'], importance_df['Mean SHAP Value'], color='salmon')

    for bar, value in zip(bars, importance_df['Mean SHAP Value']):
        sign = "+" if value >= 0 else "-"
        plt.text(value + 0.05, bar.get_y() + bar.get_height()/2, f'{sign}{abs(value):.2f}',
                va='center', ha='left', fontsize=12, color='salmon')

    plt.barh(importance_df['Feature'], importance_df['Mean SHAP Value'], color='salmon')
    plt.xlabel('Mean(|SHAP Value|) (impact on model output magnitude)')
    plt.title('Mean SHAP Values')
    plt.show()

# Main function to orchestrate the workflow
def main():
    # Step 1: Data Preprocessing
    df = load_data(r"https://raw.githubusercontent.com/serterergun/Implementation/main/bank_customer_churn_analysis/data/bank_customer_churn_dataset.csv")  # Load the dataset
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)  # Drop unnecessary columns
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)  # Convert categorical variables to dummy/indicator variables
    X = df.drop('Exited', axis=1)  # Define features (X)
    y = df['Exited']  # Define target variable (y)

    # Ensure all columns are numeric and fill any missing values
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    if df.isnull().sum().sum() > 0:  #aaaaaaaa
        df.fillna(df.mean(), inplace=True)

    # Convert all columns to float64 to ensure compatibility with np.isnan
    df = df.astype(np.float64)

    # Step 2: Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets
    model = train_model(X_train, y_train)

    # Evaluate Model
    evaluate_model(model, X_test, y_test)  # Evaluate the model on the test data

    # Step 3: Calculate SHAP Values
    model_explanation(model, X_test)  # Generate SHAP summary plot
    mean_shap_plot(model, X_test)  # Generate mean SHAP value plot

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
