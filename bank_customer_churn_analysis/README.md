# 📊 Comparative Analysis of SHAP Values and Causal Inference Methods for Explainability in Machine Learning Models

# 📝 Overview
Welcome to the repository for my thesis, "Comparative Analysis of SHAP Values and Causal Inference Methods for Explainability in Machine Learning Models". This research explores the strengths and limitations of SHAP values and introduces causal inference methods to enhance model explainability.

# 🎯 Aim of Research
To perform a comparative analysis of SHAP values and causal inference methods, highlighting their importance and roles in the field of information technology.

# 🛠 How to build this framework?

I have builded this framework on Colab. I will explain how to use this framework step by step. You can repeat the implementation following steps.

# Steps of SHAP Values and Causal Discovery

Following steps for SHAP Values Analysis

# Step 1 - Install shap library

pip install shap

# Step 2 - Import necessary libraries

import pandas as pd

import numpy as np

import xgboost as xgb

import shap

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Step 3 - Read the CSV file into a pandas DataFrame

df = pd.read_csv(r"https://raw.githubusercontent.com/serterergun/Implementation/main/bank_customer_churn_analysis/data/bank_customer_churn_dataset.csv")

# Step 4 - Drop unnecessary columns

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Step 5 - Convert categorical variables to dummy/indicator variables

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Step 6 - Define features (X)

X = df.drop('Exited', axis=1)

# Step 7 - Define target variable (y)

y = df['Exited']

# Step 8 - Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9 - Initialize the model

model = xgb.XGBClassifier(objective='binary:logistic', verbosity=1, seed=42)

# Step 10 - Train the model using the training data

model.fit(X_train, y_train)

# Step 11 - Predict the target values for the test set

y_pred = model.predict(X_test)

# Step 12 - Calculate the accuracy of the model

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Step 13 - Initialize the SHAP explainer

explainer = shap.Explainer(model)

# Step 14 - Calculate SHAP values for the dataset

shap_values = explainer(X_test)

# Step 15 - Create a summary plot of SHAP values

shap.summary_plot(shap_values, X_test)

plt.show()

![image](https://github.com/user-attachments/assets/a420d7f8-7d9e-4635-875b-9c1911d959a9)


# Step 16 - Calculate the mean absolute SHAP values for each feature

shap_sum = np.abs(shap_values.values).mean(axis=0)

importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T

importance_df.columns = ['Feature', 'Mean SHAP Value']

importance_df.sort_values(by='Mean SHAP Value', ascending=True, inplace=True)

# Step 17 - Plot the feature importance as a horizontal bar chart

plt.figure(figsize=(10, 6))

bars = plt.barh(importance_df['Feature'], importance_df['Mean SHAP Value'], color='salmon')

for bar, value in zip(bars, importance_df['Mean SHAP Value']):

    sign = "+" if value >= 0 else "-"
    
    plt.text(value + 0.05, bar.get_y() + bar.get_height()/2, f'{sign}{abs(value):.2f}',
    
             va='center', ha='left', fontsize=12, color='salmon')

plt.xlabel('Mean(|SHAP Value|) (impact on model output magnitude)')

plt.title('Mean SHAP Values')

plt.show()

![image](https://github.com/user-attachments/assets/7e43a240-0627-4d13-9123-c6e6477d7230)



# Following steps for Causal Discovery without Domain Knowledge


# Step 1 - Install causallearn library

pip install causal-learn==0.1.3.8

# Step 2 - Import necessary libraries

import pandas as pd

import numpy as np

from causallearn.search.ConstraintBased.PC import pc

from causallearn.utils.GraphUtils import GraphUtils

import matplotlib.pyplot as plt

import io

import matplotlib.image as mpimg

# Step 3 - Read the CSV file into a pandas DataFrame

file_path = 'https://raw.githubusercontent.com/serterergun/Implementation/main/bank_customer_churn_analysis/data/bank_customer_churn_dataset.csv'

df = pd.read_csv(file_path)

# Step 4 - Drop unnecessary columns

df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)

# Step 5 - Convert categorical variables to dummy/indicator variables

df = pd.get_dummies(df, columns=['Gender','Geography'], drop_first=True)

# Step 6 - Ensure all columns are numeric and fill any missing values

for column in df.columns:

    df[column] = pd.to_numeric(df[column], errors='coerce')
    
if df.isnull().sum().sum() > 0:

    df.fillna(df.mean(), inplace=True)
    
# Step 7 - Convert all columns to float64 to ensure compatibility with np.isnan

df = df.astype(np.float64)

# Step 8 - Get labels and convert data to numpy array

labels = df.columns.tolist()

data = df.to_numpy()

# Step 9 - Generate Causal Graph without Domain Knowledge

cg = pc(data)

pyd = GraphUtils.to_pydot(cg.G, labels=labels)

tmp_png = pyd.create_png(f="png")

fp = io.BytesIO(tmp_png)

img = mpimg.imread(fp, format='png')

plt.figure(figsize=(50, 50))

plt.axis('off')

plt.imshow(img)

plt.show()

![image](https://github.com/user-attachments/assets/30414c4d-80f9-4c29-a0fe-8af067d1420b)




# Following steps for Causal Discovery with Domain Knowledge


# Step 1 - Import necessary libraries for adding Domain Knowledge

from causallearn.graph.Edge import Edge

from causallearn.graph.Node import Node

from causallearn.graph.GraphNode import GraphNode

from causallearn.graph.Endpoint import Endpoint

import networkx as nx

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

# Step 2 - Adding Domain Knowledge

def apply_domain_knowledge(cg):

  cg.G.add_directed_edge(GraphNode("Age"), GraphNode("EstimatedSalary"))
  
  cg.G.add_directed_edge(GraphNode("Age"), GraphNode("CreditScore"))
  
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("CreditScore"))
  
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("NumOfProducts"))
  
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("IsActiveMember"))
  
  cg.G.add_directed_edge(GraphNode("HasCrCard"), GraphNode("Exited"))
  
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("Exited"))
  
  cg.G.add_directed_edge(GraphNode("EstimatedSalary"), GraphNode("Exited"))

  return cg

# Step 3 - Generate Causal Graph with Domain Knowledge

cg = pc(data, node_names=labels)

cg = apply_domain_knowledge(cg)

pyd = GraphUtils.to_pydot(cg.G, labels=labels)

tmp_png = pyd.create_png(f="png")

fp = io.BytesIO(tmp_png)

img = mpimg.imread(fp, format='png')

plt.figure(figsize=(50, 50))

plt.axis('off')

plt.imshow(img)

plt.show()

![image](https://github.com/user-attachments/assets/227e9c3f-9a48-4e58-a72b-3a0a7cda36c9)
