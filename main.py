# Step 1: Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Data Collection
data = pd.read_csv("C:\\Users\\zanju\\Downloads\\healthcare_noshows.csv")  # Load your dataset here
data = data.fillna(0)  # Fill missing values with 0 or an appropriate method

# Step 3: Exploratory Data Analysis (EDA)
print(data.head())  # Show the first few rows of the dataset
print(data.info())  # Get info on the data types and missing values
print(data.describe())  # Get statistical summary of the dataset

# Step 4: Feature Engineering
# Convert categorical variables to numerical if needed
data['Scholarship'] = data['Scholarship'].astype(int)
data['Hipertension'] = data['Hipertension'].astype(int)
data['Diabetes'] = data['Diabetes'].astype(int)
data['Alcoholism'] = data['Alcoholism'].astype(int)
data['Handcap'] = data['Handcap'].astype(int)
data['SMS_received'] = data['SMS_received'].astype(int)
data['Showed_up'] = data['Showed_up'].astype(int)  # Target variable

# Step 5: Splitting the dataset into training and testing sets (Train-Test Split)
X = data[['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = data['Showed_up']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% testing

# Step 6: Model Training with Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)  # Fit Naive Bayes model

# Step 6.1: Model Training with Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)  # Create a Decision Tree model
dt_model.fit(X_train, y_train)  # Fit Decision Tree model

# Step 7: Model Evaluation for Naive Bayes
y_pred_nb = nb_model.predict(X_test)  # Predict with Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

print(f'Naive Bayes Accuracy: {accuracy_nb:.2f}')
print('Naive Bayes Confusion Matrix:')
print(conf_matrix_nb)

# Step 7.1: Model Evaluation for Decision Tree
y_pred_dt = dt_model.predict(X_test)  # Predict with Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print(f'Decision Tree Accuracy: {accuracy_dt:.2f}')
print('Decision Tree Confusion Matrix:')
print(conf_matrix_dt)

# Step 8: Visualization of Confusion Matrix for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did not show', 'Showed up'],
            yticklabels=['Did not show', 'Showed up'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Step 9: Visualizing the Decision Tree
plt.figure(figsize=(20, 10))  # Adjust figure size for better visibility
plot_tree(dt_model, feature_names=X.columns, class_names=['Did Not Show Up', 'Showed Up'],
          filled=True, rounded=True, fontsize=10, max_depth=3)  # Limit the depth to 3 levels
plt.title('Decision Tree Visualization (Limited Depth)')
plt.show()

# Step 10: Single User Input Testing Function
def predict_user_input(input_data, model):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)
    return "Showed Up" if prediction[0] == 1 else "Did Not Show Up"

# Example user input
user_input = [30, 0, 1, 0, 0, 0, 1]
result_nb = predict_user_input(user_input, nb_model)
result_dt = predict_user_input(user_input, dt_model)

print(f'Naive Bayes Prediction for {user_input}: {result_nb}')
print(f'Decision Tree Prediction for {user_input}: {result_dt}')
