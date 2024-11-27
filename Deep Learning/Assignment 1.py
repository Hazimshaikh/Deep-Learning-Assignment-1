
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the Kaggle Pima Indians Diabetes dataset
file_path =   # Replace with your file path
df = pd.read_csv(file_path)

# Step 2: Handle missing values
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_check] = df[columns_to_check].replace(0, pd.NA)
df.fillna(df.median(), inplace=True)

# Step 3: Visualize the data distribution using a boxplot before normalization
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[columns_to_check])
plt.title("Boxplot Before Normalization")
plt.show()

# Step 4: Normalize the features using StandardScaler
scaler = StandardScaler()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X = scaler.fit_transform(X)

# Step 5: Visualize the data distribution using a boxplot after normalization
df_normalized = pd.DataFrame(X, columns=df.columns[:-1])  # Create a DataFrame for normalized features
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_normalized)
plt.title("Boxplot After Normalization")
plt.show()

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train and evaluate Single-Layer Perceptron (SLP)
slp = Perceptron(max_iter=1000, random_state=42)
slp.fit(X_train, y_train)
y_pred_slp = slp.predict(X_test)
accuracy_slp = accuracy_score(y_test, y_pred_slp)

# Step 8: Train and evaluate Multi-Layer Perceptron (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# Step 9: Print the results and comparison
print(f"Single-Layer Perceptron (SLP) Accuracy: {accuracy_slp}")
print(f"Multi-Layer Perceptron (MLP) Accuracy: {accuracy_mlp}")

# Confusion matrices and classification reports for detailed evaluation
print("\nSingle-Layer Perceptron (SLP) Confusion Matrix:\n", confusion_matrix(y_test, y_pred_slp))
print("Single-Layer Perceptron (SLP) Classification Report:\n", classification_report(y_test, y_pred_slp))

print("\nMulti-Layer Perceptron (MLP) Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))
print("Multi-Layer Perceptron (MLP) Classification Report:\n", classification_report(y_test, y_pred_mlp))
