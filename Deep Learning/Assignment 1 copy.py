import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the Kaggle Pima Indians Diabetes dataset
file_path = "C:/Users/hp/Downloads/Deep Learning/diabetes.csv" 
df = pd.read_csv(file_path)

# Step 2: Handle Missing Values
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_check] = df[columns_to_check].replace(0, pd.NA)
df.fillna(df.median(), inplace=True)

# Step 3: Visualize the Data Distribution using a Boxplot (Before Normalization)
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[columns_to_check])
plt.title("Boxplot Before Normalization")
plt.show()

# Step 4: Normalize the Features using StandardScaler
scaler = StandardScaler()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_scaled = scaler.fit_transform(X)

# Step 5: Visualize the Data Distribution using a Boxplot (After Normalization)
df_normalized = pd.DataFrame(X_scaled, columns=df.columns[:-1])
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_normalized)
plt.title("Boxplot After Normalization")
plt.show()

# Step 6: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train and Evaluate Single-Layer Perceptron (SLP)
slp = Perceptron(max_iter=1000, random_state=42)
slp.fit(X_train, y_train)
y_pred_slp = slp.predict(X_test)

# Calculate accuracy for Single-Layer Perceptron
accuracy_slp = accuracy_score(y_test, y_pred_slp)

# Step 8: Train and Evaluate Multi-Layer Perceptron (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Calculate accuracy for Multi-Layer Perceptron
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# Step 9: Print the Results for Comparison
print(f"Single-Layer Perceptron (SLP) Accuracy: {accuracy_slp}")
print(f"Multi-Layer Perceptron (MLP) Accuracy: {accuracy_mlp}")

# Step 10: Confusion Matrices and Classification Reports for Detailed Evaluation
cm_slp = confusion_matrix(y_test, y_pred_slp)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

print("\nSingle-Layer Perceptron (SLP) Confusion Matrix:\n", cm_slp)
print("Single-Layer Perceptron (SLP) Classification Report:\n", classification_report(y_test, y_pred_slp))

print("\nMulti-Layer Perceptron (MLP) Confusion Matrix:\n", cm_mlp)
print("Multi-Layer Perceptron (MLP) Classification Report:\n", classification_report(y_test, y_pred_mlp))

# Step 11: Heatmap for Confusion Matrices
plt.figure(figsize=(8, 6))
sns.heatmap(cm_slp, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Single-Layer Perceptron Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title("Multi-Layer Perceptron Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Bar plot for Outcome distribution
sns.countplot(x='Outcome', data=df)
plt.title('Distribution of Outcome (Diabetes vs No Diabetes)')
plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
plt.ylabel('Count')
plt.show()
