from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

# Load the wine dataset
wine = datasets.load_wine()

print("=== Dataset Information ===")
print("Feature names:", wine.feature_names)
print("Target names:", wine.target_names)
print("Data shape:", wine.data.shape)
print("Target shape:", wine.target.shape)
print("First 5 rows of data:")
print(wine.data[0:5])
print("Target values:", wine.target)

# ============================================================================
# 1. HANDLING MISSING VALUES
# ============================================================================
print("\n=== Handling Missing Values ===")

# Check for missing values (wine dataset typically has no missing values, but we'll demonstrate the process)
print("Missing values in features:", np.isnan(wine.data).sum())
print("Missing values in target:", np.isnan(wine.target).sum())

# Create imputer for potential missing values (using median strategy)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(wine.data)

print("Data after imputation shape:", X_imputed.shape)

# ============================================================================
# 2. DATA NORMALIZATION
# ============================================================================
print("\n=== Data Normalization ===")

# Split data before normalization to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X_imputed, wine.target, test_size=0.3, random_state=42)

# Normalize data using MinMaxScaler (scales between 0 and 1)
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

print("Original data range - Min:", X_train.min(), "Max:", X_train.max())
print("Normalized data range - Min:", X_train_normalized.min(), "Max:", X_train_normalized.max())

# ============================================================================
# 3. ELBOW METHOD FOR OPTIMAL K SELECTION
# ============================================================================
print("\n=== Elbow Method for Optimal K Selection ===")

# Test different k values and calculate accuracy
k_range = range(1, 21)  # Test k from 1 to 20
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_normalized, y_train)
    y_pred = knn.predict(X_test_normalized)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"k={k}: Accuracy = {accuracy:.4f}")

# Find optimal k (highest accuracy)
optimal_k = k_range[np.argmax(accuracies)]
print(f"\nOptimal k value: {optimal_k} with accuracy: {max(accuracies):.4f}")

# ============================================================================
# 4. VISUALIZATION OF ELBOW METHOD
# ============================================================================
print("\n=== Creating Elbow Method Visualization ===")

plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Elbow Method for Optimal K Selection')
plt.grid(True, alpha=0.3)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# 5. FINAL MODEL WITH OPTIMAL K
# ============================================================================
print("\n=== Final Model Performance ===")

# Train final model with optimal k
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train_normalized, y_train)
final_y_pred = final_knn.predict(X_test_normalized)

print(f"Final Model Accuracy (k={optimal_k}): {metrics.accuracy_score(y_test, final_y_pred):.4f}")

# ============================================================================
# 6. COMPARISON: WITH AND WITHOUT NORMALIZATION
# ============================================================================
print("\n=== Comparison: With vs Without Normalization ===")

# Without normalization
knn_no_norm = KNeighborsClassifier(n_neighbors=optimal_k)
knn_no_norm.fit(X_train, y_train)
pred_no_norm = knn_no_norm.predict(X_test)
acc_no_norm = metrics.accuracy_score(y_test, pred_no_norm)

# With normalization (already done above)
acc_with_norm = metrics.accuracy_score(y_test, final_y_pred)

print(f"Accuracy without normalization: {acc_no_norm:.4f}")
print(f"Accuracy with normalization: {acc_with_norm:.4f}")
print(f"Improvement: {acc_with_norm - acc_no_norm:.4f} ({((acc_with_norm - acc_no_norm) / acc_no_norm * 100):.2f}%)")

# ============================================================================
# 7. CLASSIFICATION REPORT
# ============================================================================
print("\n=== Detailed Classification Report ===")
print(metrics.classification_report(y_test, final_y_pred, target_names=wine.target_names))



"""
# Assigning features and label variables
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
# Label or target variable
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers
weather_encoded=le.fit_transform(weather)
print(weather_encoded)


# converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
#combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))

model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(features,label)
# Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Prediction: " ,predicted)
"""
