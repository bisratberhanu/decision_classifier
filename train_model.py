import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
# Load Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Use feature names
y = pd.DataFrame(data.target, columns=["target"])  # Target column

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

print("Feature Data (X):")
print(X.head())  

print("\nTarget Data (y):")
print(y.head())  

print("\nFeature Data Statistics:")
print(X.describe())  

print("\nTarget Class Distribution:")
print(y['target'].value_counts())  # Counts of each target class

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and feature names
joblib.dump({"model": clf, "feature_names": data.feature_names, "target_names": data.target_names}, "decision_tree_model.pkl")
print("Model and metadata saved as 'decision_tree_model.pkl'")




# Export tree as text
print(export_text(clf, feature_names=data.feature_names))

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)

# Save the plot as an image
plt.savefig("decision_tree.png")
print("Decision tree plot saved as 'decision_tree.png'")
