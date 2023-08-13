import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump

# Load and preprocess the dataset
data_path = "spambase.data"
column_names = [f"attribute_{i}" for i in range(1, 58)] + ["label"]
data = pd.read_csv(data_path, header=None, names=column_names)

# Split the dataset into training and testing sets
X = data.iloc[:, :-1]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Model
knn_params = {"n_neighbors": np.arange(1, 21), "weights": ["uniform", "distance"]}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring="accuracy")
knn_grid.fit(X_train, y_train)
knn_best = knn_grid.best_estimator_

# Decision Tree Model
tree_params = {"criterion": ["gini", "entropy"], "max_depth": np.arange(1, 21)}
tree = DecisionTreeClassifier()
tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring="accuracy")
tree_grid.fit(X_train, y_train)
tree_best = tree_grid.best_estimator_

# Determine best model based on accuracy
best_model_name = "KNN" if accuracy_score(y_test, knn_best.predict(X_test)) > accuracy_score(y_test, tree_best.predict(X_test)) else "Decision Tree"
best_model = knn_best if best_model_name == "KNN" else tree_best

# Export the best model
dump(best_model, f"{best_model_name}_best_model.pkl")

# Print the name of the best model for reference
print(f"The best model is: {best_model_name}")

# Return the modified script for review
# modified_script
