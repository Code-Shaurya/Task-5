import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree

df = pd.read_csv('heart.csv')  # Use relative path if file is in same folder
df = df.dropna()
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

dot_data = tree.export_graphviz(dt, out_file=None, feature_names=X.columns, class_names=True, filled=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)
graph.view()  # Opens the decision_tree.png image

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()  # Show plot directly in VS Code

cv_dt = cross_val_score(dt, X, y, cv=5).mean()
cv_rf = cross_val_score(rf, X, y, cv=5).mean()

print(f"\nDecision Tree Accuracy: {acc_dt:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print(f"Decision Tree CV Accuracy: {cv_dt:.4f}")
print(f"Random Forest CV Accuracy: {cv_rf:.4f}")