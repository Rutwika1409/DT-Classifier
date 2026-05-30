import pandas as pd
import joblib
import os
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv('data/cleaned_data.csv')
selected_features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'radius_worst',
    'texture_worst',
    'perimeter_worst',
    'area_worst'
]

x = df[selected_features]
y = df['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("=" * 60)
print("PRE-PRUNING WITH GRID SEARCH")
print("=" * 60)

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}

grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid.fit(x_train, y_train)

pre_pruned_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

pre_pred = pre_pruned_model.predict(x_test)

pre_accuracy = accuracy_score(
    y_test,
    pre_pred
)

pre_precision = precision_score(
    y_test,
    pre_pred
)

pre_recall = recall_score(
    y_test,
    pre_pred
)

pre_f1 = f1_score(
    y_test,
    pre_pred
)

print("\nPre-Pruned Results")
print("Accuracy :", pre_accuracy)
print("Precision:", pre_precision)
print("Recall   :", pre_recall)
print("F1 Score :", pre_f1)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        pre_pred
    )
)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        pre_pred
    )
)

print("\n" + "=" * 60)
print("POST-PRUNING")
print("=" * 60)

full_tree = DecisionTreeClassifier(
    random_state=42
)

full_tree.fit(
    x_train,
    y_train
)

path = full_tree.cost_complexity_pruning_path(
    x_train,
    y_train
)

ccp_alphas = path.ccp_alphas

best_alpha = 0
best_f1_post = 0
best_post_model = None

for alpha in ccp_alphas:

    model = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=alpha
    )

    model.fit(
        x_train,
        y_train
    )

    pred = model.predict(
        x_test
    )

    score = f1_score(
        y_test,
        pred
    )

    if score > best_f1_post:
        best_f1_post = score
        best_alpha = alpha
        best_post_model = model

print("\nBest Alpha:")
print(best_alpha)

post_pred = best_post_model.predict(
    x_test
)

post_accuracy = accuracy_score(
    y_test,
    post_pred
)

post_precision = precision_score(
    y_test,
    post_pred
)

post_recall = recall_score(
    y_test,
    post_pred
)

post_f1 = f1_score(
    y_test,
    post_pred
)

print("\nPost-Pruned Results")
print("Accuracy :", post_accuracy)
print("Precision:", post_precision)
print("Recall   :", post_recall)
print("F1 Score :", post_f1)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        post_pred
    )
)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        post_pred
    )
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Pre-Pruned F1 Score : {pre_f1:.4f}")
print(f"Post-Pruned F1 Score: {post_f1:.4f}")

if pre_f1 >= post_f1:
    best_model = pre_pruned_model
    best_model_name = "Pre-Pruned Decision Tree"
    best_predictions = pre_pred
else:
    best_model = best_post_model
    best_model_name = "Post-Pruned Decision Tree"
    best_predictions = post_pred

print(f"\nBest Model: {best_model_name}")

print("\nFinal Accuracy:",
      accuracy_score(
          y_test,
          best_predictions
      ))

print("\nFinal Classification Report")
print(
    classification_report(
        y_test,
        best_predictions
    )
)

print("\nFinal Confusion Matrix")
print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)

if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(
    best_model,
    'model/model.pkl'
)

joblib.dump(
    selected_features,
    'model/features.pkl'
)

print("\nBest model saved successfully")