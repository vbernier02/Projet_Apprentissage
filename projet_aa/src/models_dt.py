from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

def train_decision_tree(preprocessor, X_train, y_train):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    parametre= {
        'classifier__max_depth': [3, 5, 7, 9, 11],
        'classifier__criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(
        model,
        parametre,
        cv=3,
        scoring='accuracy',
    )

    #model.fit(X_train, y_train)
    #return model

    grid_search.fit(X_train, y_train)
    print("Meilleurs hyperparamètres (Decision Tree) :", grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_decision_tree(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("--- Rapport de Classification (Decision Tree) ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    y_test_binarized = label_binarize(y_test, classes=range(len(target_names)))
    roc_auc = roc_auc_score(y_test_binarized, y_proba, multi_class="ovr", average="macro")
    print(f"Score ROC AUC (Macro Average) : {roc_auc:.4f}")

    return y_pred, y_proba
