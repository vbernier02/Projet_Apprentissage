import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import is_classifier


def cv_evaluation(model_name, best_estimator, X, y_encoded, n_splits=5, scoring='accuracy'):

    if not is_classifier(best_estimator):
        print(f"ATTENTION: {model_name} non reconnu. Skip de la CV.")
        return

    print(f"\nValidation Croisée Finale : {model_name} (k={n_splits})")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        best_estimator, 
        X, 
        y_encoded, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=-1
    )
    
    print(f"Scores de {scoring.capitalize()} par pli : {scores}")
    print(f"{scoring.capitalize()} Moyenne (CV) : {np.mean(scores):.4f})")
    print(f"--------------------------------------------------")