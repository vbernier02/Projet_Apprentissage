from preprocessing import load_data, encode_target, build_preprocessor, split_data
from models_rf import train_random_forest, evaluate_random_forest
from models_dt import train_decision_tree, evaluate_decision_tree
from clustering import run_kmeans, run_cah
from visualisation import (plot_multiclass_roc, plot_confusion_matrix, 
                           plot_feature_importance, plot_clustering_comparison,
                           plot_models_comparison)
from cross_validation import cv_evaluation
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
import pandas as pd


def main():
    # --- 1. Chargement des données ---
    X, y = load_data("../../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")

    # --- 2. Encodage de la cible ---
    y_encoded, target_names = encode_target(y)
    n_classes = len(target_names)
    K = n_classes

    # --- 3. Préprocesseur ---
    preprocessor = build_preprocessor(X)

    # --- 4. Split train/test ---
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    # --- 5. Random Forest ---
    print("\n========== RANDOM FOREST ==========")
    rf_model = train_random_forest(preprocessor, X_train, y_train)
    y_pred_rf, y_proba_rf = evaluate_random_forest(rf_model, X_test, y_test, target_names)
    cv_evaluation("Random Forest", rf_model, X, y_encoded, n_splits=5)
    
    # Accuracy RF
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # --- 6. Decision Tree ---
    print("\n========== DECISION TREE ==========")
    dt_model = train_decision_tree(preprocessor, X_train, y_train)
    y_pred_dt, y_proba_dt = evaluate_decision_tree(dt_model, X_test, y_test, target_names)
    cv_evaluation("Decision Tree", dt_model, X, y_encoded, n_splits=5)
    
    # Accuracy DT
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # --- 7. Clustering ---
    print("\n========== CLUSTERING ==========")
    X_processed = preprocessor.fit_transform(X)
    X_processed_df = pd.DataFrame(X_processed)

    print("\n--- K-MEANS ---")
    clusters_kmeans, sil_k, ari_k, cont_k = run_kmeans(X_processed_df, y_encoded, K)
    print(f"Silhouette : {sil_k:.4f}")
    print(f"ARI : {ari_k:.4f}")
    print(cont_k)

    print("\n--- CAH ---")
    clusters_cah, sil_c, ari_c, cont_c = run_cah(X_processed_df, y_encoded, K)
    print(f"Silhouette : {sil_c:.4f}")
    print(f"ARI : {ari_c:.4f}")
    print(cont_c)

    # --- 8. Visualisations ---
    print("\n========== GÉNÉRATION DES VISUALISATIONS ==========")
    
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))
    
    # ROC Curves
    print("Génération des courbes ROC...")
    plot_multiclass_roc(y_test_binarized, y_proba_rf, target_names, 
                        "roc_curve_random_forest.png", "Random Forest")
    plot_multiclass_roc(y_test_binarized, y_proba_dt, target_names, 
                        "roc_curve_decision_tree.png", "Decision Tree")
    
    # Matrices de confusion
    print("Génération des matrices de confusion...")
    plot_confusion_matrix(y_test, y_pred_rf, target_names, 
                         "confusion_matrix_rf.png", "Random Forest")
    plot_confusion_matrix(y_test, y_pred_dt, target_names, 
                         "confusion_matrix_dt.png", "Decision Tree")
    
    # Feature importance
    print("Génération des importances des features...")
    X_processed_full = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    
    plot_feature_importance(rf_model, feature_names, 
                           "feature_importance_rf.png", "Random Forest")
    plot_feature_importance(dt_model, feature_names, 
                           "feature_importance_dt.png", "Decision Tree")
    
    # Comparaison clustering
    print("Génération de la comparaison des clustering...")
    plot_clustering_comparison(y_encoded, clusters_kmeans, clusters_cah, 
                              "clustering_comparison.png")
    
    # Comparaison des modèles
    print("Génération de la comparaison des modèles...")
    from sklearn.metrics import roc_auc_score
    metrics_dict = {
        'Random Forest': {
            'accuracy': acc_rf,
            'roc_auc': roc_auc_score(y_test_binarized, y_proba_rf, 
                                     multi_class="ovr", average="macro")
        },
        'Decision Tree': {
            'accuracy': acc_dt,
            'roc_auc': roc_auc_score(y_test_binarized, y_proba_dt, 
                                     multi_class="ovr", average="macro")
        }
    }
    plot_models_comparison(metrics_dict, "models_comparison.png")
    
    print("Fichiers créés :")
    print("  - roc_curve_random_forest.png")
    print("  - roc_curve_decision_tree.png")
    print("  - confusion_matrix_rf.png")
    print("  - confusion_matrix_dt.png")
    print("  - feature_importance_rf.png")
    print("  - feature_importance_dt.png")
    print("  - clustering_comparison.png")
    print("  - models_comparison.png")

if __name__ == "__main__":
    main()