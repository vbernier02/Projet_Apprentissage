import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np


def plot_multiclass_roc(y_test_binarized, y_proba, target_names, save_path, model_name):
    plt.figure(figsize=(10, 8))
    n_classes = len(target_names)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC Classe {target_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR) / Rappel')
    plt.title(f'Courbe ROC Multi-classes : {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, target_names, save_path, model_name):
    """Matrice de confusion avec heatmap"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matrice de Confusion : {model_name}')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe PrÃ©dite')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path, model_name, top_n=20):
    """Importance des features (pour RF et DT)"""
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features Importantes : {model_name}')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def plot_clustering_comparison(y_encoded, clusters_kmeans, clusters_cah, save_path):
    """Comparaison des résultats de clustering"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # K-Means
    from sklearn.metrics import confusion_matrix
    cm_kmeans = confusion_matrix(y_encoded, clusters_kmeans)
    sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Greens', ax=axes[0])
    axes[0].set_title('Contingence K-Means')
    axes[0].set_ylabel('Vraie Classe')
    axes[0].set_xlabel('Cluster')
    
    # CAH
    cm_cah = confusion_matrix(y_encoded, clusters_cah)
    sns.heatmap(cm_cah, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title('Contingence CAH')
    axes[1].set_ylabel('Vraie Classe')
    axes[1].set_xlabel('Cluster')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_models_comparison(metrics_dict, save_path):
    """Compare les performances des différents modèles"""
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    roc_aucs = [metrics_dict[m]['roc_auc'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].bar(models, accuracies, color=['skyblue', 'lightcoral'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Comparaison des Accuracy')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # ROC AUC
    axes[1].bar(models, roc_aucs, color=['skyblue', 'lightcoral'])
    axes[1].set_ylabel('ROC AUC (Macro)')
    axes[1].set_title('Comparaison des ROC AUC')
    axes[1].set_ylim([0, 1])
    for i, v in enumerate(roc_aucs):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()