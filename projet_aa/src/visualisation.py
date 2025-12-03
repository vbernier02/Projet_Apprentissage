import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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