import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc
import os


def plot_confusionMatrix_v2(model, X_test, y_test, name_model, output_dir="results/plots"):
    
    filename= f"confusion_matrix_{name_model}.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusión")

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {filepath}")

    
def plot_training_accuracy(history, name_model, output_dir="results/plots"):
    filename= f"training_acc_{name_model}.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Precisión durante el entrenamiento")
    ax.plot(history['accuracy'], label='Precisión de Entrenamiento')
    ax.plot(history['val_accuracy'], label='Precisión de Validación')
    ax.set_xlabel('Época')
    ax.set_ylabel('Precisión')
    ax.legend()
    ax.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Curva de precisión durante entrenamiento guardada en: {filepath}")
    

def plot_training_loss(history, name_model, output_dir="results/plots"):
    filename= f"training_loss_{name_model}.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Pérdidas durante el entrenamiento")
    ax.plot(history['loss'], label='Pérdida en el Entrenamiento')
    ax.plot(history['val_loss'], label='Pérdida en la Validación')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.legend()
    ax.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Curva de pérdidas durante el entrenamiento guardada en: {filepath}")
    

def plot_precision_recall_curve(name_model, y_test, y_pred_proba, output_dir="results/plots"):
    
    filename= f"precision_recall_{name_model}.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Curva Precision-Recall")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    ax.plot(recall, precision, marker='.', label=f'AUC-PR = {auc_pr:.2f}')
    ax.set_xlabel('Precisión')
    ax.set_ylabel('Recall')
    ax.legend(loc="lower left")
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Curva Precision-Recall guardada en: {filepath}")