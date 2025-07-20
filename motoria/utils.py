import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os


def plot_confusion_v2(model, X_test, y_test, name_model, output_dir="results/plots"):
    
    filename= f"confusion_matrix_{name_model}.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusión")

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {filepath}")

    
def training_accuracy(history, name_model, output_dir="results/plots"):
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
    

def training_loss(history, name_model, output_dir="results/plots"):
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