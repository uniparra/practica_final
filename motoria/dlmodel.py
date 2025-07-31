import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight

from .iamodel import IaModel

class DlModel(IaModel):
    def __init__(self, entry: dict):
        super().__init__(entry)
        self.train_info: dict = entry.get("train")
        self.best_params = None
        self.history = None
        self.model = None
        self.scaler = StandardScaler()

        # Escalamos los datos
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Partición para validación durante búsqueda de hiperparámetros
        self.X_train_hp, self.X_val_hp, self.y_train_hp, self.y_val_hp = train_test_split(
            self.X_train_scaled, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )

        # Cálculo de pesos de clase
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        self.class_weights_dict = dict(enumerate(class_weights))

        # Entrenamiento y optimización
        self.trained_model = self.train()
        self.y_pred = (self.trained_model.predict(self.X_test_scaled) > 0.5).astype("int32").flatten()
        self.y_proba = self.trained_model.predict(self.X_test_scaled).flatten()
        self.metrics = self.evaluate(self.trained_model)

    def create_model(self, input_dim, num_hidden_layers, units, learning_rate, dropout_rate):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        for _ in range(num_hidden_layers):
            model.add(layers.Dense(units, activation='relu'))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(1, activation='sigmoid'))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def objective(self, params):
        tf.keras.backend.clear_session()

        model = self.create_model(
            input_dim=self.X_train_hp.shape[1],
            num_hidden_layers=int(params['num_hidden_layers']),
            units=int(params['units']),
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate']
        )

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        history = model.fit(
            self.X_train_hp, self.y_train_hp,
            epochs=int(params['epochs']),
            batch_size=int(params['batch_size']),
            validation_data=(self.X_val_hp, self.y_val_hp),
            verbose=0,
            callbacks=[reduce_lr, early_stopping],
            class_weight=self.class_weights_dict
        )

        val_accuracy = history.history['val_accuracy'][-1]

        return {
            'loss': -val_accuracy,
            'status': STATUS_OK,
            'val_accuracy': val_accuracy,
            'model': model,
            'history': history
        }

    def train(self, mode: str = "hyperopt") -> keras.Model:
        """
        Entrena una red neuronal con búsqueda de hiperparámetros usando Hyperopt.
        Permite configurar el espacio de búsqueda y los callbacks desde el entry.
        """
        config = self.train_info
        max_evals = config.get("max_evals", 50)

        # Espacio de búsqueda definido por el usuario o por defecto
        user_space = config.get("search_space", {})

        # Asegurarse de que los valores por defecto sean listas válidas para min/max
        default_lr_values = user_space.get('learning_rate', [0.0001, 0.01]) # Asegúrate que haya min y max si no está en config
        default_dropout_values = user_space.get('dropout_rate', [0.0, 0.5]) # Asegúrate que haya min y max si no está en config


        space = {
            'num_hidden_layers': hp.choice('num_hidden_layers', user_space.get('num_hidden_layers', [1, 2, 3])),
            'units': hp.choice('units', user_space.get('units', [128, 256, 512])),
            # CORRECCIÓN AQUÍ: Usar min() y max() sobre la lista completa
            'learning_rate': hp.loguniform('learning_rate',
                                           np.log(min(default_lr_values)),
                                           np.log(max(default_lr_values))),
            # CORRECCIÓN AQUÍ: Usar min() y max() sobre la lista completa
            'dropout_rate': hp.uniform('dropout_rate',
                                       min(default_dropout_values),
                                       max(default_dropout_values)),
            'batch_size': hp.choice('batch_size', user_space.get('batch_size', [64, 128, 256])),
            'epochs': hp.choice('epochs', user_space.get('epochs', [50, 100]))
        }

        # Callbacks personalizados
        cb_config = config.get("callbacks_config", {})
        reduce_lr_cfg = cb_config.get("reduce_lr", {"monitor": "val_loss", "factor": 0.5, "patience": 5, "min_lr": 1e-5})
        early_stopping_cfg = cb_config.get("early_stopping", {"monitor": "val_loss", "patience": 15, "restore_best_weights": True})

        def objective(params):
          tf.keras.backend.clear_session()

          model = self.create_model(
              input_dim=self.X_train_hp.shape[1],
              num_hidden_layers=int(params['num_hidden_layers']),
              units=int(params['units']),
              learning_rate=params['learning_rate'],
              dropout_rate=params['dropout_rate']
          )

          history = model.fit(
              self.X_train_hp, self.y_train_hp,
              epochs=int(params['epochs']),
              batch_size=int(params['batch_size']),
              validation_data=(self.X_val_hp, self.y_val_hp),
              verbose=0,
              callbacks=[
                  ReduceLROnPlateau(**reduce_lr_cfg),
                  EarlyStopping(**early_stopping_cfg)
              ],
              class_weight=self.class_weights_dict
          )

          val_accuracy = history.history['val_accuracy'][-1]
          return {
              'loss': -val_accuracy,
              'status': STATUS_OK,
              'val_accuracy': val_accuracy,
              'model': model,
              'history': history
          }

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        self.best_params = best
        best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
        self.history = best_trial['history']
        return best_trial['model']


    def evaluate(self, model) -> dict:
        """
        Evalúa la red neuronal entrenada.
        """
        y_pred = self.y_pred
        y_proba = self.y_proba

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        roc_auc = roc_auc_score(self.y_test, y_proba)

        results = {
            "accuracy": round(accuracy_score(self.y_test, y_pred), 3),
            "precision": round(precision_score(self.y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(self.y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(self.y_test, y_pred, zero_division=0), 3),
            "roc_auc": round(roc_auc, 3),
            "specificity": round(specificity, 3),
            "false_positive_rate": round(fpr, 3),
            "npv": round(npv, 3),
            "balanced_accuracy": round(balanced_accuracy_score(self.y_test, y_pred), 3),
            "mcc": round(matthews_corrcoef(self.y_test, y_pred), 3)
        }

        return results

    def get_best_model_info(self):
        return {
            "type": "deep_learning",
            "best_params": self.best_params,
            "best_score": self.metrics["accuracy"],
            "architecture_summary": self.trained_model.summary()
        }

    def get_selected_features(self):
        return list(self.X.columns)

    def print_model_summary(self):
        print("--- Resultados del modelo DL ---")
        print(f"Mejores hiperparámetros (índices): {self.best_params}")
        print("Resumen del modelo:")
        self.trained_model.summary()
        print(f"Espacio de características: {self.get_selected_features()}")

    def plot_confusion(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Canceled', 'Canceled'])
        disp.plot()
        plt.title("Matriz de confusión") # Add title back
        plt.show()

    def plot_roc_curve(self):
      fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
      roc_auc = roc_auc_score(self.y_test, self.y_proba)

      plt.figure()
      plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
      plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC) Curve')
      plt.legend(loc="lower right")
      plt.show()

    def plot_model_evaluation(self):
        self.plot_confusion(self)
        self.plot_roc_curve(self)
        print(self.metrics)