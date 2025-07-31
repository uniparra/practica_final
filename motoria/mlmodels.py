import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    balanced_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import Pipeline

from .iamodel import IaModel

logger = logging.getLogger("motoria.mlmodels")


class MlModel(IaModel):
    """
    Clase para la creación de modelos de Machine Learning.
    Ejemplo de un input: dict,
            input = {
                    "entry_id": "Logistic1"
                    "data_input": {
                            "dataset": pd.DataFrame,
                            "target_name": "income"
                        },
                    "data_prepoces: {
                          "cols_label_encod": ["marital_status", "occupation", "relationship"],
                          "cols_one_hot_encod" : ["workclass", "education"],
                          "cols_to_drop": []
                          "test_size": 0.2,
                          "stratify": True
                        }
                    "train": {
                        "mode": "grid_search", # Puede ser "grid_search", "random_search" o "self_train"
                        "pipeline": pipeline = Pipeline([
                                            ('scaler', StandardScaler()),
                                            ('selector', SelectKBest()),
                                            ('model', LogisticRegression())
                                            ]),
                        "param_grid": {
                            {
                                'selector__score_func': [f_classif, mutual_info_classif],
                                'selector__k': [5, 10, 'all'],
                                'model__C': [0.01, 0.1, 1, 10],
                                'model__penalty': ['l2'],
                                'model__solver': ['liblinear']
                            },
                            {
                                'model': [DecisionTreeClassifier()],
                                'selector__score_func': [f_classif, mutual_info_classif], #Analizar la posibilidad de usar otro selector de espacio de, p. ej. SelectFromModel
                                'selector__k': [5, 10, 'all'],
                                'model__max_depth': [None, 5, 10],
                                'model__min_samples_split': [2, 5, 10]
                            }
                        },
                        "scoring": "accuracy",
                        "cv": 3
                    }
                }
    """

    def __init__(self, entry: dict):

        super().__init__(entry)
        self.train_info: dict = entry.get("train")
        self.pipeline: Pipeline = self.train_info.get("pipeline")
        self.trained_model = self.train(self.train_info.get("mode"))
        if self.train_info.get("mode") in ["random_search", "grid_search", "halving_grid_search",
                                           "halving_random_search"]:
            self.model = self.trained_model.best_estimator_.named_steps['model']
            self.selector = self.trained_model.best_estimator_.named_steps.get('selector')
        else:
            self.model = self.trained_model.named_steps['model']
            self.selector = self.trained_model.named_steps.get('selector')

        self.model_name = type(self.model).__name__
        self.y_pred = self.trained_model.predict(self.X_test)
        self.y_proba = self.trained_model.predict_proba(self.X_test)[:, 1]
        self.metrics: dict = self.evaluate(self.trained_model)

    def train(self, mode: str = "self_train"):
        """
        Entrena el modelo con los datos proporcionados.
        """
        match mode:
            case "grid_search":
                modelo = GridSearchCV(self.pipeline, param_grid=self.train_info.get("param_grid"),
                                      scoring=self.train_info.get("scoring"), cv=self.train_info.get("cv"), n_jobs=-1)
            case "random_search":
                modelo = RandomizedSearchCV(self.pipeline, param_distributions=self.train_info.get("param_grid"),
                                            scoring=self.train_info.get("scoring"), cv=self.train_info.get("cv"),
                                            n_jobs=-1)
            case "halving_grid_search":
                modelo = HalvingGridSearchCV(self.pipeline, param_grid=self.train_info.get("param_grid"),
                                             scoring=self.train_info.get("scoring"), cv=self.train_info.get("cv"),
                                             n_jobs=-1)
            case "halving_random_search":
                modelo = HalvingRandomSearchCV(self.pipeline, param_distributions=self.train_info.get("param_grid"),
                                               scoring=self.train_info.get("scoring"), cv=self.train_info.get("cv"),
                                               n_jobs=-1)
            case _:
                modelo = self.pipeline
        return modelo.fit(self.X_train, self.y_train)

    def evaluate(self, model) -> dict:
        """
        Evalúa el modelo y devuelve métricas de evaluación completas.
        """
        try:
            y_proba = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_proba)
        except AttributeError:
            roc_auc = None

        # Matriz de confusión para métricas basadas en TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()

        # Evitar divisiones por cero
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        results = {
            "accuracy": round(accuracy_score(self.y_test, self.y_pred), 3),
            "precision": round(precision_score(self.y_test, self.y_pred, zero_division=0), 3),
            "recall": round(recall_score(self.y_test, self.y_pred, zero_division=0), 3),
            "f1": round(f1_score(self.y_test, self.y_pred, zero_division=0), 3),
            "roc_auc": roc_auc,
            "specificity": round(specificity, 3),
            "false_positive_rate": round(fpr, 3),
            "npv": round(npv, 3),
            "balanced_accuracy": round(balanced_accuracy_score(self.y_test, self.y_pred), 3),
            "mcc": round(matthews_corrcoef(self.y_test, self.y_pred), 3)
        }
        logger.info(f"Métricas calculadas correctamente")
        return results

    def get_best_model_info(self):
        model_obj = self.trained_model

        if isinstance(model_obj, (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV)):
            return {
                "type": "search",
                "best_pipeline": model_obj.best_estimator_,
                "best_params": model_obj.best_params_,
                "best_score": model_obj.best_score_
            }
        elif isinstance(model_obj, Pipeline):
            return {
                "type": "direct",
                "best_pipeline": model_obj,
                "best_params": "No aplica (pipeline directo)",
                "best_score": "No aplica (pipeline directo)"
            }
        else:
            raise ValueError("Tipo de modelo no soportado")

    def get_selected_features(self):
        info = self.get_best_model_info()
        pipeline = info["best_pipeline"]

        if 'selector' in pipeline.named_steps:
            selector = pipeline.named_steps['selector']
            try:
                mask = selector.get_support()
                return list(self.X.columns[mask])
            except Exception:
                return list(self.X.columns)
        return list(self.X.columns)

    def print_model_summary(self):
        info = self.get_best_model_info()
        selected_features = self.get_selected_features()

        print("Resultados del modelo con búsqueda de hiperparámetros:" if info["type"] == "search"
              else "Resultados del modelo con pipeline directo:")

        print(f"Mejores parámetros: {info['best_params']}")
        print(f"Mejor puntuación: {info['best_score']}")
        print(f"Espacio de características: {selected_features}")

    def plot_confusion(self):
        """
        Grafica la matriz de confusión del modelo.
        """
        ConfusionMatrixDisplay.from_estimator(self.trained_model, self.X_test, self.y_test)
        plt.title("Matriz de confusión")
        plt.show()


    def plot_roc_curve(self):
        """
        Grafica la curva ROC del modelo.
        """
        RocCurveDisplay.from_estimator(self.trained_model, self.X_test, self.y_test)
        plt.title("Curva ROC")
        plt.show()

    def plot_feature_importance(self, top_n=20):
        """
        Gráfica la importancia de características para modelos que tienen feature_importances_.
        """
        # mask aquí es una lista o array con los nombres de las columnas seleccionadas
        mask = self.get_selected_features()

        # importancias: si el modelo tiene feature_importances_ lo usamos,
        # sino coef_ para regresión logística (coef_ puede ser 2D, aplanamos)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            logger.warning(f"Modelo {type(self.model).__name__} no tiene feature_importances_ ni coef_")
            return

        # Nos aseguramos que mask es un array numpy de strings (nombres de columnas)
        feature_names = np.array(mask)

        # Ordenamos los índices por importancia descendente y cogemos top_n
        indices = np.argsort(importances)[::-1][:top_n]

        # Seleccionamos características y valores según el orden top_n
        selected_features = feature_names[indices]
        selected_values = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(selected_features[::-1], selected_values[::-1])
        plt.xlabel("Importancia")
        plt.title(f"Top {top_n} características más importantes")
        plt.tight_layout()
        plt.show()

    def plot_model_evaluation(self):
        info = self.get_best_model_info()
        selected_features = self.get_selected_features()
        final_model = info["best_pipeline"].named_steps.get('model', info["best_pipeline"])

        self.plot_confusion(self)
        self.plot_feature_importance(self)
        self.plot_roc_curve(self)
        print(self.metrics)