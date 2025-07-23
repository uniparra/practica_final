import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .iamodel import IaModel

logger = logging.getLogger("motoria.mlmodels")

class MlModel(IaModel):
    """
    Clase para la creación de modelos de Machine Learning.
    Ejemplo de un input: dict,
            input = {
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
        self.model = self.trained_model.best_estimator_['model'] if self.train_info in ["rs_train", "gs_train"] else self.trained_model['model']
        if self.train_info in ["rs_train", "gs_train"]:
            self.selector = self.trained_model.best_estimator_.named_steps['selector'] if 'selector' in self.trained_model.best_estimator_.named_steps else None
        else:
            self.selector = self.trained_model.named_steps[
                'selector'] if 'selector' in self.trained_model.named_steps else None

        self.model_name = type(self.model).__name__
        self.metrics: dict = self.evaluate(self.trained_model)


    @staticmethod
    def gs_train(self, param_grid: dict, scoring: str, cv: int = 2) -> GridSearchCV:
        """
        Entrena un modelo de sklearn con GridSearchCV
        """
        grid = GridSearchCV(self.pipeline, param_grid=param_grid, scoring=scoring, cv=cv)
        result = grid.fit(self.X_train, self.y_train)
        logger.info(f"Se entrena {type(self.pipeline['model']).__name__} con lo hiperparámetros {grid.best_params_} y con un score promedio en CV de, {grid.best_score_}")
        return result
    @staticmethod
    def rs_train(self, param_grid: dict, scoring: str, cv: int) -> RandomizedSearchCV:
        """
        Entrena un modelo de sklearn con RandomizedSearchCV
        """
        grid = RandomizedSearchCV(self.pipeline, param_distributions=param_grid, scoring=scoring, cv=cv)
        result = grid.fit(self.X_train, self.y_train)
        logger.info(f"Se entrena {type(self.pipeline['model']).__name__} con lo hiperparámetros {grid.best_params_} y con un score promedio en CV de, {grid.best_score_}")
        return result
    @staticmethod
    def self_train(self):
        """
        Entrena un modelo de sklearn sin GridSearchCV ni RandomizedSearchCV
        """
        model = self.pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Se entrena {type(self.pipeline['model']).__name__} con los hiperparámetros {self.pipeline.get_params()}")
        return model

    def train(self, mode: str="self_train") -> GridSearchCV:
        """
        Entrena el modelo con los datos proporcionados.
        """
        match mode:
            case "grid_search": modelo=MlModel.gs_train(self, self.train_info.get("param_grid"), self.train_info.get("scoring"), self.train_info.get("cv"))
            case "random_search": modelo=MlModel.rs_train(self, self.train_info.get("param_grid"), self.train_info.get("scoring"), self.train_info.get("cv"))
            case "self_train": modelo=MlModel.self_train(self)
        return modelo

    def evaluate(self, model) -> dict:
        """
        Evalua los modelos y devuelve las distintas métricas para poder compararlas después.
        """
        y_pred = model.predict(self.X_test)
        try:  # No todos los modelos tienen predict_proba, ver si los requeridos sí y si eso eliminar el try.
            y_proba = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_proba)
        except AttributeError:
            roc_auc = None

        results = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc
        }
        logger.info(f"Métricas calculadas correctamente")
        return results

    def get_best_model_info(self):
        model_obj = self.trained_model

        if isinstance(model_obj, (GridSearchCV, RandomizedSearchCV)):
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



    def print_model_summary(self, modelo):
        info = self.get_best_model_info(modelo)
        selected_features = self.get_selected_features(modelo)

        print("Resultados del modelo con búsqueda de hiperparámetros:" if info["type"] == "search"
              else "Resultados del modelo con pipeline directo:")

        print(f"Mejores parámetros: {info['best_params']}")
        print(f"Mejor puntuación: {info['best_score']}")
        print(f"Espacio de características: {selected_features}")

    def plot_model_evaluation(self, modelo):
        info = self.get_best_model_info(modelo)
        selected_features = self.get_selected_features(modelo)
        final_model = info["best_pipeline"].named_steps.get('model', info["best_pipeline"])

        IaModel.plot_confusion(modelo.trained_model, modelo.X_test, modelo.y_test)
        print(modelo.metrics)
        IaModel.plot_feature_importance(final_model, selected_features)
        IaModel.plot_roc_curve(modelo.trained_model, modelo.X_test, modelo.y_test)


