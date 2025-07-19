from .iamodel import IaModel
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import logging

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
                        "scoring": "accuracy"
                    }
                }
    """
    def __init__(self, input: dict):
        super()
        self.data_input: dict = input.get("data_input")
        self.preprocess: dict = input.get("data_preproces")
        self.train_info: dict = input.get("train")
        self.pipeline: Pipeline = self.train_info.get("pipeline")
        self.df: pd.DataFrame = self.data_input.get("dataset")
        self.target_name: str = self.data_input.get("target_name")

        self.df_preprocessed: pd.DataFrame = super().preprocessing(self.df, self.preprocess)
        print(self.df_preprocessed.columns)

        self.X: pd.DataFrame = self.df_preprocessed.drop(columns=self.target_name)
        self.y: pd.DataFrame = self.df_preprocessed[self.target_name]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.X, self.y, test_size=self.preprocess.get("test_size"), random_state=1, stratify=self.y)) if self.preprocess.get("stratify") \
            else train_test_split(self.X, self.y, test_size=self.preprocess.get("test_size"), random_state=1)

        self.optimized_model: GridSearchCV = self.train(self.train_info.get("param_grid"), self.train_info.get("scoring"))

        self.metrics: dict = self.evaluate(self, self.optimized_model)


    @staticmethod
    def evaluate(self, model: GridSearchCV) -> dict:
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

    def train(self, param_grid: dict, scoring: str) -> GridSearchCV:
        """
        Entrena un modelo de sklearn con GridSearchCV
        """
        grid = GridSearchCV(self.pipeline, param_grid=param_grid, scoring=scoring, cv=5)
        result = grid.fit(self.X_train, self.y_train)
        logger.info(f"Se entrena {type(self.pipeline['model']).__name__} con lo hiperparámetros {grid.best_params_} y con un score promedio en CV de, {grid.best_score_}")
        return result

