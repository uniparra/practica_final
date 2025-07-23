from __future__ import annotations

from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, get_dummies
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, accuracy_score, precision_score, \
  recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import logging

from sklearn.model_selection import GridSearchCV, train_test_split

logger = logging.getLogger("motoria")

class IaModel(object):

  def __init__(self, imput: dict):
    """:
    Clase base para la creación de modelos de IA.
    """

    self.data_input: dict = imput.get("data_input")
    self.preprocess: dict = imput.get("data_preproces")
    self.df: pd.DataFrame = self.data_input.get("dataset")
    self.target_name: str = self.data_input.get("target_name")


    self.df_preprocessed: pd.DataFrame = self.preprocessing(self.df, self.preprocess)

    self.X: pd.DataFrame = self.df_preprocessed.drop(columns=self.target_name)
    self.y: pd.DataFrame = self.df_preprocessed[self.target_name]

    self.X_train, self.X_test, self.y_train, self.y_test = (
      train_test_split(self.X, self.y, test_size=self.preprocess.get("test_size"), random_state=1,
                       stratify=self.y)) if self.preprocess.get("stratify") \
      else train_test_split(self.X, self.y, test_size=self.preprocess.get("test_size"), random_state=1)
    self.model_name = None
    self.trained_model = None
    self.model = None
    self.selector = None
    self.name = None
    self.metrics = None


  def construir_tabla_comparacion(self, others: List[IaModel]):
    # Creamos la tabla de comparación
    data = []
    modelos: list = others + [self]
    for modelo in modelos:
      name = modelo.model_name
      acc = round(modelo.metrics.get("accuracy", 0), 3)
      f1 = round(modelo.metrics.get("f1", 0), 3)
      roc = round(modelo.metrics.get("roc_auc", 0), 3)
      data.append([name, acc, f1, roc])

    df = pd.DataFrame(data, columns=["Modelo", "Accuracy", "F1-score", "ROC-AUC"])
    print("Tabla de comparación:")
    df["Suma"] = df[["Accuracy", "F1-score", "ROC-AUC"]].sum(axis=1)
    df_sorted = df.sort_values(by="Suma", ascending=False).drop(columns="Suma")

    print(df_sorted.to_string(index=False))

    # Ordenamos usando __gt__
    ranking = sorted(modelos, reverse=True)
    print("\nRanking de modelos (mejor a peor):")
    for i, modelo in enumerate(ranking, 1):
      print(f"{i}. {modelo.model_name}({modelo.model.get_params().get()})")

  @abstractmethod
  def train(self, mode: str):
    """
    Entrena el modelo con los datos proporcionados.
    """
    pass


  @abstractmethod
  def evaluate(self, model):
    """
    Evalúa el modelo con los datos de prueba.
    """
    pass

  @staticmethod
  def clean_null(df: DataFrame) -> DataFrame:
    """
    Limpia los valores nulos de un dataframe.
    """
    null_percentage = df.isnull().mean()
    cols_to_drop = null_percentage[null_percentage > 0.5]
    logger.warning(f"Se borrarán las columnas{cols_to_drop}") if len(cols_to_drop) > 0 else logger.info(f"No hay columnas con valores nulos")
    df_data_drop_cols = df.drop(columns=cols_to_drop.index)
    return df_data_drop_cols

  @staticmethod
  def cols_encoder(df: DataFrame, cols_label_encod: list, cols_one_hot_encod: list) -> DataFrame:
    """
    Codifica las variables categóricas de un dataframe.
    """
    for col in cols_label_encod:
      conv_dict = {i: idx for idx, i in enumerate(df[col].unique())}
      df[col] = df[col].replace(conv_dict)

    df = get_dummies(df, columns=cols_one_hot_encod, dtype=int, drop_first=True)
    logger.info(f"Se ha realizado el preprocesamiento correctamente")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

  @staticmethod
  def fillin_nan(df: DataFrame, nan_col_mean: list, nan_cols_mode: list) -> DataFrame:
    """
    Rellena los valores nulos de un dataframe.
    """
    for col in nan_col_mean:
      df[col].fillna(df[col].mean(), inplace=True)

    for col in nan_cols_mode:
      df[col].fillna(df[col].mode()[0], inplace=True)

    logger.info(f"Se han rellenado los valores nulos correctamente")
    return df

  @staticmethod
  def preprocessing(df: DataFrame, cols_description: dict) -> DataFrame:
    """
    Codifica las variables categóricas de un dataframe.
    """
    cols_to_drop = cols_description.get("cols_to_drop", [])
    nan_col_mean = cols_description.get("nan_col_mean", [])
    nan_cols_mode = cols_description.get("nan_cols_mode", [])
    cols_label_encod = cols_description.get("cols_label_encod", [])
    cols_one_hot_encod = cols_description.get("cols_one_hot_encod", [])

    df_data_drop_cols = df.drop(columns=cols_to_drop).copy()
    df_data_nan_filled = IaModel.fillin_nan(df_data_drop_cols, nan_col_mean, nan_cols_mode)
    df_data_clean_null = IaModel.clean_null(df_data_nan_filled)
    df_data_cols_encoded = IaModel.cols_encoder(df_data_clean_null, cols_label_encod, cols_one_hot_encod)
    return df_data_cols_encoded

  def get_selected_features(self):
    if 'selector' in self.model.named_steps:
      selector = self.model.named_steps['selector']
      try:
        mask = selector.get_support()
        return (self.X.columns[mask])
      except Exception:
        return list(self.X.columns)
    return list(self.X.columns)

  def plot_confusion(self):
    """
    Grafica la matriz de confusión del modelo.
    """
    mask = self.get_selected_features()
    ConfusionMatrixDisplay.from_estimator(self.model, self.X_test.loc[:, mask], self.y_test)
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
    mask = self.get_selected_features()
    importances = self.model.feature_importances_ if hasattr(self.model, 'feature_importances_') else np.abs(self.model.coef_).flatten()
    feature_names = np.array(mask)
    indices = np.argsort(importances)[::-1][:top_n]
    selected_features = feature_names[indices]
    selected_values = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(selected_features[::-1], selected_values[::-1])
    plt.xlabel("Importancia")
    plt.title(f"Top {top_n} características más importantes")
    plt.tight_layout()
    plt.show()

  def __gt__(self, other):
    """
    Método para comparar dos modelos.
    """
    if not isinstance(other, IaModel):
      return NotImplemented
    j = 0
    for i in self.metrics.keys():
      if self.metrics[i] > other.metrics[i]:
        j+=1
      elif self.metrics[i] < other.metrics[i]:
        j-=1
      else:
        j+=0
    return j > 0

  def __lt__(self, other):
    """
    Método para comparar dos modelos.
    """
    return not self.__gt__(other)

  def __eq__(self, other):
    """
    Método para comparar dos modelos.
    """
    return self.__gt__(other) and not self.__gt__(other)


