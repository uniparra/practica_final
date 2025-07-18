from abc import abstractmethod

import numpy as np
from pandas import DataFrame, get_dummies
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import logging

from sklearn.model_selection import GridSearchCV

logger = logging.getLogger("motoria")

class IaModel():

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
  @abstractmethod
  def train(self, param_grid: dict, scoring: str) -> GridSearchCV:
    """
    Entrena el modelo con los datos proporcionados.
    """
    pass

  def plot_confusion(model, X_test, y_test):
    """
    Grafica la matriz de confusión del modelo.
    """
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Matriz de confusión")
    plt.show()

  def plot_roc_curve(model, X_test, y_test):
    """
    Grafica la curva ROC del modelo.
    """
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("Curva ROC")
    plt.show()

  def plot_feature_importance(model, feature_names, top_n=20): #tal vez tenga que ir en mlmodels.py al no servir para las cnn
    """
    Gráfica la importancia de características para modelos que tienen feature_importances_.
    """
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_).flatten()
    indices = np.argsort(importances)[::-1][:top_n]
    features = np.array(feature_names)[indices]
    values = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(features[::-1], values[::-1])
    plt.xlabel("Importancia")
    plt.title(f"Top {top_n} características más importantes")
    plt.tight_layout()
    plt.show()