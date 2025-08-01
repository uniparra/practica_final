from __future__ import annotations
import re
from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, get_dummies
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import logging

from sklearn.model_selection import GridSearchCV, train_test_split

logger = logging.getLogger("motoria")

class IaModel(object):
  def __init__(self, imput: dict):
    """
    Clase base para la creación de modelos de IA.
    """
    self.entry_id = imput.get("entry_id")
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
    self.y_pred = None
    self.y_proba = None

  def construir_tabla_comparacion(self, others: List[IaModel]):
    data = []
    modelos: list = others + [self]

    for modelo in modelos:
        acc = round(modelo.metrics.get("accuracy", 0), 3)
        f1 = round(modelo.metrics.get("f1", 0), 3)
        roc = round(modelo.metrics.get("roc_auc", 0), 3)
        recall = round(modelo.metrics.get("recall", 0), 3)
        name = modelo.entry_id
        data.append([name, acc, f1, roc, recall])  # Guardamos el objeto modelo también

    # Creamos el DataFrame con una columna extra para el objeto
    df = pd.DataFrame(data, columns=["Modelo", "Accuracy", "F1-score", "ROC-AUC", "Recall"])
    df["Suma"] = df[["Accuracy", "F1-score", "ROC-AUC", "Recall"]].sum(axis=1)

    # Ordenamos y mostramos
    df_sorted = df.sort_values(by="Suma", ascending=False)
    print("Tabla de comparación:")
    print(df_sorted.drop(columns=["Suma"]))

    # Obtener el mejor modelo directamente desde el DataFrame
    best_model_name = df_sorted.iloc[0]["Modelo"]
    best_model_obj = next(x for x in modelos if x.entry_id == best_model_name)

    # Mostrar info del mejor modelo
    print(f"\nMejor modelo: {best_model_obj.entry_id} con hiperparámetros:")
    info = best_model_obj.get_best_model_info()
    for k, v in info.items():
        print(f"{k}: {v}")



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
  def preprocess_outliers(df: DataFrame, cols, apply_log=False, lower=0.01, upper=0.99):
    df_copy = df.copy()
    for col in cols:
      # Capping
      q_low = df_copy[col].quantile(lower)
      q_high = df_copy[col].quantile(upper)
      df_copy[col + '_capped'] = df_copy[col].clip(q_low, q_high)

      # Variable binaria: es outlier superior
      df_copy[col + '_is_outlier'] = (df_copy[col] > q_high).astype(int)

      # Log transform si se desea
      if apply_log:
        df_copy[col + '_log'] = np.log1p(df_copy[col + '_capped'])

    return df_copy



  def evaluate_feature_importance(self, df, features):
      """
      Evalúa la importancia de variables con respecto a una variable target.
      """
      X = df[features]
      y = df[self.target_name]

      selector = SelectKBest(score_func=mutual_info_classif, k='all')
      selector.fit(X, y)

      scores = pd.DataFrame({
          'feature': features,
          'score': selector.scores_
      }).sort_values(by='score', ascending=False)

      return scores


  def select_best_transformations_replace(self, df_data, target_col=None):
      # Crear df con transformaciones
      df_data = df_data.copy()
      score_threshold=self.preprocess.get("outliers_threshold", 0.05)
      num_cols = df_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
      df_original = self.preprocess_outliers(df_data, cols=num_cols, apply_log=True).copy()

      # Detectar sólo las columnas transformadas nuevas, sin la base ni target
      new_features = [col for col in df_original.columns if col not in df_data.columns and col != target_col]

      # Calcular scores de importancia para las nuevas features
      scores_df = self.evaluate_feature_importance(df_original, features=new_features)

      filtered = scores_df[scores_df['score'] > score_threshold].copy()

      def base_name(feat):
          return re.sub(r'(_log|_capped|_is_outlier)$', '', feat)

      filtered['base'] = filtered['feature'].apply(base_name)

      best = filtered.loc[filtered.groupby('base')['score'].idxmax()]

      bases_selected = best['base'].tolist()
      features_selected = best['feature'].tolist()

      df_new = df_data.copy()

      cols_to_drop = []
      for base in bases_selected:
          # Eliminar la base original
          if base in df_new.columns:
              cols_to_drop.append(base)

          # Eliminar transformaciones no seleccionadas (que pueden estar en df_data si ya existen)
          pattern = re.compile(f"^{re.escape(base)}(_log|_capped|_is_outlier)$")
          to_remove = [col for col in df_new.columns if pattern.match(col) and col not in features_selected]
          cols_to_drop.extend(to_remove)

      df_new.drop(columns=cols_to_drop, inplace=True, errors='ignore')

      # Ahora añadimos las columnas transformadas seleccionadas (las mejores), copiándolas de df_original
      for feat in features_selected:
          base = re.sub(r'(_log|_capped|_is_outlier)$', '', feat)
          df_new[base] = df_original[feat]

      # Opcional: ordenar columnas para que target quede al final
      if target_col and target_col in df_new.columns:
          cols = [c for c in df_new.columns if c != target_col] + [target_col]
          df_new = df_new[cols]

      best_sorted = best[['feature', 'base', 'score']].sort_values(by='score', ascending=False).reset_index(drop=True)

      return df_new



  def preprocessing(self, df: DataFrame, cols_description: dict) -> DataFrame:
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
      df_data_outliers = (
          self.select_best_transformations_replace(df_data_clean_null,
                                                      target_col=self.target_name)
          if (self.preprocess.get("outliers", False))
          else df_data_clean_null)
      df_data_cols_encoded = IaModel.cols_encoder(df_data_outliers, cols_label_encod, cols_one_hot_encod)

      # Ensure the target column is of integer type
      if self.target_name in df_data_cols_encoded.columns:
          df_data_cols_encoded[self.target_name] = df_data_cols_encoded[self.target_name].astype('int64')

      return df_data_cols_encoded


  def get_selected_features(self):
    if 'selector' in self.model.named_steps:
      selector = self.model.named_steps['selector']
      try:
        mask = selector.get_support()
        return list(self.X.columns[mask])
      except Exception:
        return list(self.X.columns)
    return list(self.X.columns)

  def plot_confusion(self):
    """
    Grafica la matriz de confusión del modelo.
    """
    pass

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




