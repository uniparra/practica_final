from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from motoria.iamodel import IaModel
from motoria.mlmodels import MlModel

df_data = pd.read_csv("./dataset_practica_final.csv")

input = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('model', LogisticRegression())
        ]),

        "param_grid": [
            {
                'selector__score_func': [f_classif, mutual_info_classif],
                'selector__k': [5, 10, 'all'],
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear']
            }
        ],
        "scoring": "accuracy"
    }
}

regresion_config = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['hotel', 'distribution_channel', 'arrival_date_year', 'arrival_date_month',
                         'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
                         'stays_in_week_nights', 'adults', 'children', 'babies', 'country', 'is_repeated_guest',
                         'reserved_room_type', 'booking_changes', 'agent', 'company', 'days_in_waiting_list', 'adr',
                         'reservation_status', 'reservation_status_date', 'meal', 'market_segment'],
        "cols_label_encod": ['assigned_room_type'],
        "cols_one_hot_encod": ['deposit_type', 'customer_type', ],
        "nan_col_mean": [],
        "nan_col_mode": []
    },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('model', LogisticRegression())
        ]),

        "param_grid": [
            {
                'selector__score_func': [f_classif],
                'selector__k': ['all'],
                'model__C': [1],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear']
            }
        ],
        "scoring": "accuracy"
    }
}

inputBalanceado = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('balance', SMOTE()),
            ('model', LogisticRegression())
        ]),

        "param_grid": [
            {
                'selector__score_func': [f_classif],
                'selector__k': ['all'],
                'model__C': [1],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear']
            }
        ],
        "scoring": "accuracy"
    }
}

decision_config = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
        },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod" : ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type', 'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode" : ["country"]
        },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('model', DecisionTreeRegressor())
            ]),

        "param_grid": [
                {
                    'model': [DecisionTreeClassifier()],
                    'selector__score_func': [f_classif], #Analizar la posibilidad de usar otro selector de espacio de, p. ej. SelectFromModel
                    'selector__k': [5, 10, 'all'],
                    'model__max_depth': [None, 5, 10],
                    'model__min_samples_split': [2, 5, 10]
                },
                ],
        "scoring": "accuracy"
        }
}

random_forest_config_pesado = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('selector', SelectFromModel(RandomForestClassifier(n_estimators=100))),
            ('model', RandomForestClassifier())
        ]),
        "param_grid": [
            {
                'selector__estimator__n_estimators': [50, 100],
                'selector__estimator__max_depth': [5, 10, None],
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2],
                'model__bootstrap': [True, False]
            }
        ],
        "scoring": "accuracy"
    }
}

random_forest_config = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('selector', SelectFromModel(RandomForestClassifier(n_estimators=50, max_depth=5))),
            ('model', RandomForestClassifier(n_estimators=50, max_depth=5))
        ]),
        "param_grid": [
            {
                'model__n_estimators': [50],
                'model__max_depth': [5, 10],
                'model__min_samples_split': [2],
                'model__min_samples_leaf': [1],
                'model__bootstrap': [True]
            }
        ],
        "scoring": "accuracy"
    }
}

xgboost_config_pesado = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('selector', SelectFromModel(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))),
            ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ]),
        "param_grid": [
            {
                'selector__estimator__n_estimators': [50, 100],
                'selector__estimator__max_depth': [3, 6, 10],
                'model__n_estimators': [100, 200],
                'model__max_depth': [6, 10, 15],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.6, 0.8, 1.0],
                'model__colsample_bytree': [0.6, 0.8, 1.0]
            }
        ],
        "scoring": "accuracy"
    }
}

xgboost_config = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('selector', SelectFromModel(XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                                       n_estimators=50, max_depth=3))),
            ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                    n_estimators=50, max_depth=3))
        ]),
        "param_grid": [
            {
                'model__learning_rate': [0.1],
                'model__subsample': [0.8],
                'model__colsample_bytree': [0.8]
            }
        ],
        "scoring": "accuracy"
    }
}

from xgboost import XGBClassifier

xgboost_config_full = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('selector', 'passthrough'),
            ('model', XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42
            ))
        ]),
        "param_grid": [
            {
                'model__booster': ['gbtree', 'dart'],  # comparas booster
                'model__n_estimators': [50, 100],
                'model__max_depth': [3, 6],
                'model__learning_rate': [0.05, 0.1],
                'model__subsample': [0.8, 1.0],
                'model__colsample_bytree': [0.8, 1.0],
                'model__gamma': [0, 1],
                'model__reg_alpha': [0, 0.1],
                'model__reg_lambda': [1, 5]
            }

        ],
        "cv": 2,
        "scoring": "roc_auc"  # mejor métrica para binaria balanceada o no
    }
}
xgboost_config_rs = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "mode": "random_search",
        "pipeline": Pipeline([
            ('selector', 'passthrough'),
            ('model', XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42
            ))
        ]),
        "param_grid": [
            {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.03, 0.05, 0.1],
                'model__max_depth': [3, 4, 6],
                'model__subsample': [0.8, 1.0],
                'model__colsample_bytree': [0.8, 1.0],
                'model__gamma': [0, 0.1, 0.2],
                'model__reg_alpha': [0, 0.1],
                'model__reg_lambda': [1, 5],
                'model__booster': ['gbtree']
            }
        ],
        "cv": 14,
        "scoring": "accuracy"  # mejor métrica para binaria balanceada o no
    }
}
xgboost_config_simple = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "mode": "self_train",
        "pipeline": Pipeline([
            ('selector', SelectFromModel(
                estimator=XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    objective='binary:logistic',
                    random_state=42,
                    n_jobs=-1
                ),
                threshold="median"  # a veces 'median' funciona mejor que 'mean'
            )),
            ('model', XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    objective='binary:logistic',
                    random_state=42,
                    n_jobs=-1
                ))
        ]),
        "scoring": "roc_auc"  # mejor métrica para binaria balanceada o no
    }
}


# modeloDecision = MlModel(decision_config)
# modeloRegresionMejorado = MlModel(regresion_config)
# modeloRandomForest = MlModel(random_forest_config)
modeloXGBoost = MlModel(xgboost_config_rs)

# def mostrar_resultados(modelo: MlModel):
#     best_pipeline = modelo.optimized_model.best_estimator_
#     mask = best_pipeline.named_steps['selector'].get_support()
#
#
#     print("Resultados del modelo:")
#     print(f"Mejores parámetros: {modelo.optimized_model.best_params_}")
#     print(f"Mejor puntuación: {modelo.optimized_model.best_score_}")
#     print(f"Mejor estimador: {modelo.optimized_model.best_estimator_}")
#     # print(f"Coeficientes del modelo: {modelo.optimized_model.best_estimator_.named_steps['model'].coef_}")
#     print(f"Espacio de características, {list(modelo.X.columns[mask])}")
#     IaModel.plot_confusion(modelo.optimized_model, modelo.X_test, modelo.y_test)
#     print(modelo.metrics)
#     IaModel.plot_feature_importance(modelo.optimized_model,
#                                     list(modelo.X.columns[mask]))
#     IaModel.plot_roc_curve(modelo.optimized_model, modelo.X_test, modelo.y_test)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

def mostrar_resultados(modelo: MlModel):
    model_obj = modelo.trained_model

    # Detectar si es GridSearchCV o RandomizedSearchCV
    if isinstance(model_obj, (GridSearchCV, RandomizedSearchCV)):
        best_pipeline = model_obj.best_estimator_
        best_params = model_obj.best_params_
        best_score = model_obj.best_score_
        print("Resultados del modelo con búsqueda de hiperparámetros:")
    elif isinstance(model_obj, Pipeline):
        best_pipeline = model_obj
        best_params = "No aplica (pipeline directo)"
        best_score = "No aplica (pipeline directo)"
        print("Resultados del modelo con pipeline directo:")
    else:
        raise ValueError("Tipo de modelo no soportado")

    # Extraer modelo final (paso 'model' del pipeline)
    final_model = best_pipeline.named_steps.get('model', best_pipeline)

    # Intentar recuperar las columnas seleccionadas
    if 'selector' in best_pipeline.named_steps:
        selector = best_pipeline.named_steps['selector']
        try:
            mask = selector.get_support()
            selected_features = list(modelo.X.columns[mask])
        except Exception:
            selected_features = list(modelo.X.columns)
    else:
        selected_features = list(modelo.X.columns)

    print(f"Mejores parámetros: {best_params}")
    print(f"Mejor puntuación: {best_score}")
    print(f"Espacio de características: {selected_features}")

    # Graficar resultados
    IaModel.plot_confusion(model_obj, modelo.X_test, modelo.y_test)
    print(modelo.metrics)
    IaModel.plot_feature_importance(final_model, selected_features)
    IaModel.plot_roc_curve(model_obj, modelo.X_test, modelo.y_test)

# mostrar_resultados(modeloDecision)
# print("-----------------------------------------------------")
# mostrar_resultados(modeloRegresionMejorado)
# print("-----------------------------------------------------")
# mostrar_resultados(modeloRandomForest)
# print("-----------------------------------------------------")
mostrar_resultados(modeloXGBoost)