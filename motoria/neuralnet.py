from iamodel import IaModel
import neural_feature_selection as nfs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification


# hotel_data = pd.read_csv('../data/dataset_practica_final.csv')

# Aqui definiríamos la llamada a la clase que preprocese los datos, o incluso que traiga directamente los conjuntos generados y tratados ahí hasta aquí para solo usarlos
# X_train, X_test, y_train, y_test = train_test_split(X_kfeat, y_kfeat, test_size=0.2, random_state=42)


class SimpleNeuralNetwork(IaModel, ):
    
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
                        "dropout": 0.3, 
                        "reg": 0.0001, 
                        "learning_rate": 0.001, 
                        "dense_units": 32, 
                        "patience": 10,
                        "scaler": StandardScaler,
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
    
    
    def __init__(self, entry):
        # Inicializaciones correspondientes a los conjuntos de datos
        self.scaler = entry['train']['scaler']
        self.train_info = entry['train']
        # self.X_preprocessed = self.X_train
        # self.y_train_preprocessed = self.y_train
        self.X_train_tunned, self.y_train_tunned, self.X_test_tunned = nfs.feature_selection_And_balancing(self.X_train, self.X_test, self.y_train)
        self.input_shape = self.X_train_tunned.shape[1]
        
        # Parametros de inicialización de la estructura
        self.layer_dropout = self.train_info['dropout']
        self.layer_reg = self.train_info['reg']
        self.optimizer_lr = self.train_info['learning_rate']
        self.layer_dense_units = self.train_info['dense_units']
        self.patience = self.train_info['patience']
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_tunned)
        self.X_test_scaled = self.scaler.transform(self.X_test_tunned)
        
        
        # Lanzamos la creación, entrenamiento y evaluación del modelo
        self.trained_model, _ = self.train()
        self.metrics = self.evaluate_model()
        
        # super().__init__(entry)
        # self.train_info: dict = entry.get("train")
        # self.pipeline: Pipeline = self.train_info.get("pipeline")
        # self.trained_model = self.train(self.train_info.get("mode"))
        # self.model = self.trained_model.best_estimator_['model'] if self.train_info in ["rs_train", "gs_train"] else self.trained_model['model']
        # if self.train_info in ["rs_train", "gs_train"]:
        #     self.selector = self.trained_model.best_estimator_.named_steps['selector'] if 'selector' in self.trained_model.best_estimator_.named_steps else None
        # else:
        #     self.selector = self.trained_model.named_steps[
        #         'selector'] if 'selector' in self.trained_model.named_steps else None

        # self.model_name = type(self.model).__name__
        # self.metrics: dict = self.evaluate(self.trained_model)
        
        


    def simple_neural_network_build(self):
        # Creamos el modelo con 3 
        model = Sequential()
        model.add(Dense(self.layer_dense_units, input_shape=(self.input_shape,), activation='relu', kernel_regularizer=l2(self.layer_reg)))
        model.add(Dropout(self.layer_dropout))
        model.add(Dense(self.layer_dense_units, activation='relu', kernel_regularizer=l2(self.layer_reg)))
        model.add(Dropout(self.layer_dropout))
        model.add(Dense(self.layer_dense_units, activation='relu', kernel_regularizer=l2(self.layer_reg)))
        model.add(Dropout(self.layer_dropout))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=self.optimizer_lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def train_simple_nn(self, model, X_train, y_train):
        
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=self.patience, restore_best_weights=True) # el mode a max es para buscar siempre la máxima accuracy
        history = model.fit(X_train, y_train, callbacks=[early_stopping], epochs=200, batch_size=10, validation_split=0.2, shuffle=True, verbose=1)
        
        return model, history
    
    def eval(self, trained_model):
        loss, accuracy = trained_model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        y_pred = trained_model.predict(self.y_test)
        y_proba = trained_model.predict_proba(self.X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        results = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc
        }
        return results
    
    
    def train(self):
        
        simple_nn_model = self.simple_neural_network_build()
        trained_model, training_history = self.train_simple_nn(model=simple_nn_model, X_train=self.X_train_scaled, y_train=self.y_train_tunned)
        
        return trained_model, training_history
    
    
    def evaluate_model(self):
        results = self.eval(trained_model=self.trained_model, X_test_scaled=self.X_test_scaled, y_test=self.y_test)
        return results















