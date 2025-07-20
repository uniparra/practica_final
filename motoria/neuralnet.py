import pandas as pd
import numpy as np
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


hotel_data = pd.read_csv('../data/dataset_practica_final.csv')

# Aqui definiríamos la llamada a la clase que preprocese los datos, o incluso que traiga directamente los conjuntos generados y tratados ahí hasta aquí para solo usarlos
# X_train, X_test, y_train, y_test = train_test_split(X_kfeat, y_kfeat, test_size=0.2, random_state=42)


class SimpleNeuralNetwork():
    
    
    def __init__(self, X_train_set, y_train_set, X_test_set, y_test_set,  dropout = 0.3, reg = 0.0001, learning_rate = 0.001, dense_units = 32):
        # Inicializaciones correspondientes a los conjuntos de datos
        self.scaler = StandardScaler()
        self.X_train = X_train_set
        self.X_test = X_test_set
        self.y_train = y_train_set
        self.y_test = y_test_set
        self.input_shape = X_train_set.shape[1]
        
        # Parametros de inicialización de la estructura
        self.layer_dropout = dropout
        self.layer_reg = reg
        self.optimizer_lr = learning_rate
        self.layer_dense_units = dense_units
        
        # Lanzamos la creación, entrenamiento y evaluación del modelo
        self.launch_snn()
        
        


    def simple_neural_network_arch(self):
        # Creamos el modelo con 3 
        model = Sequential()
        model.add(Dense(self.layer_dense_units, input_shape=(self.input_shape,), activation='relu', kernel_regularizer=l2(self.layer_reg)))
        model.add(Dropout(self.layer_dropout))
        model.add(Dense(self.layer_dense_units, activation='relu', kernel_regularizer=l2(self.layer_reg)))
        model.add(Dropout(self.layer_dropout))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=self.optimizer_lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def train_simple_nn(self, model, X_train, y_train):
        
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True) # el mode a max es para buscar siempre la máxima accuracy
        history = model.fit(X_train, y_train, callbacks=[early_stopping], epochs=200, batch_size=10, validation_split=0.2, shuffle=True, verbose=1)
        
        return model, history
    
    def eval(self, trained_model, X_test_scaled, y_test):
        loss, accuracy = trained_model.evaluate(X_test_scaled, y_test, verbose=0)
        
        return loss, accuracy
    
    
    def launch_snn(self):
        
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        simple_nn_model = self.simple_neural_network_arch()
        trained_model, training_history = self.train_simple_nn(model=simple_nn_model, X_train=X_train_scaled, y_train=self.y_train)
        loss, acc = self.eval(trained_model=trained_model, X_test_scaled=X_test_scaled, y_test=self.y_test)














shape = X_train.shape[1]

def create_cnn_model(input_shape, num_classes, filters=32, kernel_size=3, dense_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    # Convolutional layer 1 + MaxPooling
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # Vector latente
    model.add(Flatten())

    # Capa expamsión y regularización
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))

    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    loss_func = 'binary_crossentropy'
    metrics_list = ['accuracy']

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics_list)
    return model







input_shape_cnn = (X_train.shape[1], 1)
num_classes = 2 

# Prueba la creación del modelo
my_cnn_model = create_cnn_model(input_shape=input_shape_cnn, num_classes=num_classes)
# my_cnn_model.summary()

cnn_model_wrapper = KerasClassifier(model=create_cnn_model, verbose=0, model__input_shape=input_shape_cnn, model__num_classes=num_classes)


pipeline_cnn = Pipeline([
    ('selector', SelectKBest()), 
    ('scaler', StandardScaler()),
    ('cnn', cnn_model_wrapper)
])


param_grid_cnn = {
    
    'selector__score_func': [f_classif, mutual_info_classif],
    'selector__k': [5, 'all'], 

    
    'cnn__model__filters': [16, 32], 
    'cnn__model__kernel_size': [2, 3], 
    'cnn__model__pool_size': [2], 
    'cnn__model__dense_units': [32, 64],
    'cnn__model__dropout_rate': [0.0, 0.2], 
    'cnn__model__learning_rate': [0.001, 0.0005], 

    
    'cnn__epochs': [10, 20], 
    'cnn__batch_size': [16, 32] 
}


grid_search_cnn = GridSearchCV(estimator=pipeline_cnn, param_grid=param_grid_cnn, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)

grid_search_cnn.fit(X_train, y_train)

# Imprime los mejores resultados
print(f"\nMejor puntuación: {grid_search_cnn.best_score_:.4f}")
print(f"Mejores parámetros: {grid_search_cnn.best_params_}")


best_cnn_pipeline = grid_search_cnn.best_estimator_
y_pred = best_cnn_pipeline.predict(X_test)