

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier # O KerasRegressor si es un problema de regresión
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential # Import Sequential directly
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # Import specific layer types
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


hotel_data = pd.read_csv('../data/dataset_practica_final.csv')

# Aqui definiríamos la llamada a la clase que preprocese los datos, o incluso que traiga directamente los conjuntos generados y tratados ahí hasta aquí para solo usarlos
# X_train, X_test, y_train, y_test = train_test_split(X_kfeat, y_kfeat, test_size=0.2, random_state=42)

shape = X_train.shape[1]

def simple_neural_network(shape):
    # Creamos el modelo
    model = Sequential()
    model.add(Dense(16, input_shape=(shape,), activation='relu')) 
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilamos el modelo
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_simple_nn(model, X_train, y_train):
    
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True) # el mode a max es para buscar siempre la máxima accuracy
    history = model.fit(X_train, y_train, callbacks=[early_stopping], epochs=200, batch_size=10, validation_split=0.2, shuffle=True, verbose=1)
    
    return model, history





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