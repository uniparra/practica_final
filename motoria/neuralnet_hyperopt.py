import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
import os

class NeuralNet_Hyperopt():
    
    def __init__(self, X, y): # Aqui directamente pasamos lo que serían los datos preprocesados
        
        self.scaler = StandardScaler()
        train_val_sets = self.train_val_obtainer(X, y)
        
        self.X_train = train_val_sets['X_train']
        self.X_test = train_val_sets['X_test']
        self.y_train = train_val_sets['y_train']
        self.y_test = train_val_sets['y_test']
        self.X_train_hp = train_val_sets['X_train_hp']
        self.y_val_hp = train_val_sets['y_val_hp']
        self.X_val_hp = train_val_sets['X_val_hp']
        self.y_train_hp = train_val_sets['y_train_hp']
        
        self.class_weights = self.prepare_class_weights(self.y_train)
        self.input_dim = self.X_train_hp.shape[1]
        
        self.best_hyperOpt_params = None
        self.final_model = None
        self.loss = None
        self.accuracy = None
        self.final_history = None
        
        
    def prepare_class_weights(self, y_train):
        
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', # Esto asigna pesos inversamente proporcionales a la frecuencia de clase
            classes=np.unique(y_train),
            y=y_train
        )
        return dict(enumerate(class_weights))
    
    
    def train_val_obtainer(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_hp, X_val_hp, y_train_hp, y_val_hp = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
        train_validation_sets = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_hp': X_train_hp,
            'X_val_hp': X_val_hp,
            'y_train_hp': y_train_hp,
            'y_val_hp': y_val_hp
        }
        
        return train_validation_sets
    
    
    def create_nn_model(self, input_dim, num_hidden_layers, units, learning_rate, dropout_rate):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        for i in range(num_hidden_layers):
            model.add(layers.Dense(units, activation='relu'))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(1, activation='sigmoid')) 

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    def objective_func_for_hyperopt(self, params):
        tf.keras.backend.clear_session() # Limpiar la sesión de Keras para cada ensayo
 
        model = self.create_nn_model(
            input_dim = self.input_dim,
            num_hidden_layers = int(params['num_hidden_layers']), # Hyperopt devuelve float, convertir a int
            units = int(params['units']),
            learning_rate = params['learning_rate'],
            dropout_rate = params['dropout_rate']
        )

        
        # Funciones que ayudan durante la busqueda de los mejores parametros) 
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        # Reduce el learning rate a medida que el modelo entrena favoreciendo que la funcion de perdida alcance un mínimo
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        # cuando el rendimiento del modelo no mejore mas detenemos su entrenamiento antes

        history = model.fit(
            self.X_train_hp, self.y_train_hp,
            epochs = params['epochs'],
            batch_size = int(params['batch_size']),
            verbose=0, 
            validation_data = (self.X_val_hp, self.y_val_hp),
            callbacks = [early_stopping, reduce_lr],
            class_weight = self.class_weights
        )
        
        val_accuracy = history.history['val_accuracy'][-1]
        loss = history.history['val_loss'][-1]

        return {
            'loss': -val_accuracy, 
            'status': STATUS_OK,
            'val_accuracy': val_accuracy,
            'val_loss': loss
        }
        
        
    def hyperOptSearch(self, space, max_evals):
        
        trials = Trials()
        best = fmin(fn = self.objective_func_for_hyperopt, space = space, algo = tpe.suggest, max_evals = max_evals, trials = trials)
        self.best_hyperOpt_params = {
            'num_hidden_layers': [1, 2, 3][best['num_hidden_layers']],
            'units': [128, 256, 512][best['units']],
            'learning_rate': best['learning_rate'],
            'dropout_rate': best['dropout_rate'],
            'batch_size': [64, 128, 256][best['batch_size']],
            'epochs': [50, 100][best['epochs']]
        }
        return self.best_hyperOpt_params
    
    
    def eval_best_hyperOptModel(self, save_model_path='results/predict_canceledornot_hotel.keras'):
        
        final_model = self.create_nn_model(
            input_dim = self.X_train.shape[1],
            num_hidden_layers = self.best_hyperOpt_params['num_hidden_layers'],
            units = self.best_hyperOpt_params['units'],
            learning_rate = self.best_hyperOpt_params['learning_rate'],
            dropout_rate = self.best_hyperOpt_params['dropout_rate']
        )
        
        final_model.fit(
            self.X_train, self.y_train,
            epochs = self.best_hyperOpt_params['epochs'],
            batch_size = self.best_hyperOpt_params['batch_size'],
            verbose=1, 
            class_weight = self.class_weights
        )
        
        self.loss, self.accuracy = final_model.evaluate(self.X_test, self.y_test, verbose=0)
        
        final_model.save(save_model_path)
        
        return self.loss, self.accuracy
    
    
    
if __name__ == '__main__':
    nn_optimizer = NeuralNet_Hyperopt(X_df, y_series) # datos ficticios
    
    space = {
        'num_hidden_layers': hp.choice('num_hidden_layers', [1, 2, 3]), # Número de capas ocultas
        'units': hp.choice('units', [64, 128, 256]), # Unidades por capa
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)), # Tasa de aprendizaje (logarítmica)
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5), # Tasa de Dropout
        'batch_size': hp.choice('batch_size', [32, 64, 128]), # Tamaño del batch
        'epochs': hp.choice('epochs', [50, 100, 150]) # Número máximo de épocas (EarlyStopping lo limitará)
    }

    # Ejecutar la búsqueda de hiperparámetros.
    # max_evals: número de combinaciones de hiperparámetros que Hyperopt probará.
    max_evaluations = 30 
    print(f"\nRealizando búsqueda de Hyperopt con {max_evaluations} evaluaciones...")
    best_params_found = nn_optimizer.hyperOptSearch(space, max_evals=max_evaluations)

    # Entrenar el modelo final con los mejores parámetros encontrados
    # Esto usa el conjunto completo de entrenamiento 
    nn_optimizer.train_final_model()

    # Evaluar el modelo final en el conjunto de prueba
    loss, accuracy = nn_optimizer.evaluate_final_model()