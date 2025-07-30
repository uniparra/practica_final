import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Creo esta clase porque entrenar con todo el conjunto de características es muy costoso y lento. Estoy aplicando el 
# preprocesamiento visto en una clase de 'Neural Network Binary Classification' de Jhon Lacea en yt para el tema de balanceo 
# de clases

def feature_selection_And_balancing(X_train, X_test, y_train):
    
    # El preprocesado de características nos ha dejado un total de 41 columnas lo cual no es práctico
    # Primero sacamos las 25 más relevantes con mutual_info_classif que es un método de filtrado que captura 
    # relaciones no lineales. Por ejemplo nos quedamos con las 25 mas relevantes
    
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    scores_series = pd.Series(scores, index=X_train.columns) 
    scores_series = scores_series.sort_values(ascending=False)
    features_mic = scores_series.head(25).index.tolist()
    
    X_train_mic = X_train[features_mic]
    X_test_mic = X_test[features_mic]
    
    # Ahora estas variables las volvemos a pasar por otro filtro. El filtro embebido de Random Forest Classifier nos permite obtener
    # las features_importances, es un método que se emplea para capturar relaciones complejas. 
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_mic, y_train)

    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train_mic.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    features_rfc = feature_importances.head(15).index.tolist()
    
    X_train_final = X_train_mic[features_rfc]
    X_test_final = X_test_mic[features_rfc]
    
    # Una vez filtradass las features mas relevantes usamos SMOTE para balancear las clases, actualmente tenemos:
    # is_canceled - 0: 75166 y 1: 44224. Aplicamos SMOTE para rebalancear clases y que el modelo no se sesgue hacia 
    # la clase dominante
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)
    
    return X_train_resampled, y_train_resampled, X_test_final