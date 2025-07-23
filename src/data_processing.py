import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 

def load_and_preprocess_hotel_data(df_data, target_column, test_size=0.2, random_state=42):
    
    df_data_preprocessed = df_data.copy()  # Hcemos una copia del dataset para evitar modificar el original
    
    # Aquellas varaibles que son categóricas pero que pueden tomar posiblidades x > 5 aplicaremos técnicas de label encoding
    # Variable mes de llegada
    
    month_correspondance = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_data_preprocessed['arrival_date_month'] = df_data_preprocessed['arrival_date_month'].replace(month_correspondance)
    
    # Variables habitacion_reservada y habitacion_asignada, al tener ambas los mismos tipos de habitaciones y habitación asignada incluir dos posibilidades mas pues 
    # usamos esta para generar el diccionario
    
    reserved_rooms = sorted(df_data_preprocessed['assigned_room_type'].unique().tolist())
    rooms = {}
    for i in range(len(reserved_rooms)):
        rooms[reserved_rooms[i]] = i
    
    df_data_preprocessed['reserved_room_type'] = df_data_preprocessed['reserved_room_type'].replace(rooms)
    df_data_preprocessed['assigned_room_type'] = df_data_preprocessed['assigned_room_type'].replace(rooms)
    