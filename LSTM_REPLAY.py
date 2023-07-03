# Import delle librerie necessarie
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
from datetime import timedelta
import os

# Scarica i dati
data = yf.download('BTC-USD', '2010-01-01', '2023-01-01')[['Open', 'High', 'Low', 'Close', 'Volume']]

# Normalizza i dati
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Crea un set di dati per il modello LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 3):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back + 3, 3])  # Prevedi solo il prezzo di chiusura
    return np.array(dataX), np.array(dataY)

# Crea il dataset per l'addestramento
look_back = 2555 # Usa i prezzi di tutti i giorni precedenti per fare la previsione
X, y = create_dataset(scaled_data, look_back)

# Ridimensiona i dati per l'input del modello LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))
# 16662 16765 16674
# I prezzi effettivi per i prossimi 3 giorni (da sostituire con i tuoi dati reali)
actual_prices = [16625, 16688, 16679]

# Inizia l'addestramento e il salvataggio del modello
for _ in range(100): # Ripeti l'addestramento per 100 volte, o fino a quando non ottieni un modello con l'errore desiderato
    # Crea il modello LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=3))

    # Compila e addestra il modello
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32)

    # Prepara i dati degli ultimi giorni
    inputs = data.values[-look_back:]
    inputs = scaler.transform(inputs)

    # Crea il set di dati di input per la previsione
    X_test = []
    X_test.append(inputs)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], data.shape[1]))

    # Fai la previsione del prezzo dell'azione per i prossimi 3 giorni
    predicted_prices = model.predict(X_test)

    # Create an empty array with the same shape as the input data (but with 3 rows for our 3 predicted days)
    zeros_data = np.zeros((3, data.shape[1]))

    # Put the predicted values in the right place
    zeros_data[:,3] = predicted_prices.ravel()

    # Trasformazione inversa
    predicted_prices = scaler.inverse_transform(zeros_data)[:, 3]

    print('Prezzi previsti:', predicted_prices)

    # Controlla l'errore tra i prezzi previsti e i prezzi effettivi
    error = np.abs(predicted_prices - actual_prices)

    # Se l'errore è entro il range desiderato, salva il modello e interrompi il ciclo
    if np.all(error <= 80):
        model.save('model.h5')
        break

