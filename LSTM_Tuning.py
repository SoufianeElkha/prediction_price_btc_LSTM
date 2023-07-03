from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
from datetime import timedelta

# Scarica i dati
data = yf.download('BTC-USD', '2008-01-01', '2023-01-01')[['Open', 'High', 'Low', 'Close', 'Volume']]

# Calcola la media mobile su 14 giorni del prezzo di chiusura
data['SMA'] = data['Close'].rolling(window=14).mean()

# Gestisci i valori NaN che possono risultare dal calcolo della media mobile
data = data.dropna()

# Crea lo scaler
scaler = StandardScaler()

# Fitta lo scaler sui dati e trasforma
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
look_back = 365  # Usa i prezzi di giorni precedenti per fare la previsione
X, y = create_dataset(scaled_data, look_back)

# Ridimensiona i dati per l'input del modello LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))

# Crea un modello con un set variabile di iperparametri
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                          return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                          return_sequences=False))
    model.add(layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(layers.Dense(units=3))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')

    return model

# Crea il tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=3,  # esegue 5 prove / 3 prove
    executions_per_trial=3,  # esegue ogni modello 3 volte oppure 5 volte
    directory='my_dir',
    project_name='helloworld')

# Esegui il tuning degli iperparametri
tuner.search(X, y, epochs=10, validation_split=0.2, verbose=1)

# Ottieni il modello ottimizzato
best_model = tuner.get_best_models(num_models=1)[0]

# Prepara i dati degli ultimi giorni
inputs = data.values[-look_back:]
inputs = scaler.transform(inputs)

# Crea il set di dati di input per la previsione
X_test = []
X_test.append(inputs)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], data.shape[1]))

# Fai la previsione del prezzo dell'azione per i prossimi 3 giorni
predicted_prices = best_model.predict(X_test)

# Create an empty array with the same shape as the input data (but with 3 rows for our 3 predicted days)
zeros_data = np.zeros((3, data.shape[1]))

# Put the predicted values in the right place
zeros_data[:,3] = predicted_prices.ravel()

# Inverse transformation
predicted_prices = scaler.inverse_transform(zeros_data)[:, 3]

# Calcola le date dei prossimi 3 giorni
last_date = data.index[-1]
prediction_dates = [last_date + timedelta(days=i) for i in range(1, 4)]

# Stampa le previsioni con le rispettive date
for i, price in enumerate(predicted_prices):
    print(f'Il prezzo previsto per {prediction_dates[i].date()} è: {price}')
