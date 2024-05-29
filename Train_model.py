import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from Data_PreProcessing import prepare_data



#Ensure the models directory exists
model_dir = os.path.join(os.path.dirname(__file__), '.', 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Loading and preprocessing data
df = prepare_data('./DataSet/train.csv', './DataSet/test.csv')

# Define features and target
features = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]
X = df[features]
y = df['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Saving  the scaler
scaler_path = os.path.join(model_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Building and training the model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)

batch_size = 64
NN_Classifier = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, callbacks=[es], validation_split=0.2)

model.save(os.path.join(model_dir, 'ANN_model.h5'))

# Saving the training history
history_path = os.path.join(model_dir, 'history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(NN_Classifier.history, f)

y_pred = (model.predict(X_test) > 0.5).astype("int32")

