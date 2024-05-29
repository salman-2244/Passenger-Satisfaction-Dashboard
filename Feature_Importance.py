import os
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import Data_PreProcessing as dpp
from utils import KerasClassifierWrapper
import numpy as np
import pickle

# Loading and preprocess data
df = dpp.prepare_data('./DataSet/train.csv', './DataSet/test.csv')

# Defining features and target
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

# Loading the saved model
model_path = os.path.join(os.path.dirname(__file__), '.', 'model', 'ANN_model.h5')
model = load_model(model_path)

# Loading the saved scaler
scaler_path = os.path.join(os.path.dirname(__file__), '.', 'model', 'scaler.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Wrapping the Keras model
wrapped_model = KerasClassifierWrapper(model)

#  Setting the classes_ manually
wrapped_model.classes_ = np.unique(y)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the test data
X_test = scaler.transform(X_test)

# Calculating permutation importance
perm_importance = permutation_importance(wrapped_model, X_test, y_test, scoring='accuracy', n_repeats=10, random_state=42)


feature_importance = pd.DataFrame({'Feature': features, 'Importance': perm_importance.importances_mean * 100})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance.set_index('Feature', inplace=True)

print(feature_importance)

# Saving the feature importance in csv
feature_importance.to_csv(os.path.join(os.path.dirname(__file__), '.', 'model', 'feature_importance.csv'))
