import os
from tensorflow.keras.models import load_model
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pickle

FEATURE_COLUMNS = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]


# ------------------------Random Forest Regression for Satisfaction----------------------------

def RandomForestClassificationForSatisfaction(df):

    # Features and target variable selection
    X = df[FEATURE_COLUMNS]
    y = df['satisfaction']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_time = time.time()


    # Creating and train the RandomForest classifier
    classifier = RandomForestClassifier(n_estimators=50, random_state=42, criterion="gini")
    classifier.fit(X_train, y_train)

    # Making predictions
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]


    # Computing the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Computing ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_prob)

    end_time = time.time()
    time_taken = end_time - start_time

    # Returning results
    return accuracy, roc_auc, classifier, X_test, y_test, y_pred, time_taken


# ------------------------XGBoost Model for Satisfaction----------------------------

def XGBoostClassificationForSatisfaction(df):

    # Features and target variable selection
    X = df[FEATURE_COLUMNS]
    y = df['satisfaction']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_time = time.time()
    # Creating an XGBoost classifier
    classifier = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, seed=42)
    classifier.fit(X_train, y_train)

    # Making predictions
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Computing the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Computing ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_prob)


    end_time = time.time()
    time_taken = end_time - start_time

    return accuracy, roc_auc, classifier, X_test, y_test, y_pred, time_taken

# ------------------------AdaBoost Model for Satisfaction----------------------------

def AdaBoostClassificationForSatisfaction(df):
    # Features and target variable selection
    X = df[FEATURE_COLUMNS]
    y = df['satisfaction']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_time = time.time()
    # Creating and train the AdaBoost classifier
    classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
    classifier.fit(X_train, y_train)

    # Making predictions
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]


    # Computing the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Computing ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_prob)

    end_time = time.time()
    time_taken = end_time - start_time


    return accuracy, roc_auc, classifier, X_test, y_test, y_pred, time_taken

# ------------------------Decision Tree Model for Satisfaction----------------------------

def DecisionTreeClassificationForSatisfaction(df, criterion='gini', max_depth=None):

    # Features and target variable selection
    X = df[FEATURE_COLUMNS]
    y = df['satisfaction']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_time = time.time()
    # Createing and train the Decision Tree classifier
    classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    classifier.fit(X_train, y_train)

    # Making predictions
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]


    # Computing the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Computing ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_prob)

    end_time = time.time()
    time_taken = end_time - start_time

    return accuracy, roc_auc, classifier, X_test, y_test, y_pred, time_taken


# ------------------------Artificial Neural Network Model for Satisfaction----------------------------

def ANNClassificationForSatisfaction(df):

    # Features and target variable selection
    X = df[FEATURE_COLUMNS]
    y = df['satisfaction']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the scaler
    model_dir = os.path.join(os.path.dirname(__file__), '.', 'model')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Scale the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Load the pre-trained ANN model
    model_path = os.path.join(model_dir, './ANN_model.h5')
    model = load_model(model_path)

    start_time = time.time()

    # Making predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    y_prob = model.predict(X_test).flatten()

    # Computing the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Computing ROC-AUC score
    roc_aucANN = roc_auc_score(y_test, y_prob)

    end_time = time.time()
    time_taken = end_time - start_time

    # Return results
    return accuracy, roc_aucANN, model, X_test, y_test, y_pred, time_taken



