import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


from ML_Models import RandomForestClassificationForSatisfaction, AdaBoostClassificationForSatisfaction, \
    DecisionTreeClassificationForSatisfaction, ANNClassificationForSatisfaction, XGBoostClassificationForSatisfaction


def create_data():
    np.random.seed(42)
    data = {
        'Gender': np.random.choice([0, 1], size=100),
        'Customer Type': np.random.choice([0, 1], size=100),
        'Age': np.random.randint(18, 70, size=100),
        'Type of Travel': np.random.choice([0, 1], size=100),
        'Class': np.random.choice([0, 2], size=100),
        'Flight Distance': np.random.randint(31, 4983, size=100),
        'Inflight wifi service': np.random.randint(0, 5, size=100),
        'Departure/Arrival time convenient': np.random.randint(0, 5, size=100),
        'Ease of Online booking': np.random.randint(0, 5, size=100),
        'Gate location': np.random.randint(0, 5, size=100),
        'Food and drink': np.random.randint(0, 5, size=100),
        'Online boarding': np.random.randint(0, 5, size=100),
        'Seat comfort': np.random.randint(0, 5, size=100),
        'Inflight entertainment': np.random.randint(0, 5, size=100),
        'On-board service': np.random.randint(0, 5, size=100),
        'Leg room service': np.random.randint(0, 5, size=100),
        'Baggage handling': np.random.randint(1, 5, size=100),
        'Checkin service': np.random.randint(0, 5, size=100),
        'Inflight service': np.random.randint(0, 5, size=100),
        'Cleanliness': np.random.randint(0, 5, size=100),
        'Departure Delay in Minutes': np.random.randint(0, 1592, size=100),
        'Arrival Delay in Minutes': np.random.randint(0.0, 1584.0, size=100),
        'satisfaction': np.random.choice([0, 1], size=100)
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def simulated_data():
    return create_data()

def test_random_forest_model_training_and_prediction(simulated_data):
    accuracy, roc_auc, classifier, X_test, y_test, y_pred, time_taken = RandomForestClassificationForSatisfaction(simulated_data)
    assert isinstance(classifier, RandomForestClassifier)
    assert 0 <= accuracy <= 1
    assert 0 <= roc_auc <= 1
    assert len(y_pred) == len(y_test)
    assert isinstance(time_taken, float)

class XGBoostClassification:
    pass



def test_adaboost_model_training_and_prediction(simulated_data):
    accuracy, roc_auc, classifier, X_test, y_test, y_pred, time_taken = AdaBoostClassificationForSatisfaction(
        simulated_data)
    assert isinstance(classifier, AdaBoostClassifier)
    assert 0 <= accuracy <= 1
    assert 0 <= roc_auc <= 1
    assert len(y_pred) == len(y_test)
    assert isinstance(time_taken, float)


def test_decision_tree_model_training_and_prediction(simulated_data):
    accuracy, roc_auc, classifier, X_test, y_test, y_pred, time_taken = DecisionTreeClassificationForSatisfaction(
        simulated_data, criterion='gini', max_depth=None)
    assert isinstance(classifier, DecisionTreeClassifier)
    assert 0 <= accuracy <= 1
    assert 0 <= roc_auc <= 1
    assert len(y_pred) == len(y_test)
    assert isinstance(time_taken, float)


def test_ann_model_training_and_prediction(simulated_data):
    accuracy, roc_aucANN, model, X_test, y_test, y_pred, time_taken = ANNClassificationForSatisfaction(simulated_data)
    assert str(type(model)).find('keras') != -1  # Check for Keras model
    assert 0 <= accuracy <= 1
    assert 0 <= roc_aucANN <= 1
    assert len(y_pred) == len(y_test)
    assert isinstance(time_taken, float)

def test_random_forest_invalid_input():
    with pytest.raises(Exception):
        RandomForestClassificationForSatisfaction(pd.DataFrame())

def test_xgboost_missing_columns(simulated_data):
    simulated_data = simulated_data.drop(['Gender'], axis=1)
    with pytest.raises(KeyError):
        XGBoostClassificationForSatisfaction(simulated_data)



def test_random_forest_large_dataset():
    large_data = create_data()._append([create_data()] * 10, ignore_index=True)
    _, _, classifier, _, _, _, _ = RandomForestClassificationForSatisfaction(large_data)
    assert isinstance(classifier, RandomForestClassifier)


