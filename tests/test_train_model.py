import pickle
from sklearn.model_selection import train_test_split
from Data_PreProcessing import prepare_data
from Feature_Importance import scaler


def test_directory_creation(tmp_path):
    import os
    model_dir = tmp_path / "model"
    if not model_dir.exists():
        os.makedirs(model_dir)
    assert model_dir.exists(), "Model directory should be created if not present."


def test_data_loading():
    df = prepare_data('./DataSet/train.csv', './DataSet/test.csv')
    assert not df.empty, "Dataframe should not be empty after loading and preprocessing data."
    assert 'satisfaction' in df.columns, "Dataframe must include 'satisfaction' column."


def test_scaler_saving(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "scaler.pkl"
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()  
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    assert scaler_path.exists(), "Scaler file should be saved."


def test_feature_and_target_extraction():
    df = prepare_data('./DataSet/train.csv', './DataSet/test.csv')
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
    assert not X.empty and not y.empty, "Features and target arrays should not be empty."
    assert set(features).issubset(df.columns), "All features should be present in the DataFrame."



def test_train_test_split():
    df = prepare_data('./DataSet/train.csv', './DataSet/test.csv')
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
    assert len(X_train) > 0 and len(X_test) > 0, "Training and testing sets should not be empty."
    assert len(X_train) > len(X_test), "Training set should be larger than testing set."

