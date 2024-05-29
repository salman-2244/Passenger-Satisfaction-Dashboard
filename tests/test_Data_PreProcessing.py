import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Data_PreProcessing import CheckFileExistence, CheckMissingData, ReafFile, load_and_concatenate, process_columns, encode_categorical, prepare_data, check_plausibility_data_set

# ------------------ Fixtures ------------------
@pytest.fixture
def s_dataframe():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'satisfaction': ['neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied'],
        'Arrival Delay in Minutes': [10, None, 5]
    })

# ------------------ Test Cases ------------------

def test_CheckFileExistence_missing_file():
    with pytest.raises(Exception) as e:
        CheckFileExistence('nonexistentfile.csv')
    assert str(e.value) == "Oops! File Does Not Exist"

# def test_CheckMissingData_no_missing_data(s_dataframe):
#     assert CheckMissingData(s_dataframe) == False

def test_CheckMissingData_with_missing_data():
    df = pd.DataFrame({'A': [1, None, 3]})
    with pytest.raises(Exception) as e:
        CheckMissingData(df)
    assert str(e.value) == "Oops! Some Data is missing in DataSet"

def test_ReafFile_non_existent():
    with pytest.raises(Exception) as e:
        ReafFile('path/that/does/not/exist.csv')
    assert str(e.value) == "Oops! File Does Not Exist"

def test_load_and_concatenate():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    df1.to_csv('temp_df1.csv', index=False)
    df2.to_csv('temp_df2.csv', index=False)
    result = load_and_concatenate('temp_df1.csv', 'temp_df2.csv')
    assert len(result) == 4
    assert list(result['A']) == [1, 2, 3, 4]

def test_process_columns(s_dataframe):
    processed = process_columns(s_dataframe)
    assert processed['satisfaction'].equals(pd.Series([0, 1, 0]))

def test_encode_categorical(s_dataframe):
    encode_categorical(s_dataframe)
    assert 'satisfaction' in s_dataframe.columns
    assert s_dataframe['satisfaction'].dtype == int



def test_CheckFileExistence_existing_file(tmp_path):
    d = tmp_path / "example.csv"
    d.write_text("data")
    assert CheckFileExistence(str(d)) is None  

def test_ReafFile_correct_format(tmp_path):
    d = tmp_path / "valid.csv"
    d.write_text("A,B\n1,2\n3,4")
    df = ReafFile(str(d))
    assert not df.empty
    assert list(df.columns) == ['A', 'B']

def test_load_and_concatenate_empty_files(tmp_path):
    df1 = tmp_path / "empty1.csv"
    df2 = tmp_path / "empty2.csv"
    df1.write_text("")
    df2.write_text("")
    result = load_and_concatenate(str(df1), str(df2))
    assert result.empty



def test_encode_categorical_missing_column(s_dataframe):
    s_dataframe.drop('satisfaction', axis=1, inplace=True)
    with pytest.raises(KeyError):
        encode_categorical(s_dataframe)


def test_prepare_data_invalid_paths():
    with pytest.raises(Exception):
        prepare_data('invalid/train_path.csv', 'invalid/test_path.csv')


def test_check_plausibility_data_set_incorrect_data(s_dataframe):
    s_dataframe.loc[0, 'Age'] = 1000  
    with pytest.raises(Exception):
        check_plausibility_data_set(s_dataframe)

def test_CheckMissingData_invalid_input_type():
    with pytest.raises(Exception) as e:
        CheckMissingData([1, 2, 3])
    assert str(e.value) == "Oops! it is not a DataFrame"


def test_process_columns_missing_critical_columns():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    with pytest.raises(KeyError) as excinfo:
        process_columns(df)
    assert "Missing expected columns: satisfaction, Arrival Delay in Minutes" in str(excinfo.value)

def test_process_columns_valid_input():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'satisfaction': ['neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied'],
        'Arrival Delay in Minutes': [10, None, 5]
    })
    processed_df = process_columns(df)
    assert all(processed_df['satisfaction'] == pd.Series([0, 1, 0], index=[0, 1, 2]))
    assert processed_df['Arrival Delay in Minutes'].isnull().sum() == 0  
    assert processed_df['Arrival Delay in Minutes'].iloc[1] == (10 + 5) / 2 

def test_process_columns_all_columns_present_but_not_included():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'satisfaction': ['neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied'],
        'Arrival Delay in Minutes': [10, None, 5]
    })
    processed_df = process_columns(df)
    assert 'A' not in processed_df.columns and 'B' not in processed_df.columns
    assert 'satisfaction' in processed_df.columns and 'Arrival Delay in Minutes' in processed_df.columns



def test_check_plausibility_not_a_dataframe():
    with pytest.raises(Exception) as excinfo:
        check_plausibility_data_set([1, 2, 3]) 
    assert "Input is not a DataFrame" in str(excinfo.value)

def test_check_plausibility_valid_data():
    data = {
        'Gender': [1, 0],
        'Customer Type': [1, 0],
        'Age': [25, 34],
        'Type of Travel': [1, 0],
        'Class': [1, 2],
        'Flight Distance': [100, 3000],
        'Inflight wifi service': [3, 2],
        'Departure/Arrival time convenient': [4, 3],
        'Ease of Online booking': [3, 5],
        'Gate location': [4, 2],
        'Food and drink': [3, 4],
        'Online boarding': [4, 3],
        'Seat comfort': [3, 4],
        'Inflight entertainment': [4, 5],
        'On-board service': [4, 5],
        'Leg room service': [3, 4],
        'Baggage handling': [3, 4],
        'Checkin service': [4, 5],
        'Inflight service': [5, 4],
        'Cleanliness': [5, 4],
        'Departure Delay in Minutes': [10, 15],
        'Arrival Delay in Minutes': [20.0, 30.0],
        'satisfaction': [0, 1]
    }
    df = pd.DataFrame(data)
    assert check_plausibility_data_set(df) == True, "Data should be valid and return True"
