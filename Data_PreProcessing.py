import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import itertools
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)


# --------------------File Existence Check------------------------------------------------------------
def CheckFileExistence(filePath):
    file = Path(filePath)
    if not file.is_file():
        raise Exception("Oops! File Does Not Exist")


# ----------------------------------------------------------------------------------------------------


# ------------------------------Check Missing Data in DataSet -------------------------------------------

def CheckMissingData(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! it is not a DataFrame")
    MissingData = False
    NaNRows = DataFrame.isnull()
    NaNRows = NaNRows.any(axis=1)
    DataFrameWithNaN = DataFrame[NaNRows]
    num_rows_with_nan = DataFrameWithNaN.shape[0]
    if num_rows_with_nan != 0:
        raise Exception("Oops! Some Data is missing in DataSet")
    return MissingData


# -------------------------------------------------------------------------------------------------------


# ------------------Read Data From File ---------------------------------------------------------------
def ReafFile(filePath):
    file = Path(filePath)
    if not file.is_file():
        raise Exception("Oops! File Does Not Exist")
    DataFrame = pd.read_csv(filePath)
    return DataFrame


# -------------------------------------------------------------------------------------------------------


# ------------------Loading and Concatenating Data ------------------------------------------------------

def load_and_concatenate(train_path, test_path):
    try:
        df_train = pd.read_csv(train_path)
    except pd.errors.EmptyDataError:
        df_train = pd.DataFrame()
    try:
        df_test = pd.read_csv(test_path)
    except pd.errors.EmptyDataError:
        df_test = pd.DataFrame()
    return pd.concat([df_train, df_test], ignore_index=True)



# -------------------------------------------------------------------------------------------------------


# ------------------Removing Unnecessary Columns ------------------------------------------------------

def process_columns(df):
    expected_columns = ['satisfaction', 'Arrival Delay in Minutes']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns: {', '.join(missing_columns)}")
    ds = df.iloc[:, 2:]
    if 'satisfaction' in ds.columns:
        ds['satisfaction'] = ds['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1})
    else:
        raise KeyError("Column 'satisfaction' is required but not present in the DataFrame.")
    if 'Arrival Delay in Minutes' in ds.columns:
        ds['Arrival Delay in Minutes'] = ds['Arrival Delay in Minutes'].fillna(ds['Arrival Delay in Minutes'].mean())
    else:
        raise KeyError("Column 'Arrival Delay in Minutes' is required but not present in the DataFrame.")

    return ds



# -------------------------------------------------------------------------------------------------------


# ------------------Encoding Data ------------------------------------------------------

def encode_categorical(ds):
    if 'satisfaction' not in ds.columns:
        raise KeyError("Missing 'satisfaction' column")
    categorical_features = ds.select_dtypes(include=['object']).columns
    lencoders = {}
    for col in categorical_features:
        lencoders[col] = LabelEncoder()
        ds[col] = lencoders[col].fit_transform(ds[col])
        mapping = dict(zip(lencoders[col].classes_, lencoders[col].transform(lencoders[col].classes_)))
        # print(f"Encoding for {col}: {mapping}")


# -------------------------------------------------------------------------------------------------------

# ------------------Preparing Data ------------------------------------------------------

def prepare_data(train_path, test_path):
    df = load_and_concatenate(train_path, test_path)
    ds = process_columns(df)
    encode_categorical(ds)
    return ds


# -------------------------------------------------------------------------------------------------------


ds = prepare_data('./DataSet/train.csv', './DataSet/test.csv')
pd.set_option('display.max_columns', None)

# print(ds.head())


# -----------------------------To Get the Numbers of Column from DataSet
def DisplayNumbersOfColumns(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.shape[1]

# print(DisplayNumbersOfColumns(DataFrame))
# -------------------------------------------------------------------------------------------------

# --------------------------------Display the Numbers of rows in DataSet---------------------------
def DisplayNumbersOfRows(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.shape[0]

# print(DisplayNumbersOfRows(DataFrame))
# -------------------------------------------------------------------------------------------------

# ------------------------------Get the Names of Colums from DataSet ------------------------------

def GetColumnsName(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.columns.tolist()


# ---------------------------------------------------------------------------------------------------

# ------------------------------Check the Varience of Columns---------------------------------------

def CheckColumnsVarience(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.var()

# print(CheckColumnsVarience(DataFrame))
# ----------------------------------------------------------------------------------------------------

# -------------------------------Correlation Check of Given Data Set----------------------------------

def CheckCorrelation(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    DataFrameIntegerFloatsColumns = DataFrame.select_dtypes(include=['int64','float64'])
    correlation = {}
    columns = DataFrameIntegerFloatsColumns.columns.tolist()
    # print(columns)
    for col_a, col_b in itertools.combinations(columns, 2):
        correlation[col_a + '__'+ col_b] = pearsonr(DataFrameIntegerFloatsColumns.loc[:,col_a],DataFrameIntegerFloatsColumns.loc[:,col_b])

    # print(correlation)
    FinalResult = pd.DataFrame.from_dict(correlation, orient='index')
    FinalResult.columns = ['PCC', 'p-value']
    return FinalResult.columns

# print(CheckCorrelation(DataFrame))

# ------------------------------------------------------------------------------------------------------

# ------------------------------Getting Minimum and Maximum values from Each Col----------------------------------
def print_min_max_values(DataFrame):
    for column in DataFrame.columns:
        min_value = DataFrame[column].min()
        max_value = DataFrame[column].max()
        # print(f"Column: {column}, Min: {min_value}, Max: {max_value}")

# ------------------------------------------------------------------------------------------------------


# --------------------------Check the Plauseability of Given DataSet------------------------------------
def check_plausibility_data_set(Data_Frame):
    if type(Data_Frame).__name__ != "DataFrame":
        raise Exception("Input is not a DataFrame")

    column_ranges = {
        'Gender': (0, 1),
        'Customer Type': (0, 1),
        'Age': (7, 85),
        'Type of Travel': (0, 1),
        'Class': (0, 2),
        'Flight Distance': (31, 4983),
        'Inflight wifi service': (0, 5),
        'Departure/Arrival time convenient': (0, 5),
        'Ease of Online booking': (0, 5),
        'Gate location': (0, 5),
        'Food and drink': (0, 5),
        'Online boarding': (0, 5),
        'Seat comfort': (0, 5),
        'Inflight entertainment': (0, 5),
        'On-board service': (0, 5),
        'Leg room service': (0, 5),
        'Baggage handling': (1, 5),
        'Checkin service': (0, 5),
        'Inflight service': (0, 5),
        'Cleanliness': (0, 5),
        'Departure Delay in Minutes': (0, 1592),
        'Arrival Delay in Minutes': (0.0, 1584.0),
        'satisfaction': (0, 1)
    }

    plausibility_values = {}

    for column, (min_val, max_val) in column_ranges.items():
        invalid_count = sum((Data_Frame[column] < min_val) | (Data_Frame[column] > max_val))
        if invalid_count > 0:
            plausibility_values[column] = invalid_count

    if plausibility_values:
        for key, value in plausibility_values.items():
            raise Exception(f"Data issue found: {key} has {value} incorrect entries.")

    return not bool(plausibility_values)

# DaaFrame = prepare_data("./DataSet/train.csv", "./DataSet/test.csv")
# print(check_plausibility_data_set(DaaFrame))

# ---------------------------------------------------------------------------------------------------------------
