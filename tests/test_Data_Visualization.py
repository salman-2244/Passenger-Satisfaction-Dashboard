import numpy as np
import plotly.express as px
import pytest
import pandas as pd
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from Data_Visualization import plot_age_vs_satisfaction, plot_roc_curve, plot_model_performance
from sklearn.model_selection import train_test_split

@pytest.fixture
def testing_dataframe():
    return pd.DataFrame({
        'Age': [22, 45, 30, 35, 40],
        'satisfaction': ['satisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied']
    })


@pytest.fixture
def testing_dataset():
    return pd.DataFrame({
        'satisfaction': ['satisfied', 'neutral', 'dissatisfied', 'satisfied', 'neutral'],
        'Checkin service': [1, 3, 5, 2, 4],
        'Inflight service': [5, 3, 2, 4, 1]
    })


@pytest.fixture
def model_and_data():
    data = pd.DataFrame({
        'feature1': [0, 1, 0, 1],
        'feature2': [1, 0, 1, 0]
    })
    target = [0, 1, 0, 1]

    model = RandomForestClassifier()
    model.fit(data, target)

    return model, data, target


def test_plot_roc_curve(model_and_data):
    model, features_test, target_test = model_and_data
    fig = plot_roc_curve(model, features_test, target_test)


@pytest.fixture
def testing_dataframe():
    return pd.DataFrame({
        'Age': [22, 45, 30, 35, 40],
        'satisfaction': ['satisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied']
    })


@pytest.fixture
def edge_case_dataframe():
    return pd.DataFrame({
        'Age': [25.5, 'thirty', 40, None, 50],
        'satisfaction': ['happy', 'sad', np.nan, 'neutral or dissatisfied', 'satisfied']
    })


@pytest.fixture
def empty_dataframe():
    return pd.DataFrame()


def test_plot_age_vs_satisfaction_normal(testing_dataframe):
    fig = plot_age_vs_satisfaction(testing_dataframe)
    assert not fig.data == []  


def test_plot_age_vs_satisfaction_empty(empty_dataframe):
    with pytest.raises(ValueError):
        plot_age_vs_satisfaction(empty_dataframe)


@pytest.fixture
def sample_data():
    data = {
        'Age': [25, 30, 35, 40, 45, 50],
        'satisfaction': ['satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied',
                         'neutral or dissatisfied']
    }
    return pd.DataFrame(data)


def test_plot_axis_titles(sample_data):
    fig = plot_age_vs_satisfaction(sample_data)
    assert fig.layout.xaxis.title.text == "Age", "X-axis title should be 'Age'."
    assert fig.layout.yaxis.title.text == "Count", "Y-axis title should be 'Count'."


def test_plot_color_mapping(sample_data):
    fig = plot_age_vs_satisfaction(sample_data)
    assert fig.data[0].marker.color == '#0091D5', "Colors for 'satisfied' should match the specified color map."
    assert fig.data[
               1].marker.color == '#EA6A47', "Colors for 'neutral or dissatisfied' should match the specified color map."


def test_plot_age_range(sample_data):
    fig = plot_age_vs_satisfaction(sample_data)
    age_range = list(range(5, 80))
    all_ages_in_data = all(age in age_range for age in sample_data['Age'])
    assert all_ages_in_data, "All age data should be within the specified range."



def create_dummy_classifier_and_data():
    X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def test_roc_curve_output_type():
    model, features_test, target_test = create_dummy_classifier_and_data()
    fig = plot_roc_curve(model, features_test, target_test)
    assert isinstance(fig, go.Figure), "The function should return a Plotly Figure object."

def test_roc_curve_with_valid_classifier():
    model, features_test, target_test = create_dummy_classifier_and_data()
    fig = plot_roc_curve(model, features_test, target_test)
    assert 'ROC curve' in fig.data[0].name, "ROC curve trace should be in the plot data."

@pytest.mark.parametrize("data_type", [pd.DataFrame, np.ndarray])
def test_input_data_types(data_type):
    model, features_test, target_test = create_dummy_classifier_and_data()
    features_test = data_type(features_test) if data_type is pd.DataFrame else features_test
    fig = plot_roc_curve(model, features_test, target_test)
    assert isinstance(fig, go.Figure), "Function should handle both DataFrame and ndarray inputs."

def test_model_name_customization():
    model, features_test, target_test = create_dummy_classifier_and_data()
    custom_name = "Custom Model"
    fig = plot_roc_curve(model, features_test, target_test, model_name=custom_name)
    assert custom_name in fig.layout.annotations[0].text, "Custom model name should be displayed in the plot annotations."

def test_roc_curve_chance_line():
    model, features_test, target_test = create_dummy_classifier_and_data()
    fig = plot_roc_curve(model, features_test, target_test)
    assert 'Chance (baseline)' in fig.data[1].name, "Chance line should be included in the plot."



@pytest.fixture
def sample_model_stats():
    return {
        'Model': ['Model A', 'Model B', 'Model C'],
        'Time taken': [120, 150, 180],
        'Accuracy': [0.85, 0.90, 0.88],
        'ROC_AUC': [0.92, 0.93, 0.91]
    }

def test_plot_output_type(sample_model_stats):
    fig = plot_model_performance(sample_model_stats)
    assert isinstance(fig, go.Figure), "The function should return a Plotly Figure object."

def test_data_representation_in_plot(sample_model_stats):
    fig = plot_model_performance(sample_model_stats)
    trace_names = [trace.name for trace in fig.data]
    assert "Time Taken (s)" in trace_names, "Time Taken should be represented as a bar chart."
    assert "Accuracy" in trace_names, "Accuracy should be plotted on a secondary y-axis."
    assert "ROC AUC" in trace_names, "ROC AUC should be plotted on a secondary y-axis."

def test_plot_axis_titles(sample_model_stats):
    fig = plot_model_performance(sample_model_stats)
    assert fig.layout.xaxis.title.text == "Model", "X-axis title should be 'Model'."
    assert fig.layout.yaxis.title.text == "<b>Time Taken (s)</b>", "Primary Y-axis title incorrect."
    assert fig.layout.yaxis2.title.text == "<b>Accuracy / ROC AUC</b>", "Secondary Y-axis title incorrect."

def test_plot_legends(sample_model_stats):
    fig = plot_model_performance(sample_model_stats)
    legend_labels = [legend_entry.name for legend_entry in fig.data]
    expected_labels = ['Time Taken (s)', 'Accuracy', 'ROC AUC']
    assert all(label in legend_labels for label in expected_labels), "All expected legends must be present."

