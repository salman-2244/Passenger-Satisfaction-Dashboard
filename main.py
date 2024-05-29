import json
import os
import pickle
import dash as dash
import pandas as pd
import numpy as np
import math
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from Data_Visualization import plot_accuracy, plot_loss

# Importing DataPrepProcessing File
import Data_PreProcessing as dpp

# Importing MachineLearning Models File
import ML_Models as ml
from ML_Models import FEATURE_COLUMNS

# Importing DataVisualization File
import Data_Visualization as dv

# Reading the Data From File

DataFrame = dpp.prepare_data('./DataSet/train.csv', './DataSet/test.csv')

# Reading Unlabeled DataFrame

dff = pd.read_csv('./DataSet/train.csv')


def classify_and_extract_metrics(classification_func, data):
    accuracy, roc_auc, model, *_, time_taken = classification_func(data)
    return accuracy, roc_auc, model, time_taken

# These will be later used for Comparing Different Models
classification_funcs = [
    ml.RandomForestClassificationForSatisfaction,
    ml.XGBoostClassificationForSatisfaction,
    ml.AdaBoostClassificationForSatisfaction,
    ml.ANNClassificationForSatisfaction,
    ml.DecisionTreeClassificationForSatisfaction
]

results = {}
for func in classification_funcs:
    model_name = func.__name__.replace("ClassificationForSatisfaction", "")
    results[model_name] = classify_and_extract_metrics(func, DataFrame)

# Accessing results
# metrics for Random Forest Algorithm
accuracy_rf, roc_auc_rf, model_rf, time_rf = results["RandomForest"]
# metrics for XGBoost Algorithm
accuracy_xgb, roc_auc_xgb, model_xgb, time_xgb = results["XGBoost"]
# metrics for AdaBoost Algorithm
accuracy_ab, roc_auc_ab, model_ab, time_ab = results["AdaBoost"]
# metrics for ANN
accuracy_ann, roc_auc_ann, model_ann, time_ann = results["ANN"]
# metrics for Decsion Tree Algorithm
accuracy_dt, roc_auc_dt, model_dt, time_dt = results["DecisionTree"]



model_stats = {
    'Model': ['Random Forest', 'XGBoost', 'AdaBoost', 'Artificial Neural Network', 'Decision Tree'],
    'Accuracy': [accuracy_rf, accuracy_xgb, accuracy_ab, accuracy_ann, accuracy_dt],
    'ROC_AUC': [roc_auc_rf, roc_auc_xgb, roc_auc_ab, roc_auc_ann, roc_auc_dt],
    'Time taken': [time_rf, time_xgb, time_ab, time_ann, time_dt]
}

# Random Forest feature importances
feature_importance_rf = pd.DataFrame(model_rf.feature_importances_ * 100, columns=['Importance'], index=FEATURE_COLUMNS)
feature_importance_rf = feature_importance_rf.sort_values("Importance", ascending=False)

# XGBoost feature importances
feature_importance_xgb = pd.DataFrame(model_xgb.feature_importances_ * 100, columns=['Importance'], index=FEATURE_COLUMNS)
feature_importance_xgb = feature_importance_xgb.sort_values("Importance", ascending=False)

# AdaBoost feature importances
feature_importance_ab = pd.DataFrame(model_ab.feature_importances_ * 100, columns=['Importance'], index=FEATURE_COLUMNS)
feature_importance_ab = feature_importance_ab.sort_values("Importance", ascending=False)

# Decision Tree feature importances
feature_importance_dt = pd.DataFrame(model_dt.feature_importances_ * 100, columns=['Importance'], index=FEATURE_COLUMNS)
feature_importance_dt = feature_importance_dt.sort_values("Importance", ascending=False)

Comparison_Graph = dv.plot_model_performance(model_stats)



# Feature Importance Figures

FetureImportanceFigure = dv.DrawFeatureImportanceGraph(feature_importance_rf)

feature_importance_ANN = pd.read_csv('./model/feature_importance.csv', index_col='Feature')
FetureImportanceFigureANN = dv.DrawFeatureImportanceGraph(feature_importance_ANN)

XGBoostFetureImportanceFigure = dv.DrawFeatureImportanceGraph(feature_importance_xgb)

AdaBoostFetureImportanceFigure = dv.DrawFeatureImportanceGraph(feature_importance_ab)

DecisionTreeFetureImportanceFigure = dv.DrawFeatureImportanceGraph(feature_importance_dt)


# feature importance lists for each model
feature_mapping = {
    "RandomForest": feature_importance_rf.index[:5],
    "XGBoost": feature_importance_xgb.index[:5],
    "AdaBoost": feature_importance_ab.index[:5],
    "ANN": feature_importance_ANN.index[:5],
    "DecisionTree": feature_importance_dt.index[:5]
}


# Loading the training history

model_dir = os.path.join(os.path.dirname(__file__), '.', 'model')
history_path = os.path.join(model_dir, 'history.pkl')

with open(history_path, 'rb') as f:
    history = pickle.load(f)

# Extracting data from history

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = list(range(len(acc)))

# Generating Plots

training_validation_accuracy =plot_accuracy(epochs, acc, val_acc)

training_validation_loss = plot_loss(epochs, loss, val_loss)

age_satisfaction = dv.plot_age_vs_satisfaction(dff)

boxplot_flight_entertainment = dv.plot_boxplot(dff, "Inflight entertainment", "Flight Distance")

hist_flight_entertainment = dv.plot_histogram_stacked(dff, "Flight Distance", "Inflight entertainment")

boxplot_flight_entertainment2 = dv.plot_boxplot(dff, "Leg room service", "Flight Distance")

hist_flight_entertainment2 = dv.plot_histogram_stacked(dff, "Flight Distance", "Leg room service")

satisfaction_delay = dv.plot_satisfaction_delay(dff)

satisfaction_distribution=dv.plot_satisfaction_distribution(dff)


cleanliness_distribution = dv.plot_cleanliness_distribution(DataFrame)

cleanliness_satisfaction = dv.plot_cleanliness_satisfaction_rate(DataFrame)

Class_Clealiness = dv.plot_gate_class_distribution(dff)

comfort_legroom_satisfaction= dv.plot_comfort_and_legroom_satisfaction(dff)


# Generating Slider values for Live Predictions
def generate_slider_values(features, df):
    slider_vals = {}
    for feature in features:
        min_value = math.floor(df[feature].min())
        mean_value = round(df[feature].mean())
        max_value = round(df[feature].max())
        slider_vals[feature] = {'min': min_value, 'mean': mean_value, 'max': max_value}
    return slider_vals



reverse_mappings = {
    'Customer Type': {0: 'Loyal Customer', 1: 'Disloyal Customer'},
    'Gender': {0: 'Female', 1: 'Male'},
    'Type of Travel': {0: 'Business Travel', 1: 'Personal Travel'},
    'Class': {0: 'Business', 1: 'Eco', 2: 'Eco Plus'}
}



# Code for Dash App

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



# Header Components

HeaderComponent = html.Div("DashBoard for Satisfaction Prediction", className='header')
HeaderComponent2 = html.Div("DashBoard for Visualization", className='header')

# Empty Component for Spacing Purposes

EmptyBoxComponent = html.Div([
    html.Div(
        [

        ]
    )
], className='empty-box')


SliceTypeDisplayComponent = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                html.Div(id='slider-container', className='div1'),
                type="default"
            )
        ], width=4),
        dbc.Col([
            dcc.Loading(
                html.Div([
                    html.Div("Content Box 1", className='content-box accuracy-display', id='content-box-1',
                             style={'background-color': '#1C4E80', 'margin-bottom': '10px'}),
                    html.Div("Content Box 2", className='content-box accuracy-display', id='content-box-2',
                             style={'background-color': '#1C4E80', 'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='feature-graph-1', className='confusion-roc')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='feature-graph-2', className='confusion-roc')
                        ], width=6)
                    ], className='graph-row')
                ], className='box-container', style={'margin-bottom': '20px'}),
                type="default"
            )
        ], width=8)
    ])
])




def create_details_display(button_text, details_text, button_id, collapse_id):
    return html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Button(
                    button_text,
                    id=button_id,
                    color="primary",
                    className="mb-2 w-100"
                ),
                width=4,
                className="mx-auto"
            ),
        ], justify="center"),
        dbc.Row([
            dbc.Col(
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(details_text),
                        style={'margin': 8}
                    ),
                    id=collapse_id,
                    is_open=False
                ),
                width=12
            ),
        ]),
    ])




Age_Satisfaction_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=age_satisfaction, className="graphs-style")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(
            create_details_display(
                button_text="Details",
                details_text=(
                    ' Analyzing this graph, we observe that the proportion of dissatisfied passengers is considerably higher than that'
                    ' of satisfied passengers within the age ranges of 7-to-38 and 61-to-79. This trend indicates a greater level of dissatisfaction'
                    ' among these age groups. On the other hand, in the age range of 39-to-60, the situation reverses, with the proportion of satisfied'
                    ' passengers surpassing that of dissatisfied ones. This suggests that passengers in this middle age group tend to have a more positive'
                    ' experience compared to the younger and older age groups.'
                ),
                button_id="collapse-details-info-btn-1",
                collapse_id="collapse-casual-info-1"
            ), width=12
        )
    ]),
], fluid=True)



Inflight_Entertainment_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=boxplot_flight_entertainment, className="graphs-style")
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=hist_flight_entertainment, className="graphs-style")
        ], width=6)
    ]),

], fluid=True)



Cleanliness_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=cleanliness_satisfaction, className="graphs-style")
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=cleanliness_distribution, className="graphs-style")
        ], width=6)
    ]),
], fluid=True)


Cleanliness_Class_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=Class_Clealiness, className="graphs-style")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(
            create_details_display(
                button_text="Details",
                details_text=(
                    "A clear pattern emerges when examining cleanliness ratings. As cleanliness ratings increase, so does passenger"
                    " satisfaction. However, a significant portion of business travelers have rated cleanliness with scores of 1, 2,"
                    " or 3, indicating that cleanliness may not be meeting their expectations. In contrast, economy travelers tend to"
                    " give higher ratings, ranging from 3 to 5, suggesting that cleanliness is generally satisfactory for this group."
                    " This insight suggests that there is room for improvement in cleanliness, especially for business travelers,"
                    " which could lead to higher ratings and increased passenger satisfaction."
                ),
                button_id="collapse-details-info-btn-6",
                collapse_id="collapse-casual-info-6"
            ), width=12
        )
    ]),
], fluid=True)



Legroom_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=boxplot_flight_entertainment2, className="graphs-style")
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=hist_flight_entertainment2, className="graphs-style")
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col(
            create_details_display(
                button_text="Details",
                details_text=(
                    "Passengers traveling farther and staying in flight longer generally report higher satisfaction with in-flight"
                    " entertainment and extra legroom. This trend indicates that those on longer flights value these amenities more,"
                    " likely because they have more time to use the entertainment and enjoy the added comfort of more legroom."
                    " Therefore, for airlines, investing in these features can be especially beneficial for long-haul "
                    "flights, significantly boosting passenger satisfaction during extended trips."
                ),
                button_id="collapse-details-info-btn-2",
                collapse_id="collapse-casual-info-2"
            ), width=12
        )
    ]),
], fluid=True)



Satisfaction_Delay_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=satisfaction_delay, className="graphs-style")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(
            create_details_display(
                button_text="Details",
                details_text=(
                    "You can observe that the points align more or less along a straight line from the lower left corner"
                    " to the upper right. This indicates that, to some extent, the relationship between arrival time delay"
                    " and departure time delay is linear.The results are logical and can be explained as follows: if an airline's flight"
                    " is delayed by a certain amount of time at departure, it will likely experience a similar delay upon landing, assuming"
                    " the aircraft does not speed up during the flight to compensate for the lost time."
                ),
                button_id="collapse-details-info-btn-3",
                collapse_id="collapse-casual-info-3"
            ), width=12
        )
    ]),
], fluid=True)



Satisfaction_Distribution_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=satisfaction_distribution, className="graphs-style")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(
            create_details_display(
                button_text="Details",
                details_text=(
                                "The graph shows that shorter flights (up to around 1,000 km) have more passengers who are unhappy."
                                 " Satisfaction is highest for mid-range flights (about 1,500 to 2,000 km). For flights longer than 2,000 km,"
                                 " both satisfaction and dissatisfaction decrease. This means that mid-range flights tend to make passengers happier,"
                                 " likely because of ideal travel times and good service. On the other hand, very short or very long flights may have more"
                                 " issues that affect overall satisfaction."

                ),
                button_id="collapse-details-info-btn-4",
                collapse_id="collapse-casual-info-4"
            ), width=12
        )
    ]),
], fluid=True)




Legroom_Comfort_Componnet = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=comfort_legroom_satisfaction, className="graphs-style")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(
            create_details_display(
                button_text="Details",
                details_text=(
                    "From the graphs above, we can conclude the following: most passengers who rated the comfort of the seats and "
                    "the extra legroom at 4 and 5 points out of 5 were satisfied with the flight. This high satisfaction rating"
                    " indicates that these passengers found the seating arrangements and the additional legroom to be very comfortable,"
                    " contributing positively to their overall flight experience. Consequently, it suggests that the quality of seating"
                    " and the amount of legroom are crucial factors in determining passenger satisfaction. When airlines prioritize "
                    "and enhance these features, it is likely to lead to a more enjoyable travel experience for their customers."
                ),
                button_id="collapse-details-info-btn-5",
                collapse_id="collapse-casual-info-5"
            ), width=12
        )
    ]),
], fluid=True)



Model_Choose_Component = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("Choose Model for Prediction", className="text-center mb-4"),  # Added text above the buttons
                html.Div([
                    dbc.Button("Random Forest Regression", id="button-1", className="mb-2 w-100", color="primary", outline=True)
                ], className="d-flex justify-content-center mb-3"),
                html.Div([
                    dbc.Button("eXtreme Gradient Boosting", id="button-2", className="mb-2 w-100", color="primary", outline=True)
                ], className="d-flex justify-content-center mb-3"),
                html.Div([
                    dbc.Button("Adaptive Boosting", id="button-3", className="mb-2 w-100", color="primary", outline=True)
                ], className="d-flex justify-content-center mb-3"),
                html.Div([
                    dbc.Button("Artificial Neural Network", id="button-4", className="mb-2 w-100", color="primary", outline=True)
                ], className="d-flex justify-content-center mb-3"),
                html.Div([
                    dbc.Button("Decision Tree", id="button-5", className="mb-2 w-100", color="primary", outline=True)
                ], className="d-flex justify-content-center mb-3")
            ], className="d-flex flex-column justify-content-around")
        ], width=4),
        dbc.Col([
            html.Div([
                dcc.Graph(id='feature-graph', figure=FetureImportanceFigure)
            ], className='div2', id='graph-container')
        ], width=8)
    ])
])


ANN_Trianing_Component = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=training_validation_accuracy, className="graphs-style")
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=training_validation_loss, className="graphs-style")
        ], width=6)
    ]),
], fluid=True)



Model_Comparison_Component = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='comp-graph', figure=Comparison_Graph)
        ], width=12),
    ], id='main-row')
])


# Footer
FooterComponent = html.Footer(
    html.Div([
        html.Ul([
            html.Li(
                [html.A(
                    html.Img(
                        src=app.get_asset_url('github.png')),
                    href="https://github.com/salman-2244")
                 ]),

            html.Li(
                [html.A(
                    html.Img(
                        src=app.get_asset_url('linkedIn.png')),
                    href="https://www.linkedin.com/in/msalmanahmed01/")
                 ]),

            html.Li(
                [html.A(
                    html.Img(
                        src=app.get_asset_url('facebook.png')),
                    href="https://m.facebook.com/100378568736")
                 ]),

            html.Li(
                [html.A(
                    html.Img(
                        src=app.get_asset_url('instagram.png')),
                    href="https://www.instagram.com/salman.ahmed._/")
                 ]),
            html.Li(
                [html.A(
                    html.Img(
                        src=app.get_asset_url('twitter.png')),
                    href="https://twitter.com/salmanahmed2244")
                 ]),
            html.Li(
                [html.A(
                    html.Img(
                        src=app.get_asset_url('gmail.png')),
                    href="mailto:cs.salman.ahmed@gmail.com")
                 ]),

        ])
    ], className='list-items'),
    className='footer'
)


@app.callback(
    [Output(f"collapse-casual-info-{i}", "is_open") for i in range(1, 7)] +
    [Output(f"collapse-details-info-btn-{i}", "children") for i in range(1, 7)],
    [Input(f"collapse-details-info-btn-{i}", "n_clicks") for i in range(1, 7)],
    [State(f"collapse-casual-info-{i}", "is_open") for i in range(1, 7)]
)
def toggle_collapse(*args):
    ctx = dash.callback_context

    # Checking which button was pressed
    if not ctx.triggered:
        # If no button was clicked, don't change anything
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    states = list(args[len(args)//2:])
    button_idx = int(button_id.split('-')[-1]) if button_id else None

    # Updating is_open state and button labels
    new_is_open = [not states[i] if (i+1) == button_idx else states[i] for i in range(len(states))]
    new_labels = ["Close" if is_open else "Details" for is_open in new_is_open]

    return new_is_open + new_labels


# Contents for Tab-1
tab1_content = html.Div([
    HeaderComponent2,
    EmptyBoxComponent,
    Age_Satisfaction_Component,
    EmptyBoxComponent,
    Inflight_Entertainment_Component,
    EmptyBoxComponent,
    Legroom_Component,
    EmptyBoxComponent,
    Satisfaction_Delay_Component,
    EmptyBoxComponent,
    Satisfaction_Distribution_Component,
    EmptyBoxComponent,
    Legroom_Comfort_Componnet,
    EmptyBoxComponent,
    Cleanliness_Component,
    EmptyBoxComponent,
    Cleanliness_Class_Component,
    EmptyBoxComponent
])



tab2_content = html.Div([
    HeaderComponent,
    html.Div(id='model-id-storage', style={'display': 'none'}),  # Hidden storage for model ID
    Model_Choose_Component,
    SliceTypeDisplayComponent,
    EmptyBoxComponent,
    ANN_Trianing_Component,
    Model_Comparison_Component,
    EmptyBoxComponent

])


app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Visualization Dashboard', value='tab-1'),
        dcc.Tab(label='Prediction Dashboard', value='tab-2')
    ]),
    html.Div(id='tabs-content'),
    FooterComponent
], className='Main-background')



@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1_content
    elif tab == 'tab-2':
        return tab2_content

@app.callback(
    [
        Output('feature-graph', 'figure'),  # Feature Importance Graph
        Output('feature-graph-1', 'figure'),  # ROC Curve Graph
        Output('feature-graph-2', 'figure'),  # Confusion Matrix Graph
        Output('slider-container', 'children'), # Slider container
        Output('content-box-1', 'children'), # Output for accuracy in Content Box 1

    ],
    [
        Input('button-1', 'n_clicks'),
        Input('button-2', 'n_clicks'),
        Input('button-3', 'n_clicks'),
        Input('button-4', 'n_clicks'),
        Input('button-5', 'n_clicks')
    ]
)

def update_features_and_graph(btn1, btn2, btn3, btn4, btn5):
    ctx = dash.callback_context
    if not ctx.triggered:
        # return "Waiting for model selection..."
        button_id = 'button-2'  # default to button 2 since it is fastest
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]


    print("Button ID:", button_id)  # Debugging print

    selected_features, roc_fig, cm_fig, accuracy_val, model = get_model_data(button_id)

    if button_id == "button-1":
        selected_features = feature_mapping["RandomForest"]
        fig = FetureImportanceFigure
        current_model_name = "Random Forest"
    elif button_id == "button-2":
        selected_features = feature_mapping["XGBoost"]
        fig = XGBoostFetureImportanceFigure
        current_model_name = "XG Boost"
    elif button_id == "button-3":
        selected_features = feature_mapping["AdaBoost"]
        fig = AdaBoostFetureImportanceFigure
        current_model_name = "Ada Boost"
    elif button_id == "button-4":
        selected_features = feature_mapping["ANN"]
        fig = FetureImportanceFigureANN
        current_model_name = "Artificial Neural Network"
    elif button_id == "button-5":
        selected_features = feature_mapping["DecisionTree"]
        fig = DecisionTreeFetureImportanceFigure
        current_model_name = "Decsion Tree"

    # Generating slider values and components
    slider_values = generate_slider_values(selected_features, DataFrame)
    slider_divs = [html.H3("Adjust the Sliders for Features", className='heading')]
    for feature in selected_features:
        if feature == "Age":
            # Custom markers for the Age feature, every 10 years
            marks_feature = {
                i: {'label': str(i)}
                for i in range(slider_values[feature]['min'], slider_values[feature]['max'] + 1)
                if i % 10 == 0  # Show marker every 10 years
            }
            # Ensure min and max are included in the marks
            marks_feature[slider_values[feature]['min']] = {'label': str(slider_values[feature]['min'])}
            marks_feature[slider_values[feature]['max']] = {'label': str(slider_values[feature]['max'])}
        elif feature in reverse_mappings:
            marks_feature = {
                i: {'label': reverse_mappings[feature][i], 'style': {'id': 'slider-mark-label-l'}}
                for i in range(slider_values[feature]['min'], slider_values[feature]['max'] + 1)
            }
        else:
            marks_feature = {
                i: {'label': str(i), 'style': {'className': 'slider-mark-label'}}
                for i in range(slider_values[feature]['min'], slider_values[feature]['max'] + 1)
            }

        slider_divs.append(
            html.Div([
                html.H3(feature, className='headings'),
                dcc.Slider(
                    id={'type': 'dynamic-slider', 'index': feature.replace(" ", "")},
                    min=slider_values[feature]['min'],
                    max=slider_values[feature]['max'],
                    value=slider_values[feature]['mean'],
                    marks=marks_feature,
                    step=1
                ),
                html.Div(id=f'sliderFor{feature.replace(" ", "")}-output')
            ])
        )

    print("Slider Divs:", slider_divs)  # Debugging print
    accuracy_display = html.P(f"Accuracy of {current_model_name} Model: {accuracy_val:.3%}")

    return fig, roc_fig, cm_fig, html.Div(slider_divs, className='slider-container'), accuracy_display

def get_model_data(button_id):
    if not button_id.startswith("button-"):
        button_id = f"button-{button_id}"  # Ensuring format
    # Define a dictionary mapping button IDs to model functions and features
    model_funcs = {
        "button-1": (ml.RandomForestClassificationForSatisfaction, feature_mapping["RandomForest"]),
        "button-2": (ml.XGBoostClassificationForSatisfaction, feature_mapping["XGBoost"]),
        "button-3": (ml.AdaBoostClassificationForSatisfaction, feature_mapping["AdaBoost"]),
        "button-4": (ml.ANNClassificationForSatisfaction, feature_mapping["ANN"]),
        "button-5": (ml.DecisionTreeClassificationForSatisfaction, feature_mapping["DecisionTree"])
    }
    model_func, selected_features = model_funcs.get(button_id, (None, None))
    print("model-func")
    print(model_func)
    print("selected features")
    print(selected_features)

    if model_func is None or selected_features is None:
        print(f"Invalid button_id: {button_id}")
        return None, None, None, None, None

    try:
        accuracy_val, roc_auc, model, X_test, y_test, y_pred, _ = model_func(DataFrame)
        if button_id == "button-4":  # Checking if the selected model is ANN
            roc_fig = dv.plot_roc_curve2(model, X_test, y_test)  # Use plot_roc_curve2 for ANN

        else:
            roc_fig = dv.plot_roc_curve(model, X_test, y_test)  # Using the modified function above
        cm_fig = dv.generate_confusion_matrix(y_test, y_pred, labels=['Not Satisfied', 'Satisfied'])
    except Exception as e:
        print(f"Error executing model function for button_id {button_id}: {e}")
        return None, None, None, None, None

    print(f"Button ID: {button_id}, Model Function: {model_func.__name__}, Selected Features: {selected_features}")
    return selected_features, roc_fig, cm_fig, accuracy_val, model



@app.callback(
    Output('model-id-storage', 'children'),
    [Input('button-1', 'n_clicks'),
     Input('button-2', 'n_clicks'),
     Input('button-3', 'n_clicks'),
     Input('button-4', 'n_clicks'),
     Input('button-5', 'n_clicks')],
    prevent_initial_call=True
)

# Storing both the model id and the features in a JSON string to unpack later
def update_model_id(btn1, btn2, btn3, btn4, btn5):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_features, _, _, _, _ = get_model_data(button_id)

    if selected_features is not None:
        selected_features_list = selected_features.tolist()
        print(f"Selected features list for {button_id}: {selected_features_list}")
    else:
        selected_features_list = []
        print(f"No features selected for {button_id}")

    return json.dumps({'model_id': button_id, 'features': selected_features_list})



def calculate_feature_means(train_df):
    feature_means = {}
    for column in train_df.columns:
        feature_means[column] = train_df[column].mean()
    return feature_means

feature_means = calculate_feature_means(DataFrame)




def UpdatePrediction(model, important_features, feature_values):
    features = [
        'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]

    inputX = pd.DataFrame([feature_values], columns=important_features)
    print("Features included in inputX:", inputX.columns.tolist())

    # Validate and convert data types
    for column in inputX.columns:
        if inputX[column].dtype == 'object':
            try:
                inputX[column] = pd.to_numeric(inputX[column])
            except ValueError:
                inputX[column] = np.nan  # Replace non-convertible entries with NaN

    # Fill missing or invalid features with their mean values
    if hasattr(model, 'feature_names_in_'):  # Check if model has this attribute (Scikit-learn models)
        expected_features = model.feature_names_in_
    else:  # If not, assume it's a Keras model and use important_features
        expected_features = features

    print("Expected featuresss:", expected_features)

    for feature in expected_features:
        if feature not in inputX.columns or pd.isna(inputX[feature].iloc[0]):
            inputX[feature] = feature_means[feature]

    inputX = inputX[list(expected_features)]

    if type(model).__name__ == 'Sequential' and 'keras.src.models.sequential' in str(type(model)):
        # inputX = inputX[features]
        print("HIIII")
        model_dir = os.path.join(os.path.dirname(__file__), '.', 'model')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        inputX = scaler.transform(inputX)
        print(inputX)
        try:
            prediction = model.predict(inputX)
            print("predictionnnnnn", prediction)
            PredictedNumber = int(prediction[0][0] > 0.5)
            return PredictedNumber
        except Exception as e:
            print("Error during prediction:", e)
            return None
    else:

        try:
        # Make the prediction
            PredictedNumber = model.predict(inputX)[0]
            return PredictedNumber
        except Exception as e:
            print("Error during prediction:", e)
            print("Expected features:", model.feature_names_in_)
            print("Provided features:", inputX.columns.tolist())
            return None



@app.callback(
    Output('content-box-2', 'children'),
    [Input('model-id-storage', 'children')] +
    [Input({'type': 'dynamic-slider', 'index': ALL}, 'value')]
)

def update_prediction(stored_data, *slider_values):
    # Default initialization when the dashboard loads
    if not stored_data or stored_data == 'null':
        # Initialize with default model data, for example, button-2 (XGBoost)
        model_id = 'button-2'
        selected_features, _, _, _, model = get_model_data(model_id)
    else:
        try:
            data = json.loads(stored_data)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return "Error decoding data."

        model_id = data['model_id']
        selected_features = data['features']
        _, _, _, _, model = get_model_data(model_id)

        if not model_id or not selected_features:
            return "Error: Model data retrieval failed."

    if slider_values:
        feature_values = dict(zip(selected_features, slider_values[0]))
    else:
        # Using default mean values for sliders on initial load
        feature_values = {feature: feature_means[feature] for feature in selected_features}

    # Print feature values for debugging purposes
    print(f"Feature values: {feature_values}")


    prediction_result = UpdatePrediction(model, selected_features, feature_values)

    if prediction_result == 1:
        prediction_str = f"The Passenger is Satisfied"
    else:
        prediction_str = f"The Passenger is either Neutral or Dissatisfied"

    return prediction_str



# Starting the Server

if __name__ == "__main__":
    app.run_server(port=8050, host='0.0.0.0')