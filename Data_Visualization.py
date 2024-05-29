import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc, confusion_matrix




def plot_age_vs_satisfaction(df):

    color_discrete_map = {
        "satisfied": "#0091D5",
        "neutral or dissatisfied": "#EA6A47",
    }

    # Creating the plot
    fig = px.histogram(
        df,
        x="Age",
        color="satisfaction",
        category_orders={"Age": list(range(5, 80))},
        title="Age vs Passenger Satisfaction",
        labels={"Age": "Age", "count": "Count"},
        color_discrete_map=color_discrete_map
    )

    # Customizing layout
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Count",
        bargap=0.2,
        template="plotly_white"
    )

    return fig



def DrawFeatureImportanceGraph(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    FigImportanceFeature = go.Figure()
    FigImportanceFeature.add_trace(
        go.Bar(x=DataFrame.index, y=DataFrame["Importance"], marker_color='#0091D5'))
    FigImportanceFeature.update_layout(
        title_text='Feature Contribution in The Model',
        title_x=0.5,
        height=550,
        xaxis_title="Features",
        yaxis_title="Importance"
    )
    return FigImportanceFeature


def plot_heatmaps(dataset):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Satisfaction vs. Checkin service',
            'Satisfaction vs. Inflight service'
        ],
        horizontal_spacing=0.5  # Increase horizontal spacing
    )

    # Common y-axis labels
    satisfaction_labels = dataset['satisfaction'].unique()

    # First heatmap
    table1 = pd.crosstab(dataset['satisfaction'], dataset['Checkin service'])
    heatmap1 = go.Heatmap(
        z=table1.values.tolist(),
        x=table1.columns,
        y=satisfaction_labels,
        colorscale='Oranges',
        colorbar=dict(title='Checkin service', x=0.25)
    )
    fig.add_trace(heatmap1, row=1, col=1)

    # Second heatmap
    table2 = pd.crosstab(dataset['satisfaction'], dataset['Inflight service'])
    heatmap2 = go.Heatmap(
        z=table2.values.tolist(),
        x=table2.columns,
        y=satisfaction_labels,
        colorscale='Blues',
        colorbar=dict(title='Inflight service', x=1.0)
    )
    fig.add_trace(heatmap2, row=1, col=2)

    fig.update_layout(
        title_text='Satisfaction vs. Services Heatmaps',
        # height=600,
        # width=1400,
        showlegend=False
    )

    # fig.show()




def plot_service_satisfaction_heatmaps(df):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 6))

    features = ['Checkin service', 'Inflight service']
    colors = ['Oranges', 'Blues']

    for i, (feature, color) in enumerate(zip(features, colors)):
        table = pd.crosstab(df['satisfaction'], df[feature])
        sns.heatmap(table, cmap=color, ax=axarr[i])

    plt.tight_layout()



def plot_boxplot(dataset, x_column, y_column, palette="YlOrBr"):
    fig = go.Figure()

    # Create boxplot
    fig.add_trace(go.Box(
        x=dataset[x_column],
        y=dataset[y_column],
        marker_color='#0091D5'
    ))

    # Update layout
    fig.update_layout(
        title="Boxplot",
        xaxis=dict(title=x_column),
        yaxis=dict(title=y_column),
        height=500,
        width=650
    )

    # Show figure
    return fig


def plot_histogram_stacked(dataset, x_column, hue_column):
    colors = ['#F1F1F1', '#1C4E80', '#A5D8DD', '#EA6A47', '#0091D5']

    fig = go.Figure()

    unique_categories = dataset[hue_column].unique()
    for i, category in enumerate(unique_categories):
        subset = dataset[dataset[hue_column] == category]
        color = colors[i % len(colors)]
        fig.add_trace(go.Histogram(
            x=subset[x_column],
            histfunc="count",
            name=str(category),
            marker_color=color
        ))

    # Update layout
    fig.update_layout(
        title="Stacked Histogram",
        xaxis=dict(title=x_column),
        yaxis=dict(title="Count"),
        barmode="stack",
        height=500,
        width=650
    )

    # Showing figure
    return fig



def plot_satisfaction_delay(data):
    fig = px.scatter(data,
                     x='Departure Delay in Minutes',
                     y='Arrival Delay in Minutes',
                     color='satisfaction',
                     color_discrete_map={'satisfied': '#0091D5',
                                         'neutral or dissatisfied': '#E44242'})

    fig.update_layout(
                      title='Distribution of Departure Delay in Minutes over Arrival Delay in Minutes According to Satisfaction',
                      title_font={'size': 16},
                      height=500,
                      template="plotly_white",
                      showlegend=True)

    # Update axes to remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)

    return fig

def plot_satisfaction_distribution(dataset):

    satisfied = dataset[dataset['satisfaction'] == 'satisfied']['Flight Distance']
    neutral_or_not_sat = dataset[dataset['satisfaction'] == 'neutral or dissatisfied']['Flight Distance']

    fig = ff.create_distplot([satisfied, neutral_or_not_sat],
                             ['satisfied distance', 'neutral or not satisfied distance'],
                             show_hist=False,
                             show_rug=False)

    fig.update_layout(title='Distribution of Satisfied and Dissatisfied Distance',
                      title_font={'size': 16},
                      height=500,
                      template="plotly_white",
                      showlegend=True)

    # Update axes
    fig.update_xaxes(showgrid=False, title='Flight Distance')
    fig.update_yaxes(showgrid=True)

    return fig



def plot_comfort_and_legroom_satisfaction(dataset):

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Seat Comfort', 'Leg Room Service'))

    colors = {'satisfied': '#0091D5', 'neutral or not satisfied': '#EA6A47'}

    for satisfaction_type in dataset['satisfaction'].unique():
        filtered_data = dataset[dataset['satisfaction'] == satisfaction_type]
        counts = filtered_data['Seat comfort'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=counts.index, y=counts, name=satisfaction_type,
                   marker_color=colors.get(satisfaction_type, '#EA6A47')),
            row=1, col=1
        )

    for satisfaction_type in dataset['satisfaction'].unique():
        filtered_data = dataset[dataset['satisfaction'] == satisfaction_type]
        counts = filtered_data['Leg room service'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=counts.index, y=counts, name=satisfaction_type,
                   marker_color=colors.get(satisfaction_type, '#EA6A47')),
            row=1, col=2
        )

    fig.update_layout(
        title_text='Seat Comfort and Leg Room Service Satisfaction',
        title_font_size=16,
        barmode='group',
        height=500,
        width=1362
    )

    # Update x-axis and y-axis titles
    fig.update_xaxes(title_text='Seat Comfort', row=1, col=1)
    fig.update_xaxes(title_text='Leg Room Service', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=2)

    return fig



def plot_accuracy(epochs, acc, val_acc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=acc, mode='lines', name='Training Accuracy', line=dict(color='#EA6A47')))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Validation Accuracy', line=dict(color='#0091D5')))

    fig.update_layout(
        title='Training and Validation Accuracy of ANN',
        xaxis_title='Epochs',
        height= 450,
        yaxis_title='Accuracy'
    )

    return fig

def plot_loss(epochs, loss, val_loss):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', name='Training Loss', line=dict(color='#EA6A47')))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss', line=dict(color='#0091D5')))

    fig.update_layout(
        title='Training and Validation Loss of ANN',
        xaxis_title='Epochs',
        yaxis_title='Loss'
    )

    return fig





def plot_model_performance(model_stats):
    data = pd.DataFrame(model_stats)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=data['Model'], y=data['Time taken'], name='Time Taken (s)', marker_color='#0091D5'), 
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data['Model'], y=data['Accuracy'], name='Accuracy', mode='lines+markers',
                   line=dict(color='#1C4E80')), secondary_y=True,  
    )

    fig.add_trace(
        go.Scatter(x=data['Model'], y=data['ROC_AUC'], name='ROC AUC', mode='lines+markers',
                   line=dict(color='#EA6A47')),  # Using color #EA6A47
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Model")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Time Taken (s)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Accuracy / ROC AUC</b>", secondary_y=True)

    fig.update_layout(
        title='Model Comparison: Accuracy, Area under ROC Curve, and Time Taken',
        title_x=0.5,
        height=550,  # Increase the height of the graph
        # width= 1200,
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
    )

    return fig



def plot_roc_curve2(model, features_test, target_test, model_name='ANN Model'):
    proba = model.predict(features_test).ravel()  # Flatten the array

    fpr, tpr, thresholds = roc_curve(target_test, proba)

    roc_auc = auc(fpr, tpr)

    trace1 = go.Scatter(x=fpr, y=tpr, mode='lines',
                        line=dict(color='#EA6A47', width=3),
                        name=f'ROC curve (area = {roc_auc:.2f})')
    trace2 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                        line=dict(color='#0091D5', width=3, dash='dash'),
                        name='Chance (baseline)')

    layout = go.Layout(
        title='Receiver Operating Characteristic',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate', range=[0, 1.05]),
        showlegend=True,
        template='plotly_white',  # Add the 'white' template here
        annotations=[dict(
            x=0.95, y=0.05, xref='paper', yref='paper',
            text=model_name, showarrow=False,
            xanchor='right', yanchor='bottom'
        )]
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    return fig



def generate_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', normalize='true'):

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    annot_text = [[f"{cell:.2f}" for cell in row] for row in cm]

    # Defining a custom color scale
    custom_colorscale = [
        [0, '#EA6A47'],  # color for the lowest value
        [1, '#0091D5']   # color for the highest value
    ]

    # Creating a heatmap figure using Plotly
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels if labels else np.unique(y_pred),
        y=labels if labels else np.unique(y_true),
        annotation_text=annot_text,
        colorscale=custom_colorscale,
        showscale=True
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        template="plotly_white",
        xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels) if labels else {},
        yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels) if labels else {},
    )

    return fig



def plot_roc_curve(model, features_test, target_test, model_name='My Model'):

    proba = model.predict_proba(features_test)[:, 1]

    # Computing the ROC curve points
    fpr, tpr, thresholds = roc_curve(target_test, proba)

    roc_auc = auc(fpr, tpr)

    trace1 = go.Scatter(x=fpr, y=tpr, mode='lines',
                        line=dict(color='#EA6A47', width=3),
                        name=f'ROC curve (area = {roc_auc:.2f})')
    trace2 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                        line=dict(color='#0091D5', width=3, dash='dash'),
                        name='Chance (baseline)')

    layout = go.Layout(
        title='Receiver Operating Characteristic',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate', range=[0, 1.05]),
        showlegend=True,
        annotations=[dict(
            x=0.95, y=0.05, xref='paper', yref='paper',
            text=model_name, showarrow=False,
            xanchor='right', yanchor='bottom'
        )]
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    return fig




def plot_cleanliness_distribution(df):
    cleanliness_distribution = df['Cleanliness'].value_counts(normalize=True).reset_index()
    cleanliness_distribution.columns = ['Cleanliness', 'Distribution']

    # Creating a pie chart for the distribution of cleanliness ratings
    fig = px.pie(cleanliness_distribution, names='Cleanliness', values='Distribution',
                 title='Distribution of Cleanliness Ratings',
                 color_discrete_sequence=['#EA6A47', '#0091D5', '#7E909A', '#1C4E80', '#A5D8DD'],
                 labels={'Cleanliness': 'Rating', 'Distribution': 'Distribution (%)'})
    fig.update_layout(template='plotly_white')
    return fig



def plot_cleanliness_satisfaction_rate(df):
    cleanliness_satisfaction_rate = df.groupby('Cleanliness')['satisfaction'].mean().reset_index()
    cleanliness_satisfaction_rate.columns = ['Cleanliness', 'Satisfaction Rate']

    # Creating a bar chart for satisfaction rate by cleanliness rating
    fig = px.bar(cleanliness_satisfaction_rate, x='Cleanliness', y='Satisfaction Rate',
                 text='Satisfaction Rate',  # Display real text on the bars
                 labels={'Cleanliness': 'Rating', 'Satisfaction Rate': 'Satisfaction Rate (%)'},
                 title='Satisfaction Rate by Cleanliness Rating',
                 color_discrete_sequence=['#EA6A47', '#0091D5'])
    fig.update_traces(texttemplate='%{text:.2%}', textposition='inside', insidetextanchor='start')
    fig.update_layout(
        margin=dict(t=80, b=30, l=70, r=40),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16)
    )
    return fig



def plot_gate_class_distribution(df):
    gate_class_distribution = df.groupby(['Cleanliness', 'Class']).size().reset_index(name='Count')

    class_order = ['Business', 'Eco', 'Eco Plus']

    # Creating a bar chart for the distribution of Cleanliness by Class
    fig = px.bar(gate_class_distribution, x='Cleanliness', y='Count', color='Class',
                 title='Distribution of Cleanliness by Class',
                 labels={'Cleanliness': 'Cleanliness', 'Count': 'Count'},
                 color_discrete_sequence=['#EA6A47', '#0091D5', '#1C4E80'],
                 category_orders={'Class': class_order})

    fig.update_layout(
        margin=dict(t=80, b=30, l=70, r=40),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16)
    )
    return fig
