import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Define the function to create the heatmap
@st.cache_data
def create_corr_heatmap(df):
    """Create a correlation heatmap"""
    # Select only numerical columns
    df_numeric = df.select_dtypes(include=[np.number])
    corr = df_numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
    return fig

# Define the function to create the histogram
@st.cache_data
def create_histogram(df, column):
    fig = px.histogram(df, column)
    return fig

# Define the function to create the boxplot
@st.cache_data
def create_boxplot(df, column):
    fig = px.box(df, y=column)
    return fig

@st.cache_data
def load_data():
    """Loads the hotel booking data"""
    df = pd.read_csv('hotel_bookings.csv')

    # Process date columns
    df['arrival_date_month'] = pd.to_datetime(df['arrival_date_month'], format='%B', errors='coerce').dt.month
    df['arrival_date_year'] = pd.to_datetime(df['arrival_date_year'], format='%Y', errors='coerce').dt.year
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

    # Combine 'arrival_date_year' and 'arrival_date_month' into one column
    df['date'] = df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')

    return df.copy()

def create_visualizations(df, model, model_type, X_train):
    """Creates visualizations for the Streamlit app"""
    df = df.copy()
    df = df.sort_values('date')
    df_grouped = df.groupby(['date', 'market_segment']).size().reset_index(name='bookings')
    fig1 = px.line(df_grouped, x='date', y='bookings', color='market_segment', title='Total Bookings per Month by Market Segment')

    df['cohort_year'] = df.groupby('customer_type')['arrival_date_year'].transform('min')
    df['cohort_index'] = df.groupby('customer_type').cumcount() + 1

    # Cancelation Ratio based on Month
    cancelation_ratio_month = df.groupby('arrival_date_month')['is_canceled'].mean().reset_index(name='Cancellation Ratio')
    fig2 = px.line(cancelation_ratio_month, x='arrival_date_month', y='Cancellation Ratio', title='Cancellation Ratio by Month')

    if model_type == 'KMeans':
        X_train['cluster'] = model.labels_
        fig3 = px.scatter(X_train, x='lead_time', y='adr', color='cluster', title='Customer Clusters')
    else:  # Cover both 'DecisionTree' and 'RandomForest'
        fig3 = px.scatter(df, x='lead_time', y='adr', color='is_canceled', title='Lead Time vs. Average Daily Rate')

    fig3.update_layout(
        xaxis_title="Lead Time",
        yaxis_title="Average Daily Rate (ADR)",
        legend_title="Cluster",
        autosize=False,
        width=1000,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=10
        )
    )

    return fig1, fig2, fig3



def train_model(df, model_type, n_clusters=None):
    """Trains the selected model"""
    X = df[['lead_time', 'adr']]
    y = df['is_canceled']

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_train

st.title('Hotel Booking Analysis')

st.sidebar.header('User Input Parameters')
model_type = st.sidebar.selectbox('Model Type', ['KMeans', 'DecisionTree', 'RandomForest'])

lead_time = st.sidebar.number_input('Lead Time', min_value=0, max_value=365, value=50)
adr = st.sidebar.number_input('Average Daily Rate', min_value=0, max_value=1000, value=100)

df = load_data()

if model_type == 'KMeans':
    n_clusters = st.sidebar.number_input('Number of clusters', 1, 10, 3)
    model, accuracy, X_train = train_model(df, model_type, n_clusters)
else:
    model, accuracy, X_train = train_model(df, model_type)

st.sidebar.write(f'The accuracy of the {model_type} model is {accuracy:.2f}')

fig1, fig2, fig3 = create_visualizations(df, model, model_type, X_train)

st.header('Trend over Time with Segmentation')
st.plotly_chart(fig1)

st.header('Cohort Analysis')
st.plotly_chart(fig2)

if model_type == 'KMeans':
    st.header('Cluster Analysis')
else:
    st.header('Decision Tree Prediction Analysis')
st.plotly_chart(fig3)

st.sidebar.header('Predict Outcome for New Booking')
if model_type == 'KMeans':
    predicted_cluster = model.predict(np.array([lead_time, adr]).reshape(1, -1))[0]
    st.sidebar.write(f'The predicted cluster for a new booking with lead time {lead_time} and average daily rate {adr} is {predicted_cluster}')
elif model_type == 'DecisionTree':
    prediction = model.predict(np.array([lead_time, adr]).reshape(1, -1))[0]
    st.sidebar.write(f'The predicted outcome for a new booking with lead time {lead_time} and average daily rate {adr} is {"Cancelled" if prediction else "Not Cancelled"}')
elif model_type == 'RandomForest':
    prediction = model.predict(np.array([lead_time, adr]).reshape(1, -1))[0]
    st.sidebar.write(f'The predicted outcome for a new booking with lead time {lead_time} and average daily rate {adr} is {"Cancelled" if prediction else "Not Cancelled"}')

# Create a dropdown to select the column
column_to_plot = st.sidebar.selectbox(
    'Select column to create histogram',
    df.columns)
st.header(f'Histogram for {column_to_plot}')
hist = create_histogram(df, column_to_plot)
st.plotly_chart(hist)

column_to_plot_boxplot = st.sidebar.selectbox(
    'Select column to create boxplot',
    df.columns)
st.header(f'Boxplot for {column_to_plot_boxplot}')
box = create_boxplot(df, column_to_plot_boxplot)
st.plotly_chart(box)

st.header('Correlation Heatmap')
st.pyplot(create_corr_heatmap(df))
