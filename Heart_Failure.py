# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def main():
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>Health Failure </h1>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()


# Read,Explore, and Preprocess the dataset:

dataset = pd.read_csv(r"C:\Users\User\Desktop\Heart\Heart_Failure.py")
heart = dataset

# Drop rows with missing values
heart.dropna(inplace=True)

# Check for missing values in each row
missing_rows = heart.isnull().any(axis=1)

# Print the rows with missing values
print(heart[missing_rows])


# Visualizations:

# Graph 1
# Set up Streamlit app
st.title('Heart Disease Proportion')

# Calculate the heart disease counts
heart_disease_counts = heart['HeartDisease'].value_counts()

# Define the colors for the pie chart
colors = ['skyblue', 'red']

# Create the animated pie chart using Plotly
fig_plotly = go.Figure(go.Pie(labels=[
                       'No Heart Disease', 'Heart Disease'], values=heart_disease_counts, hole=0.5))
fig_plotly.update_traces(hoverinfo='label+percent', textinfo='value+percent',
                         textfont=dict(size=16), marker=dict(colors=colors))

# Display the plot using Streamlit
st.plotly_chart(fig_plotly)

# Graph #2
# Set up Streamlit app
st.title('Exercise-Induced Angina')

# Group the data and create a stacked bar plot
stacked_counts = heart.groupby(
    ['ExerciseAngina', 'HeartDisease']).size().unstack()

# Create the stacked bar plot using Matplotlib
fig, ax = plt.subplots()
stacked_counts.plot(kind='bar', stacked=True, color=[
                    'skyblue', 'salmon'], ax=ax)

# Customize the plot
plt.xlabel('Exercise-Induced Angina')
plt.ylabel('Count')
plt.legend(['Normal', 'Heart Disease'])

# Display the plot
st.pyplot(fig)

# Graph 3

# Set up Streamlit app
st.title('Resting Blood Pressure by Heart Disease')

# Create an interactive box plot using Plotly
fig = px.box(heart, x='HeartDisease', y='RestingBP', color='HeartDisease',
             labels={'HeartDisease': 'Heart Disease',
                     'RestingBP': 'Resting Blood Pressure'},
             color_discrete_sequence=['#2ECC40',
                                      '#FF4136']  # Change colors here
             )

# Display the interactive box plot using Streamlit
st.plotly_chart(fig)

# Graph 4

# Calculate chest pain type counts
chest_pain_counts = heart.groupby('ChestPainType')[
    'HeartDisease'].value_counts().unstack()

# Set up the plot
colors = ['brown', 'orange']
labels = chest_pain_counts.columns.tolist()

fig, ax = plt.subplots()
chest_pain_counts.plot(kind='bar', color=colors, ax=ax)
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(labels)

# Add value labels to each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Set up Streamlit app
st.title('Chest Pain Type')

# Display the plot
st.pyplot(fig)

# Provide analysis in a relevant context
st.write('1. Typical angina (brown)')
st.write('2. Atypical angina (orange)')
st.write('3. Non-anginal pain')
st.write('4. Asymptomatic')


# Graph 6
# Set up Streamlit app
st.title('Heart Disease Rate by Age')

# Calculate the heart disease rate by age
heart_disease_rate = heart.groupby('Age')['HeartDisease'].mean() * 100

# Create the line plot using Plotly
fig = px.line(heart_disease_rate, x=heart_disease_rate.index, y=heart_disease_rate,
              labels={'x': 'Age', 'y': 'Heart Disease Rate (%)'},
              )

# Display the line plot using Streamlit
st.plotly_chart(fig)

# Graph 7
st.title('Heart Disease Rate by Sex')

# Calculate the heart disease counts by sex
sex_counts = heart.groupby('Sex')['HeartDisease'].value_counts().unstack()

# Create an interactive bar chart using Plotly
fig = px.bar(sex_counts, x=sex_counts.index, y=[0, 1],
             color_discrete_sequence=['blue', 'red'],
             labels={'x': 'Sex', 'y': 'Count'},)

# Add interactivity with Streamlit
st.plotly_chart(fig)

# Graph 8
# Set up Streamlit app
st.title('Age Distribution for Different Chest Pain Types')

# Create histogram for age distribution by chest pain type
fig, ax = plt.subplots()
ax.hist([heart[heart['ChestPainType'] == 'TA']['Age'],
         heart[heart['ChestPainType'] == 'ATA']['Age'],
         heart[heart['ChestPainType'] == 'NAP']['Age'],
         heart[heart['ChestPainType'] == 'ASY']['Age']],
        color=['skyblue', 'salmon', 'lightgreen', 'purple'],
        label=['Typical Angina', 'Atypical Angina',
               'Non-Anginal Pain', 'Asymptomatic'],
        alpha=0.7)

# Customize the plot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

# Display the plot
st.pyplot(fig)


    #Graph 9 
# Set up Streamlit app
st.title('Distribution of Maximum Heart Rate')

# Create two separate data series for Normal and Heart Disease cases
normal_max_hr = heart[heart['HeartDisease'] == 0]['MaxHR']
heart_disease_max_hr = heart[heart['HeartDisease'] == 1]['MaxHR']

# Create a histogram using Streamlit
fig, ax = plt.subplots()
ax.hist([normal_max_hr, heart_disease_max_hr],
        color=['black', 'salmon'], label=['Normal', 'Heart Disease'], alpha=0.7)

# Set labels and title
ax.set_xlabel('Maximum Heart Rate Achieved')
ax.set_ylabel('Frequency')

# Add legend
ax.legend()

# Display the histogram using Streamlit
st.pyplot(fig)


    #Graph 10 
# Set up Streamlit app
st.title('Fasting Blood Sugar and Heart Disease')

# Group the data and create a bar plot
fbs_counts = heart.groupby('FastingBS')[
    'HeartDisease'].value_counts().unstack()
fbs_counts.plot(kind='bar', color=['silver', 'red'])

# Customize the plot
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Count')
plt.legend(['Normal', 'Heart Disease'])

# Display the plot
st.pyplot(plt)

st.write('fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise')


    #Graph 11
# Set up Streamlit app
st.title('ST Segment Slope')

# Group the data by ST_Slope and calculate the heart disease count
slope_counts = heart.groupby(
    'ST_Slope')['HeartDisease'].value_counts().unstack()

# Create a bar plot
fig, ax = plt.subplots()
slope_counts.plot(kind='bar', ax=ax)

# Customize the plot
plt.xlabel('ST Segment Slope')
plt.ylabel('Count')
plt.legend(['No Heart Disease', 'Heart Disease'])

# Display the plot
st.pyplot(fig)


    #Graoh 12
# Set up Streamlit app
st.title('Heart Disease Analysis: Oldpeak')

# Group the data by 'HeartDisease' and calculate the average Oldpeak
avg_oldpeak = heart.groupby('HeartDisease')['Oldpeak'].mean()

# Set the maximum number of categories to show on the plot
max_categories = len(avg_oldpeak)

# Get the top N categories based on 'HeartDisease'
top_categories = avg_oldpeak.nlargest(max_categories)

# Create a line plot
fig, ax = plt.subplots()
top_categories.plot(kind='line', marker='o', ax=ax)

# Customize the plot
plt.xlabel('Heart Disease')
plt.ylabel('Average Oldpeak')

# Display the plot
st.pyplot(fig)
