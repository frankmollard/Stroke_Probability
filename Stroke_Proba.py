import streamlit as st

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import joblib

import pandas as pd
import numpy as np

import urllib.request

st.title('Stroke Prediction')
st.text(
        """
        This application uses various AI-algorithms
        to indicate the probability of a stroke. If a stroke is suspected, 
        a doctor must always be consulted. This is a medical emergency. 
        This application is for demonstration purposes only. 
        """
        )

URL="https://strokemodels.s3.eu-central-1.amazonaws.com"

data_load_state = st.text('Loading models...')

svm1 = joblib.load(urllib.request.urlopen(URL + "/" + "svm1.pkl"))
svm2 = joblib.load(urllib.request.urlopen(URL + "/" + "svm2.pkl"))
logit1 = joblib.load(urllib.request.urlopen(URL + "/" + "logit1.pkl"))
logit2 = joblib.load(urllib.request.urlopen(URL + "/" + "logit2.pkl"))
nbc1 = joblib.load(urllib.request.urlopen(URL + "/" + "nbc1.pkl"))
nbc2 = joblib.load(urllib.request.urlopen(URL + "/" + "nbc2.pkl"))
rf1 = joblib.load(urllib.request.urlopen(URL + "/" + "rf1.pkl"))
rf2 = joblib.load(urllib.request.urlopen(URL + "/" + "rf2.pkl"))

# Notify the reader that the data was successfully loaded.
data_load_state.text("AI-Models Loaded")

st.sidebar.title("Patient Data")

age = st.sidebar.slider('Age', 0, 100, 81)  # min: 0h, max: 23h, default: 17h
bmi = st.sidebar.slider('BMI', 0, 100, 30) 
agl = st.sidebar.slider('Average Glucose Level', 0, 300, 100) 

smoking = st.sidebar.selectbox(
    'Smoking Status', ["Never Smoked", "Formally Smoked", "Smoker", "Unknown"]
    )
if smoking == "Never Smoked":   
    smoking_status_formerly_smoked = 0
    smoking_status_smokes = 0
    smoking_status_never_smoked = 1
elif smoking == "formerly smoked":
    smoking_status_formerly_smoked = 1
    smoking_status_smokes = 0
    smoking_status_never_smoked = 0
elif smoking == "smokes":
    smoking_status_formerly_smoked = 0
    smoking_status_smokes = 1
    smoking_status_never_smoked = 0
else:
    smoking_status_formerly_smoked = 0
    smoking_status_smokes = 0
    smoking_status_never_smoked = 0    
    
heart = st.sidebar.selectbox(
    'Heart Disease', ["Yes", "No"]
    )    
if heart == "Yes":
    heart_disease = 1
else:
    heart_disease = 0
    
gender = st.sidebar.selectbox(
    'Gender', ["Male", "Female"]
    )    
if gender == "Male":
    gender_Male = 1
else:
    gender_Male = 0
    
work_type = st.sidebar.selectbox(
    'Work Type', ["Children", "Government", "Never worked", "Private", "Self-employed"]
    )    
if work_type == "Children":
    work_type_children = 1
    work_type_Self_employed	= 0
    work_type_Private = 0
    work_type_Never_worked = 0
elif work_type == "Never worked":
    work_type_children = 0
    work_type_Self_employed	= 0
    work_type_Private = 0
    work_type_Never_worked = 1
elif work_type == "Private":
    work_type_children = 0
    work_type_Self_employed	= 0
    work_type_Private = 1
    work_type_Never_worked = 0
elif work_type == "Self-employed":
    work_type_children = 0
    work_type_Self_employed	= 1
    work_type_Private = 0
    work_type_Never_worked = 0
else:
    work_type_children = 0
    work_type_Self_employed	= 0
    work_type_Private = 0
    work_type_Never_worked = 0
    
married = st.sidebar.selectbox(
    'Ever Married_Yes', ["Yes", "No"]
    )    
if gender == "Yes":
    ever_married_Yes = 1
else:
    ever_married_Yes = 0
    
residence_type = st.sidebar.selectbox(
    'Residence Type', ["Urban", "Rural"]
    )    
if residence_type == "Urban":
    Residence_type_Urban = 1
else:
    Residence_type_Urban = 0
    
hyTen = st.sidebar.selectbox(
    'Hypertension', ["Yes", "No"]
    )    
if hyTen == "Yes":
    hypertension = 1
else:
    hypertension = 0
    
#smoking = st.sidebar.multiselect('Which films do you want to show data for?', FILMS)

data = pd.DataFrame(
    data=[
        [age], [hypertension], [heart_disease], [agl], [bmi], 
        [gender_Male], [work_type_Never_worked], [work_type_Private], 
        [work_type_Self_employed], [work_type_children], [ever_married_Yes], 
        [Residence_type_Urban], [smoking_status_formerly_smoked], 
        [smoking_status_never_smoked], [smoking_status_smokes]
        ], 
    index=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Male', 'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'ever_married_Yes',
       'Residence_type_Urban', 'smoking_status_formerly smoked',
       'smoking_status_never smoked', 'smoking_status_smokes']
    ).T
contVars = ["age", "avg_glucose_level", "bmi"]

psvm1 = svm1.predict_proba(data[contVars])[0][1]
psvm2 = svm2.predict_proba(data[contVars])[0][1]

pnbc1 = nbc1.predict_proba(data[contVars])[0][1]
pnbc2 = nbc2.predict_proba(data[contVars])[0][1]

prf1 = rf1.predict_proba(
    data[[i for i in data.columns if i not in ['work_type_Never_worked', 'work_type_children']]]
    )[0][1]
prf2 = rf2.predict_proba(
    data[[i for i in data.columns if i not in ['work_type_Never_worked', 'work_type_children']]]
    )[0][1]

plogit1 = logit1.predict_proba(data)[0][1]
plogit2 = logit2.predict_proba(data)[0][1]

pred = (psvm1 + psvm2) / 2 * 0.5 \
    + (pnbc1 + pnbc2) / 2 * 0.25\
    + (prf1 + prf2) / 2 * 0.15\
    + (plogit1 + plogit2) / 2 * 0.1

st.metric(label="Probability of Stroke", value=str(round(pred*100, 2)) + " %", delta=None)
