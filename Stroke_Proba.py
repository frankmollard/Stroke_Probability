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

@st.cache(allow_output_mutation=True)
def loadAllModels(url):
    m1 = joblib.load(urllib.request.urlopen(url + "/" + "svm1.pkl"))
    m2 = joblib.load(urllib.request.urlopen(url + "/" + "svm2.pkl"))
    m3 = joblib.load(urllib.request.urlopen(url + "/" + "logit1.pkl"))
    m4 = joblib.load(urllib.request.urlopen(url + "/" + "logit2.pkl"))
    m5 = joblib.load(urllib.request.urlopen(url + "/" + "nbc1.pkl"))
    m6 = joblib.load(urllib.request.urlopen(url + "/" + "nbc2.pkl"))
    m7 = joblib.load(urllib.request.urlopen(url + "/" + "rf1.pkl"))
    m8 = joblib.load(urllib.request.urlopen(url + "/" + "rf2.pkl"))
        
    return m1, m2, m3, m4, m5, m6, m7, m8

svm1, svm2, logit1, logit2, nbc1, nbc2, rf1, rf2 = loadAllModels(URL)
# Notify the reader that the data was successfully loaded.
data_load_state.text("AI-Models Loaded")

st.sidebar.title("Patient Data")

age = st.sidebar.slider('Age', 0, 100, 81)  # min: 0h, max: 23h, default: 17h
bmi = st.sidebar.slider('BMI', 0, 100, 30) 
agl = st.sidebar.slider('Average Glucose Level', 0, 400, 90) 

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
    
data_load_state.text("Predicting...")

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

@st.cache
def predict(df, cv: list):
        
    psvm1 = svm1.predict_proba(df[cv])[0][1]
    psvm2 = svm2.predict_proba(df[cv])[0][1]

    pnbc1 = nbc1.predict_proba(df[cv])[0][1]
    pnbc2 = nbc2.predict_proba(df[cv])[0][1]

    prf1 = rf1.predict_proba(
        df[[i for i in df.columns if i not in ['work_type_Never_worked', 'work_type_children']]]
        )[0][1]
    prf2 = rf2.predict_proba(
        df[[i for i in df.columns if i not in ['work_type_Never_worked', 'work_type_children']]]
        )[0][1]

    plogit1 = logit1.predict_proba(df)[0][1]
    plogit2 = logit2.predict_proba(df)[0][1]

    p = (psvm1 + psvm2) / 2 * 0.5 \
        + (pnbc1 + pnbc2) / 2 * 0.25\
        + (prf1 + prf2) / 2 * 0.15\
        + (plogit1 + plogit2) / 2 * 0.1

    return p

pred = predict(data, contVars)
data_load_state.text("Prediction done")

st.metric(label="Probability of Stroke", value=str(round(pred*100, 1)) + " %", delta=None)

if bmi > 45 and age > 75:
        st.text(
        """
        Note: Information is unreliable.
        BMI > 45 and age > 75.
        """
        )
