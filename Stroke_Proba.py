import streamlit as st

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

import joblib

import pandas as pd
import numpy as np

import urllib.request

import boto3
from botocore.config import Config
from botocore import UNSIGNED
import io

st.title('Stroke Prediction')
st.text(
        """
        This application uses various AI-algorithms
        to indicate the risk of a stroke. If a stroke is suspected, 
        a doctor must always be consulted. This is a medical emergency. 
        This application is for demonstration purposes only. 
        """
        )

URL="https://strokemodels.s3.eu-central-1.amazonaws.com"

data_load_state = st.text('Loading models...')

#Load Sklearn models
@st.cache(allow_output_mutation=True)
def loadAllModels(url):
    models=[]
    for c in ["svm1", "svm2", "logit1", "logit2", "nbc1", "nbc2", "rf1", "rf2"]:
        models.append(
            joblib.load(
                urllib.request.urlopen(url + "/" + "{}.pkl".format(c))
                )
            )

        
    return models[0], models[1], models[2], models[3], models[4], models[5], models[6], models[7]

svm1, svm2, logit1, logit2, nbc1, nbc2, rf1, rf2 = loadAllModels(URL)

#Load CatBoost


@st.cache(allow_output_mutation=True)
def loadCatBoost():
    
    s3 = boto3.resource(
        service_name='s3',
        region_name='eu-central-1',
        config=Config(signature_version=UNSIGNED)
    )
    bucket = s3.Bucket('strokemodels')

    models=[]

    for c in ["cb1", "cb2"]:
        
        obj = bucket.Object("%s" % (c))
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)# downoad to memory
        
        CB = CatBoostClassifier()
        
        models.append(CB.load_model(blob=file_stream.getvalue()))
        
    return models[0], models[1]
    
cb1, cb2 = loadCatBoost()


# Notify the reader that the data was successfully loaded.
data_load_state.text("AI-Models Loaded")

###############SIDEBAR START################

st.sidebar.title("Patient Data")

age = st.sidebar.slider('Age', 0, 100, 81)  # min: 0h, max: 23h, default: 17h
bmi = st.sidebar.slider('BMI', 0, 100, 30) 
agl = st.sidebar.slider('Average Glucose Level', 0, 400, 100) 

smoking = st.sidebar.selectbox(
    'Smoking Status', ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"]
    )
if smoking == "Never Smoked":   
    smoking_status_formerly_smoked = 0
    smoking_status_smokes = 0
    smoking_status_never_smoked = 1
    smoking_status = "never nmoked"
elif smoking == "Formerly Smoked":
    smoking_status_formerly_smoked = 1
    smoking_status_smokes = 0
    smoking_status_never_smoked = 0
    smoking_status = "formerly smoked"
elif smoking == "Smokes":
    smoking_status_formerly_smoked = 0
    smoking_status_smokes = 1
    smoking_status_never_smoked = 0
    smoking_status = "smokes"
else:
    smoking_status_formerly_smoked = 0
    smoking_status_smokes = 0
    smoking_status_never_smoked = 0   
    smoking_status = "Unknown"
    
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
    workType = "children"
elif work_type == "Never worked":
    work_type_children = 0
    work_type_Self_employed	= 0
    work_type_Private = 0
    work_type_Never_worked = 1
    workType = "Never_worked"
elif work_type == "Private":
    work_type_children = 0
    work_type_Self_employed	= 0
    work_type_Private = 1
    work_type_Never_worked = 0
    workType = "Private"
elif work_type == "Self-employed":
    work_type_children = 0
    work_type_Self_employed	= 1
    work_type_Private = 0
    work_type_Never_worked = 0
    workType = "Self-employed"
else:
    work_type_children = 0
    work_type_Self_employed	= 0
    work_type_Private = 0
    work_type_Never_worked = 0
    workType = "Govt_job"
    
married = st.sidebar.selectbox(
    'Ever Married', ["Yes", "No"]
    )    
if married == "Yes":
    ever_married_Yes = 1
    ever_married = True
else:
    ever_married_Yes = 0
    ever_married = False
    
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

st.sidebar.text(" ")
###############SIDEBAR END##################
    
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

dataC = pd.DataFrame(
    data=[
        [gender], [age], [hypertension], [heart_disease], [ever_married], 
        [workType], [residence_type], [agl], 
        [bmi], [smoking_status]
        ], 
    index=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level',
       'bmi', 'smoking_status']
    ).T
###TEST#############
#data = pd.DataFrame(
#    data=[
#        [38], [1], [1], [100], [30], 
#        [1], [0], [0], 
#        [0], [1], [1], 
#        [1], [1], 
#        [0], [0]
#        ], 
#    index=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
#       'gender_Male', 'work_type_Never_worked', 'work_type_Private',
#       'work_type_Self-employed', 'work_type_children', 'ever_married_Yes',
#       'Residence_type_Urban', 'smoking_status_formerly smoked',
#       'smoking_status_never smoked', 'smoking_status_smokes']
#    ).T

#dataC = pd.DataFrame(
#    data=[
#        [1], [38], [1], [1], [True], 
#        ["children"], ["Rural"], [100], 
#        [30], ["formerly smoked"]
#        ], 
#    index=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
#       'work_type', 'Residence_type', 'avg_glucose_level',
#       'bmi', 'smoking_status']
#    ).T
###TEST#############

contVars = ["age", "avg_glucose_level", "bmi"]

@st.cache
def predict(df, dfc, cv: list):
        
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
    
    pcb1 = cb1.predict(dfc, prediction_type='Probability')[:, 1]
    pcb2 = cb2.predict(dfc, prediction_type='Probability')[:, 1]

    p = (psvm1 * 0.82 + prf1 * 0.04 + plogit1 * 0.02 + pcb1[0] * 0.06 + pnbc1 * 0.06) / 2 + \
        (psvm2 * 0.13 + prf2 * 0.02 + plogit2 * 0.28 + pcb2[0] * 0.22 + pnbc2 * 0.35) / 2

    return p

pred = predict(data, dataC, contVars)

data_load_state.text("Prediction done")

#########Save User-data by caching############
@st.cache(allow_output_mutation=True)
def userData():
    return []

if len(userData()) == 0:
    userData().extend([0, round(pred*100, 1)])
    delta = +0
else:
    userData().pop(0)
    userData().append(round(pred*100, 1))
    delta = userData()[1] - userData()[0]
        
st.metric(
    label="Risk of Stroke", 
    value=str(round(pred*100, 1)) + " %", 
    delta=str(round(delta, 2)) + " percentage points", 
    help="""
    This is the indication for the risk of stroke, given the patient data.
    The change in percentage points compared to your previous indication is displayed smaller below.
    """,
    delta_color ="inverse"
)

#######Additional Information##################

if bmi > 45 and age > 75:
    st.text(
    """
    Note: Information is unreliable.
    BMI > 45 and age > 75.
    """
    )
elif bmi < 18.5:
    st.text("Shortweight")
elif bmi >= 18.5 and bmi < 25:
    st.text("Normal Weight")
elif bmi >= 25 and bmi < 30:
    st.text("Overweight")
elif bmi >= 30 and bmi < 35:
    st.text("Moderate Obesity")
elif bmi >= 35 and bmi < 40:
    st.text("Strong Obesity")
elif bmi >= 40:
    st.text("Extreme Obesity")
        
#####Data Visualization#########
viz = dataC
viz.rename(
    columns={
        "age": "Age",
        "bmi": "BMI",
        "avg_glucose_level": "Average Glucose Level",
        "smoking_status": "Smoking Status",
        "heart_disease": "Heart Disease",
        "gender": "Gender",
        "work_type": "Work Type",
         "ever_married": "Ever Married",
        "Residence_type": "Residence Type",
        "hypertension": "Hypertension",    
    }, 
    index={0: 'Data entered'}, 
    inplace=True
)
viz["Hypertension"] = hyTen
viz["Heart Disease"] = heart
viz["Ever Married"] = married
viz["Work Type"] = work_type
viz["Smoking Status"] = smoking

viz = viz.iloc[:, [1,8,7,9,3,0,5,4,6,2]]

st.table(data=viz.T)
