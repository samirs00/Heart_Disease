
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import logging

import os
print(os.getcwd())
import pickle
from PIL import Image

# Streamlit Local System
# heart_disease_model_lr = pickle.load(open(location+'logistic_model_pkl.pkl', 'rb'))
# heart_disease_model_dt = pickle.load(open(location+'DecisionTreeClassifier.pkl', 'rb'))
# heart_disease_model_xgb = pickle.load(open(location+'XGBoost.pkl', 'rb'))

# Streamlit Cloud
# Access the model files
# model_dir='/models'
# heart_disease_model_lr_path = os.path.join(model_dir, "logistic_model_pkl.pkl")
# heart_disease_model_dt_path = os.path.join(model_dir, "DecisionTreeClassifier.pkl")
# heart_disease_model_xgb_path = os.path.join(model_dir, "XGBoost.pkl")

# heart_disease_model_lr = pickle.load(open(heart_disease_model_lr_path, 'rb'))
# heart_disease_model_dt = pickle.load(open(heart_disease_model_dt_path, 'rb'))
# heart_disease_model_xgb = pickle.load(open(heart_disease_model_xgb_path, 'rb'))

print('path',os.getcwd())
# Reading model files
heart_disease_model_lr = pickle.load(open('/mount/src/heart_disease/models/logistic_model_pkl.pkl', 'rb'))
heart_disease_model_dt = pickle.load(open('/mount/src/heart_disease/models/DecisionTreeClassifier.pkl', 'rb'))
heart_disease_model_xgb = pickle.load(open('/mount/src/heart_disease/models/XGBoost.pkl', 'rb'))


image = Image.open('/mount/src/heart_disease/Heart_Disease.jpeg')

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Heart Disease Prediction Module',
                          
                          ['About Project',
                           'Project Contributors',
                           'Exploratory Data Analysis',
                           'Heart Disease Prediction'
                            ],
                          icons=['activity','activity','heart'],
                          default_index=0)
    
    
 
if (selected == 'About Project'):
    # page title
    st.title('Heart Disease Prediction using Machine Learning')
    st.markdown('Aim of the project is to build a machine learning model capable of predicting wheather or not someone has heart disease based on their medical attributes.')
    st.image(image, caption='')
    

if (selected == 'Project Contributors'):
    st.title("1. Samir Rathod")
    st.title("2. Sujit Date")
    st.title("3. Arpita Bhujade")
    st.title("4. Shubham Rathod")
    # title = st.text_input('Project Contributors')

if (selected == 'Exploratory Data Analysis'):
    st.markdown("https://github.com/samirs00/Heart_Disease/blob/main/heart%20disease%20analysis.ipynb")

    # page title
# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using Machine Learning')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age',value="37")
        age=int(age)
    with col2:
        sex_value = st.selectbox('Select Sex', ['Male', 'Female'])
        #sex_value = st.text_input('Sex',value="Male")
        if sex_value=='Male':
           sex=1
        else:
            sex=0
    with col3:
        cp = st.text_input('Chest Pain types',value="2")
        cp=int(cp)
    with col1:
        trestbps = st.text_input('Resting Blood Pressure',value="130")
        trestbps= int(trestbps)
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl',value="250")
        chol=int(chol)
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl',value="0")
        fbs=int(fbs)
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results',value="1")
        restecg=int(restecg)
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved',value="187")
        thalach=int(thalach)
    with col3:
        exang = st.text_input('Exercise Induced Angina',value="0")
        exang =int(exang)
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise',value="3.5")
        oldpeak=float(oldpeak)
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment',value="0")
        slope=int(slope)
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy',value="0")
        ca =int(ca)
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',value="2")
        thal =int(thal)
        
     
     
    # code for Prediction
    heart_diagnosis = ''


    input_data_array=np.asarray(tuple([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]))
    print('input_data_array type',type(input_data_array))
    input_data_reshaped = input_data_array.reshape(1,-1)

    if st.button('Heart Disease Test Result'):
        heart_prediction_lr = heart_disease_model_lr.predict(input_data_reshaped)                          
        heart_prediction_dt = heart_disease_model_dt.predict(input_data_reshaped)                          
        heart_prediction_xgb = heart_disease_model_xgb.predict(input_data_reshaped)                          
        

        #Model Ensemble
        import statistics
        ensemble_pred = statistics.mode([int(heart_prediction_lr), int(heart_prediction_dt), int(heart_prediction_xgb)])
        
        if (ensemble_pred == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    
# DOES NOT
# input_data=(67,1,0,160,286,0,0,108,1,1.5,1,3,2)

#HAS DISEASE
#input_data=(37,1,2,130,250,0,1,187,0,3.5,0,0,2)



