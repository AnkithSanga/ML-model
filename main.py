import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd
from collections import Counter
with open('lm1.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# with open('X_train_data.pkl', 'rb') as file:
#     xtd = pickle.load(file)
# print(xtd)
xtd=pd.read_pickle("X_train_data.pkl")
print(xtd)

xtr=pd.read_pickle("X_train_rfe1_data.pkl")
print(xtr)


# for i in xtd.keys():
#     xtd[i]=0

c=dict()

pre_proc=joblib.load('pre_processor.x')
import streamlit as st
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu('Bike Count Prediction System',

                           ['Count Prediction','df'],
                           icons=['activity','activity'],
                           default_index=0)

# Diabetes Prediction Page
if (selected == 'Count Prediction'):

    # page title
    st.title('Linear Regression')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
       dteday  = st.date_input('Enter Date:')

    with col2:
         season= st.selectbox('Enter Season:',[1,2,3,4]) #dropdowns

    with col3:
         xtd.yr= st.selectbox('Enter Year:',[0,1])

    with col1:
         mnth= st.selectbox('Enter Month:',[1,2,3,4,5,6,7,8,9,10,11,12])

    with col2:
         xtd.holiday= st.selectbox('Enter Holiday',[0,1]) #0 or 1

    with col3:
        weekday= st.selectbox('Enter Weekday:',[1,2,3,4,5,6,7])

    with col1:
        xtd.workingday= st.selectbox('Enter Workingday:',[0,1])

    with col2:
         weathersit= st.selectbox('Enter Weathersit:',[0,1])
    with col3:
         xtd.temp= st.text_input('Enter Temperature:')
    with col1:
        atemp= st.text_input('Enter Average Temperature:')

    with col2:
         xtd.hum= st.text_input('Enter Humidity:')
    with col3:
         xtd.windspeed= st.text_input('Enter Windspeed:')
    with col1:
        casual= st.text_input('Enter Casual days:')

    with col2:
         registered= st.text_input('Enter Registered days:')
    k="season_"+str(season)
    xtd[k]=1
    k = "mnth_" + str(mnth)
    xtd[k] = 1
    k = "weekday_" + str(weekday)
    xtd[k] = 1
    k = "weathersit_" + str(weathersit)
    xtd=pd.DataFrame(xtd)
    print("dataframe")
    print(xtd.shape)
    print(xtd.head())
    pre_xtd = pd.DataFrame(pre_proc.transform(xtd), columns=pre_proc.get_feature_names_out())
    print(xtd,pre_xtd,"XTD PRE_XTD")
    st.dataframe(xtd)
