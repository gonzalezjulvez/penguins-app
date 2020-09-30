import streamlit as st
import pandas as pd 
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
# Aplicacion para predicer el tipo de pinguino
Este proyecto es para predecir la especie de pinguino según las caracteristicas que aporte el usuario""")

st.sidebar.header('User input features')

uploaded_file = st.sidebar.file_uploader('Upload your input CSV:',type='csv')
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('female','male'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.0, 60.0,43.0)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)',13.0,22.0,17.0)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,200.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island':island,
                'sex':sex,
                'bill_length_mm':bill_length_mm,
                'bill_depth_mm':bill_depth_mm,
                'flipper_length_mm':flipper_length_mm,
                'body_mass_g':body_mass_g}
        features = pd.DataFrame(data,index=[0])
        return features
    input_df = user_input_features()


# Combine user input features with entire penguins datasets
url_data = 'https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv'
penguins_raw = pd.read_csv(url_data)
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins], axis=0)
# Transform string features to num
encode = ['island','sex']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix='col')
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:1]

# User Input Features
st.write(""" ### User input features""")
st.dataframe(df)

# Load Model 
clf = pickle.load(open('penguins_clf.pkl','rb'))
prediction = clf.predict(df)
predict_proba = clf.predict_proba(df)


st.write(""" ### Prediction""")
penguins_species = np.array(['Adeline','Chinstrap','Gentoo'])

st.write(penguins_species[prediction])
st.write(""" ### Prediction Probability""")
st.write(predict_proba)