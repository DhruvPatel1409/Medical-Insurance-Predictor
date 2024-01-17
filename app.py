import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open('C:/Users/ADMIN/Desktop/python jupyter/PROJECTS/medical insurance pred/insurance_model.sav','rb'))

def insurance_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    return str((prediction))

def main():

    st.title('Medical Insurance Prediction App')

    Age = st.number_input('Enter your Age : ')
    Sex = st.number_input('Sex : ')
    bmi = st.number_input('BMI Value : ')
    children = st.number_input('Number of children : ')
    smoker = st.number_input('Is Smoker : ')
    region = st.number_input('Region : ')

    # code for prediction

    diagnosis = ' '

    if st.button('PREDICT'):
        diagnosis = insurance_prediction([Age , Sex , bmi , children , smoker , region])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
    