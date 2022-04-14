# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:07:27 2022

@author: Chaudhary Bhai
"""


import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Chaudhary Bhai/Desktop/Parkinson_project/trained_model.sav', 'rb'))
sc = pickle.load(open('C:/Users/Chaudhary Bhai/Desktop/Parkinson_project/scaler.pkl','rb'))

# creating a function for Prediction

def parkinsons_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    # standardize the data
    std_data = sc.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction == 0):
        return "The Person does not have Parkinsons Disease"
    else:
        return "The Person has Parkinsons Disease"
  
    
  
def main():
    
    
    # giving a title
    st.title('Parkinsons Prediction Web App')
    
    
    # getting the input data from the user
    
    
    MDVP_Fo = st.text_input('Average vocal fundamental frequency')
    MDVP_Fhi = st.text_input('Maximum vocal fundamental frequency')
    MDVP_Flo = st.text_input('Minimum vocal fundamental frequency')
    MDVP_Jitter = st.text_input('Jitter in %')
    MDVP_Jitter_abs = st.text_input('Jitter absolute value')
    MDVP_RAP = st.text_input('RAP value')
    MDVP_PPQ = st.text_input('PPQ value')
    Jitter_DDP = st.text_input('Jitter DDP Value')
    MDVP_Shimmer = st.text_input('Shimmer Value')
    MDVP_Shimmer_db = st.text_input('Shimmer Value in deciBel')
    
    st.caption("Jitter = measures of variation in fundamental frequency")
    st.caption("Shimmer = measures of variation in amplitude")
    
    
    
    # creating a button for Prediction
    
    if (st.button('Parkinsons Test Result')):
        diagnosis = parkinsons_prediction([MDVP_Fo,MDVP_Fhi,MDVP_Flo,MDVP_Jitter,MDVP_Jitter_abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_db])
        st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()

    
