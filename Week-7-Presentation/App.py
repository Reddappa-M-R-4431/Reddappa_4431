# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:03:34 2023

@author: Reddymr2022
"""

import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

pickle_in = open("sk_model.pkl","rb")
lasso=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def Sales_of_Retail_Store(week_id,outlet,product_identifier,department_identifier,category_of_product,state,day,month,year):
    
    """Let's Predict the sales of retail store 
    This is using docstrings for specifications.
        
    """
   
    prediction=Lasso.predict([[week_id,outlet,product_identifier,department_identifier,category_of_product,state,day,month,year]])
    print(prediction)
    return prediction

def main():
    st.title("Sale of Retail Store")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sales of Retail Store App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    week_id = st.text_input("week_id","Type Here")
    outlet = st.text_input("outlet","Type Here")
    product_identifier = st.text_input("product_identifier","Type Here")
    department_identifier = st.text_input("department_identifier","Type Here")
    category_of_product = st.text_input("category_of_product","Type Here")
    state = st.text_input("state","Type Here")
    day = st.text_input("day","Type Here")
    month = st.text_input("month","Type Here")
    year = st.text_input("year","Type Here")  
    result=""
    if st.button("Predict"):
        result=Sales_of_Retail_Store([week_id,outlet,product_identifier,department_identifier,category_of_product,state,day,month,year])
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()