import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# Loading datset
df=pd.read_csv("dataset/Iris.csv")

# Drop unwanted columns
df=df.drop(columns=['Id'])

# Removing missing value 
df=df.dropna()

st.set_page_config( page_title="Iris Classification ",layout='wide')
st.title("Iris Classification model")
st.text("we can predict the species of flower on base of sepal and petal length and breadth")

st.dataframe(df,height=200)

fig,ax=plt.subplots(1,2,figsize=(12,3))

sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=df,ax=ax[0])

sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species',data=df,ax=ax[1])
st.pyplot(fig)

col1,col2=st.columns(2)
with col1:
    sl=st.number_input("Enter Sepal Length in cm")
    sw=st.number_input("Enter Sepal Widht in cm")

with col2:
    pl=st.number_input("enter Petal Length in cm")
    pw=st.number_input("enter Pepal Width in cm")


# loading MODEL
import joblib
model = joblib.load("model/model.pkl")


if st.button("Predict"):
     data=np.array([[sl,sw,pl,pw]])
     predict=model.predict(data)
     li=['Setosa','Versicolor','Verginic']
     st.success(li[predict[0]])

