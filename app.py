import streamlit as st
import pandas as pd
import pickle
from keras.models import load_model

st.title("Passenger Survival Chance in the Titanic Journey")
pclass=st.slider("Enter the passenger class for the user",1,3)
sex=st.selectbox("Enter the passenger Gender",['male','female'])
sibsp=st.slider("Enter the passenger total number of sibling and spouse ",1,8)
parch=st.slider("Enter the passenger total number of parents and child ",0,6)
fare=st.number_input('Enter the fare of the passenger')
embarked=st.selectbox('Enter the passenger station from which they boarded ',['Southampton','Chebourq','Queenstown'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])
if st.button('Data'):
    st.write(data)

model=load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)

with open('Scaler.pkl','rb') as file:
    scale=pickle.load(file)

data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scale.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)

y=y[0][0]

def Chance(y):
    if y>0.5:
        return 'The Passenger will survive in this journey'
    else:
        return 'The Passenger will not survive in this journey'
if st.button('Predict Survival Chance'):
    st.write('Probability of Passenger Survival Chance is',y)
    st.write(Chance(y))