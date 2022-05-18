import streamlit as st
import pickle

model = pickle.load(open('model_reg.pkl','rb'))

st.title("Sentiment Analysis Project")
st.markdown("Baseline model for sentiment prediction trained on 100.000 amazon reviews from the digital music category.")

st.subheader("Write something (max 200 char)")
deg = st.text_input('', 0,200)

st.subheader("Predicted Sentiment")
st.code(float(model.predict([[deg]])))