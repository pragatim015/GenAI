from tensorflow.keras.models import load_model
model=load_model("rnn1.keras")
import streamlit as st
st.title("Project Discussion")
pred=model.predict("hi this is vs code")
st.write(pred)