import streamlit as st
import requests

st.title("Retail Shelf Analysis")

uploaded = st.file_uploader("Upload shelf image", type=["jpg", "png"])

if uploaded:
    files = {"image": uploaded}
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files=files
    )

    if response.status_code == 200:
        data = response.json()

        st.subheader("JSON Output")
        st.json(data)

        st.subheader("Detection Visualization")
        st.image(data["visualization_path"])
