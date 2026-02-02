import streamlit as st
import requests

st.set_page_config(page_title="Retail Shelf Analysis", layout="centered")

st.title("🛒 Retail Shelf Analysis")

uploaded = st.file_uploader(
    "Upload shelf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        files = {"image": uploaded}
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files=files
        )

    if response.status_code == 200:
        st.success("Detection complete!")
        st.image(
            response.content,
            caption="Detected Products",
            use_column_width=True
        )
    else:
        st.error("Prediction failed. Please try again.")

