# Streamlit Image Classification on AIModelShare Example
This repository contains a sample streamlit web application that uses [AIModelShare](https://github.com/AIModelShare/aimodelshare) Playground as the model endpoint. This sample webapp is currently live on https://share.streamlit.io/raudipra/streamlit-image-classification/main

## How to install
`pip3 install -r requirements.txt`.

## How to run it locally
`streamlit run streamlit_app.py`.

## How to use
- Get the authorization token by logging in to www.modelshare.org.
- Put in the token as the value of query parameter 'token' of link above or your local deployment. e.g. https://share.streamlit.io/raudipra/streamlit-image-classification/main?token=secret