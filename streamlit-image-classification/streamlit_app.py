import ast
import uuid
import json
import base64
import logging
import zipfile
from io import BytesIO
from datetime import datetime
from collections import Counter

import requests
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_echarts import st_echarts


def download_data_sample(playground_url, auth_token):
    try:
        # Set the path for eval API
        eval_url = playground_url + "/prod/eval"
        
        # Set the authorization based on query parameter 'token', 
        # it is obtainable once you logged in to the modelshare website
        headers = {
            "Content-Type": "application/json", 
            "authorizationToken": auth_token,
        }

        # Set the body indicating we want to get sample data from eval API
        data = {
            "exampledata": "TRUE"
        }
        data = json.dumps(data)

        # Send the request
        sample_images = requests.request("POST", eval_url, 
                                         headers=headers, data=data).json()

        # Parsing the base64 encoded images
        images = sample_images['exampledata'].split(",")

        # Prepare the data sample in zip
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i, image in enumerate(images):
                file_name = "image_sample{}.png".format(i)
                image_buffer = BytesIO()
                image_buffer.write(base64.b64decode(image))
                zip_file.writestr(file_name, image_buffer.getvalue())
        
        # Setup a download button
        btn = st.download_button(
            label="Download data sample",
            data=zip_buffer.getvalue(),
            file_name="data_sample.zip",
            mime="application/zip",
            help="Download in-built example file to demo the app"
        )
    except Exception as e:
        logging.error(e)

# Converting photos to html tags
def path_to_image_html(photo):
    return '<img src="'+ photo + '" width="60" >'

def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def display_result(images, labels, statuses, datetimes, uuids):
    status_label = {
        True: "Success",
        False: "Failed",
    }

    # Made it this way to uniquely set the css without border
    with st.container():
        with st.container():
            _, col2 = st.columns([7, 2])
            with col2:
                dl_button = st.empty()
   
    # Create dataframe
    data_frame = pd.DataFrame()
    data_frame = data_frame.assign(time=datetimes)
    data_frame = data_frame.assign(input=[x[0] for x in images])
    data_frame = data_frame.assign(filename=[x[1] for x in images])
    data_frame = data_frame.assign(status=[status_label[x] for x in statuses])
    data_frame = data_frame.assign(result=labels)
    
    st.write(
        data_frame.to_html(escape=False, formatters={'input': image_formatter}),
        unsafe_allow_html=True
    )

    data_frame = data_frame.drop(['input'], axis=1)
    data_frame = data_frame.assign(unique_id=uuids)

    # Prepare the data sample in csv
    csv_data = data_frame.to_csv(index=False).encode('utf-8')

    # Setup a download button
    dl_button.download_button(
        label="Download result",
        data=csv_data,
        file_name="export.csv",
        mime="text/csv"
    )

def display_pie_chart(sizes, labels):
    data = [{"value": sizes[i], "name": labels[i]} for i in range(len(sizes))]
    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "left": "center"},
        "series": [
            {
                "name": "Prediction Statistics",
                "type": "pie",
                "radius": ["20%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {"show": False, "position": "center"},
                "emphasis": {
                    "label": {"show": True, "fontSize": "40", "fontWeight": "bold"}
                },
                "labelLine": {"show": False},
                "data": data,
            }
        ],
    }
    st_echarts(
        options=options, height="500px",
    )
    
def display_bar_chart(freqs, labels):
    options = {
        "xAxis": {
            "type": "category",
            "data": labels,
        },
        "yAxis": {"type": "value"},
        "series": [{"data": freqs, "type": "bar"}],
    }
    st_echarts(options=options, height="500px")
    
def display_stats(labels):
    counter = Counter(labels)
    unique_labels = list(counter.keys())
    freqs = list(counter.values()) # frequency of each labels

    # Size or portion in pie chart
    sizes = [float(x) / sum(freqs) * 100 for x in freqs]

    # Display prediction details
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            display_pie_chart(sizes, unique_labels)

        with col2:
            display_bar_chart(freqs, unique_labels)

def predict(uploaded_file, uuid_str, playground_url, auth_token):
    # Prepare the uploaded image into base64 encoded string
    image = Image.open(uploaded_file)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue())
    data = json.dumps({
        "data": encoded_string.decode('utf-8'),
        "uuid": uuid_str,
        "version": "mostrecent"
    })

    # Set the path for prediction API
    pred_url = playground_url + "/prod/m"

    # Set the authorization based on query parameter 'token', 
    # it is obtainable once you logged in to the modelshare website
    headers = {
        "Content-Type": "application/json", 
        "authorizationToken": auth_token,
    }

    # Send the request
    prediction = requests.request("POST", pred_url, 
                                  headers=headers, data=data)

    print(prediction.text)
    # Parse the prediction
    label = ast.literal_eval(prediction.text)[0]
    return label

# Resize image and extract the image filename
def transform_image(uploaded_file):
    image = Image.open(uploaded_file)
    MAX_SIZE = (150, 150)
    image.thumbnail(MAX_SIZE)
    return (image, uploaded_file.name)

# Add custom CSS for streamlit component
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def main():
    # Set custom CSS for whole layout
    local_css("style.css")

    # Set the API url accordingly based on AIModelShare Playground API.
    playground_url = "https://hwuryj604h.execute-api.us-east-1.amazonaws.com"

    # Get the query parameter
    params = st.experimental_get_query_params()
    if "token" not in params:
        st.warning("Please insert the auth token as query parameter. " 
                   "e.g. https://share.streamlit.io/raudipra/"
                   "streamlit-image-classification/main?token=secret")
        auth_token = ""
    else:
        auth_token = params['token'][0]

    labels = []
    statuses = []
    images = []
    uuids = []
    datetimes = []

    st.header("Flower Image Classification")
    
    with st.expander("Show developer's guide"):
        st.markdown("#### Guide to build a streamlit app with modelshare's API.")
        st.markdown("What you'll need: \n"
                    "- auth_token: modelshare's user authorization "
                    "token. It can be retrieved after signing in to "
                    "www.modelshare.org \n"
                    "- playground_url: API endpoint url from any modelshare's"
                    " playground to do prediction. \n\n")
        
        st.write("Use the sample code below and pass the auth_token "
                    "as a query parameter 'token' on streamlit's URL, e.g. "
                    "https://share.streamlit.io/user/apps-name/main?token=secret.")

        st.write("Here is a sample code to run a prediction of an image"
                 " using modelshare's playground url")

        code = """
import ast
import json
import base64
import requests
from PIL import Image
import streamlit as st
from io import BytesIO

playground_url = "https://hwuryj604h.execute-api.us-east-1.amazonaws.com"
auth_token = st.experimental_get_query_params()['token'][0]
image = Image.open(image_file)
def predict(image, playground_url, auth_token):
    # Prepare the uploaded image into base64 encoded string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue())
    data = json.dumps({"data": encoded_string.decode('utf-8')})

    # Set the path for prediction API
    pred_url = playground_url + "/prod/m"

    # Set the authorization based on query parameter 'token', 
    # it is obtainable once you logged in to the modelshare website
    headers = {
        "Content-Type": "application/json", 
        "authorizationToken": auth_token,
    }

    # Send the request
    prediction = requests.request("POST", pred_url, 
                                  headers=headers, data=data)

    # Parse the prediction
    label = ast.literal_eval(prediction.text)[0]
    return label
label = predict(data, playground_url, auth_token)
        """
        st.code(code, "python")

    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_files = st.file_uploader(
                label="Choose any image and get the prediction",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )

            download_data_sample(playground_url, auth_token)

        with col2:
            metric_placeholder = st.empty()
            metric_placeholder.metric(label="Request count", value=len(statuses))
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Keep the resized image and filename for prediction display
                images.append(transform_image(uploaded_file))
                
                # Create identifier for this prediction
                uuid_str = str(uuid.uuid4())
                uuids.append(uuid_str)
                
                # Capture the timestamp of prediction
                now = datetime.now()
                date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                datetimes.append(date_time)

                # Classify the image
                label = predict(uploaded_file, uuid_str, 
                                playground_url, auth_token)
                
                # Insert the label into labels
                labels.append(label)
                
                # Insert the API call status into statuses
                statuses.append(True)
            except Exception as e:
                logging.error(e)

                # add label as None if necessary
                if len(labels) < len(images):
                    labels.append(None)
                statuses.append(False)

        metric_placeholder.metric(label="Request count", value=len(statuses))
        display_stats(labels)
        display_result(images, labels, statuses, datetimes, uuids)

if __name__ == "__main__":
    main()