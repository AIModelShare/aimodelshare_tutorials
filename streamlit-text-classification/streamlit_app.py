import ast
import json
import uuid
import logging
from datetime import datetime
from collections import Counter

import requests
import numpy as np
import pandas as pd
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
        sample_tabulars = requests.request("POST", eval_url, 
                                           headers=headers, data=data).json()

        # Parsing the tabular data
        tabulars = json.loads(sample_tabulars['exampledata'])
        df = pd.DataFrame(np.array(tabulars['data']),
                          columns=tabulars['columns'])

        # Prepare the data sample in csv
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        # Setup a download button
        btn = st.download_button(
            label="Download data sample",
            data=csv_data,
            file_name="text_sample.csv",
            mime="text/csv",
            help="Download in-built example file to demo the app"
        )

    except Exception as e:
        logging.error(e)

def display_result(inputs, labels, statuses, datetimes, uuids):
    status_label = {
        True: "Success",
        False: "Failed",
    }
    
    # Made it this way to uniquely set the css
    with st.container():
        with st.container():
            _, col2 = st.columns([7, 2])
            with col2:
                dl_button = st.empty()
   
    data_frame = pd.DataFrame()
    data_frame = data_frame.assign(time=datetimes)
    data_frame = data_frame.assign(input=inputs)
    data_frame = data_frame.assign(status=[status_label[x] for x in statuses])
    data_frame = data_frame.assign(result=labels)
    st.table(data_frame)

    # Prepare the data sample in csv
    data_frame = data_frame.assign(unique_id=uuids)
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

def predict(text, uuid_str, playground_url, auth_token):
    # Prepare the uploaded csv into per row record in json
    data = {}

    data = json.dumps({
        "data": text,
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

    # Parse the prediction
    label = ast.literal_eval(prediction.text)[0]

    return label

# Add custom CSS for streamlit component
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def main():
    # Set custom CSS for whole layout
    local_css("style.css")

    # Set the API url accordingly based on AIModelShare Playground API.
    playground_url = "https://zf3g9o9owf.execute-api.us-east-1.amazonaws.com"

    # Get the query parameter
    params = st.experimental_get_query_params()
    if "token" not in params:
        st.warning("Please insert the auth token as query parameter. " 
                   "e.g. https://share.streamlit.io/raudipra/"
                   "streamlit-tabular-classification/main?token=secret")
        auth_token = ""
    else:
        auth_token = params['token'][0]

    labels = []
    statuses = []
    uuids = []
    datetimes = []
    inputs = []
    
    st.header("IMDB Movie Review Classification")

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

        st.write("Here is a sample code to run a prediction of a text record"
                 " using modelshare's playground url")

        code = """
import ast
import json
import requests
import streamlit as st

playground_url = "https://zf3g9o9owf.execute-api.us-east-1.amazonaws.com"
auth_token = st.experimental_get_query_params()['token'][0]
data = {
    'col1': ['val1'],
    'col2': ['val2'],
    'col3': ['val3'],
}
def predict(data, playground_url, auth_token):
    data = json.dumps({"data": data})

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
            input_type = st.radio(
                "Input type",
                ('Single', 'Batch')
            )
            
            if input_type == "Single":
                text_input = st.text_input('Enter your text here')
                single_predict = st.button("Predict")
            else:
                uploaded_file = st.file_uploader(
                    label="Choose one csv and get the prediction",
                    type=["csv"],
                    accept_multiple_files=False,
                )

                download_data_sample(playground_url, auth_token)

        with col2:
            metric_placeholder = st.empty()
            metric_placeholder.metric(label="Request count", value=0)
         
    if input_type == "Single":
        if single_predict:
            try:
                # Create identifier for this prediction
                uuid_str = str(uuid.uuid4())
                uuids.append(uuid_str)
                
                # Capture the timestamp of prediction
                now = datetime.now()
                date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                datetimes.append(date_time)

                # Classify the record
                inputs.append(text_input)
                label = predict(text_input, uuid_str, playground_url, auth_token)

                # Insert the label into labels
                labels.append(label)
                
                # Insert the API call status into statuses
                statuses.append(True)
            except Exception as e:
                logging.error(e)

                # add label as None if necessary
                if len(labels) < data_frame.shape[0]:
                    labels.append(None)
                statuses.append(False)

            metric_placeholder.metric(label="Request count", value=len(statuses))
            display_stats(labels)
            display_result(inputs, labels, statuses, datetimes, uuids)
    else:
        if uploaded_file:    
            data_frame = pd.read_csv(uploaded_file)
            for _, row in data_frame.iterrows():
                try:
                    # Create identifier for this prediction
                    uuid_str = str(uuid.uuid4())
                    uuids.append(uuid_str)
                    
                    # Capture the timestamp of prediction
                    now = datetime.now()
                    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                    datetimes.append(date_time)

                    # Classify the record
                    inputs.append(row['text'])
                    label = predict(row['text'], uuid_str, playground_url, auth_token)

                    # Insert the label into labels
                    labels.append(label)
                    
                    # Insert the API call status into statuses
                    statuses.append(True)
                except Exception as e:
                    logging.error(e)

                    # add label as None if necessary
                    if len(labels) < data_frame.shape[0]:
                        labels.append(None)
                    statuses.append(False)

            metric_placeholder.metric(label="Request count", value=len(statuses))
            display_stats(labels)
            display_result(inputs, labels, statuses, datetimes, uuids)

if __name__ == "__main__":
    main()