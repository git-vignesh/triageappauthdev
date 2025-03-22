import azure.functions as func
import logging
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import requests
import os

app = Flask(__name__)

# Load the valid API key from environment variables (or Azure Key Vault)
VALID_API_KEY = os.environ.get('API_KEY')  # Assuming you set the API key in the environment variable

# Function to check the API key
def check_api_key(api_key):
    if api_key != VALID_API_KEY:
        return False
    return True

# Load model and transformer
def load_model(type):
    # Same logic for loading the models
    if type == "as_group":
        with open('masdar_ticket_model_as_group.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('masdar_ticket_transformer_as_group.pkl', 'rb') as f:
            count_vect = pickle.load(f)
    elif type == "category":
        with open('masdar_ticket_model_category.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('masdar_ticket_transformer_category.pkl', 'rb') as f:
            count_vect = pickle.load(f)
    elif type == "sub_category":
        with open('masdar_ticket_model_sub_category.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('masdar_ticket_transformer_sub_category.pkl', 'rb') as f:
            count_vect = pickle.load(f)
    return [model, count_vect]

def get_top_k_predictions_with_probabilities(model, X_test, k):
    probs = model.predict_proba(X_test)
    best_n = np.argsort(probs, axis=1)[:, -k:]
    top_k_preds_with_probs = [
        [(model.classes_[index], prob_row[index]) for index in indices[::-1]]
        for indices, prob_row in zip(best_n, probs)
    ]
    return top_k_preds_with_probs

def predict_classification_scores(model, count_vect, text):
    # Preprocess the input text using the same CountVectorizer used during training
    text_vectorized = count_vect.transform([text])
    out = get_top_k_predictions_with_probabilities(model, text_vectorized, k=1)
    return out

@app.route('/')
def index():
    return "<center><h2>Simple web app</h2></center>"

@app.route('/get-group', methods=['POST'])
def get_group():
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Check for API Key in the headers
        api_key = request.headers.get('x-api-key')
        if not api_key or not check_api_key(api_key):
            return jsonify({"status": "Unauthorized", "message": "Invalid or missing API key"}), 401

        # Get the JSON data from the request
        data = request.get_json()
        if 'name' not in data:
            return jsonify({"status": "Some error occurred"}), 400

        issue_description = " ".join(data['name'].split())
        print(issue_description)

        # Call external API for task classification
        url = 'https://predicttasktype.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview&api-key=4c7b77971b00461db6acf2efb0781501'
        data = f'{{"messages":[{{"role":"system","content":[{{"type":"text","text":"You are an AI assistant that classifies the given description as \'Incident\' or \'Service Request\'. Response should only be \'Incident\' or \'Service Request\'."}}]}},{{"role":"user","content":[{{"type":"text","text":"{issue_description}"}}]}}],"temperature":0.7,"top_p":0.95,"max_tokens":800}}'
        response = requests.post(url, data=data, headers={"Content-Type": "application/json", "api-key": "4c7b77971b00461db6acf2efb0781501"})
        task_type = response.json()['choices'][0]['message']['content']
        output = {"task_type": task_type}

        # Predict assignment group
        model_output = load_model(type="as_group")
        model = model_output[0]
        count_vect = model_output[1]
        as_predict = predict_classification_scores(model, count_vect, issue_description)

        if as_predict and as_predict[0][0][0] is not None:
            output.update({"as_group_status": "success", "pred_assignment_group": as_predict[0][0][0], "pred_assignment_group_score": as_predict[0][0][1]})

        # Predict category
        model_output = load_model(type="category")
        model = model_output[0]
        count_vect = model_output[1]
        category_predict = predict_classification_scores(model, count_vect, issue_description)

        if category_predict and category_predict[0][0][0] is not None:
            output.update({"category_status": "success", "pred_category": category_predict[0][0][0], "pred_category_score": category_predict[0][0][1]})

        # Predict sub_category
        model_output = load_model(type="sub_category")
        model = model_output[0]
        count_vect = model_output[1]
        sub_category_predict = predict_classification_scores(model, count_vect, issue_description)

        if sub_category_predict and sub_category_predict[0][0][0] is not None:
            output.update({"category_status": "success", "pred_sub_category": sub_category_predict[0][0][0], "pred_sub_category_score": sub_category_predict[0][0][1]})

        return jsonify(output)

    except Exception as e:
        print(e)
        return jsonify({"status": "exception caught", "message": str(e)}), 500

if __name__ == "__main__":
    app.run()
