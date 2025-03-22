import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

output = {}
# Load the ML model and transformer
@st.cache_resource
# Load model and transformer
def load_model(type):
    if type == "as_group":
        # Load the saved model - Assignment group
        with open('masdar_ticket_model_as_group.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the CountVectorizer - Assignment group
        with open('masdar_ticket_transformer_as_group.pkl', 'rb') as f:
            count_vect = pickle.load(f)
    elif type == "category":
        # Load the saved model - Assignment group
        with open('masdar_ticket_model_category.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the CountVectorizer - Assignment group
        with open('masdar_ticket_transformer_category.pkl', 'rb') as f:
            count_vect = pickle.load(f)
    elif type == "sub_category":
        # Load the saved model - Assignment group
        with open('masdar_ticket_model_sub_category.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the CountVectorizer - Assignment group
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

# Streamlit app
def main():
    st.title("Issue Classification App")
    st.write("Provide an issue description to predict the category.")

    # Input Text Area
    issue_description = st.text_area("Enter the issue description:", "")

    # # Load Model
    # model, count_vect = load_model()

    if st.button("Predict"):
        if issue_description.strip():
            try:

                #PREDICT TASK TYPE
                
                # call api start
                issue_description = " ".join(issue_description.split())
                url = 'https://predicttasktype.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview&api-key=4c7b77971b00461db6acf2efb0781501'
                data = f'{{"messages":[{{"role":"system","content":[{{"type":"text","text":"You are an AI assistant that classifies the given description as \'Incident\' or \'Service Request\'. Response should only be \'Incident\' or \'Service Request\'."}}]}},{{"role":"user","content":[{{"type":"text","text":"{issue_description}"}}]}}],"temperature":0.7,"top_p":0.95,"max_tokens":800}}'
                response = requests.post(url, data=data,headers={"Content-Type": "application/json","api-key":"4c7b77971b00461db6acf2efb0781501"})
                print(response)
                task_type=response.json()['choices'][0]['message']['content']  
                print(response.text)
                
                # call REST api end


                #predict assignment group
                model_output = load_model(type="as_group")
                model = model_output[0]
                count_vect = model_output[1]
                as_predict = predict_classification_scores(model, count_vect, issue_description)

                if as_predict and as_predict[0][0][0] is not None:
                    output.update({"as_group_status": "success", "pred_assignment_group": as_predict[0][0][0], "pred_assignment_group_score": as_predict[0][0][1]})            
                
                #predict category
                model_output = load_model(type="category")
                model = model_output[0]
                count_vect = model_output[1]
                category_predict = predict_classification_scores(model, count_vect, issue_description)

                if category_predict and category_predict[0][0][0] is not None:
                    output.update({"category_status": "success", "pred_category": category_predict[0][0][0], "pred_category_score": category_predict[0][0][1]})
                
                #predict sub_category
                model_output = load_model(type="sub_category")
                model = model_output[0]
                count_vect = model_output[1]
                sub_category_predict = predict_classification_scores(model, count_vect, issue_description)

                if sub_category_predict and sub_category_predict[0][0][0] is not None:
                    output.update({"sub_category_status": "success", "pred_sub_category": sub_category_predict[0][0][0], "pred_sub_category_score": sub_category_predict[0][0][1]})
                
                st.success("Prediction Results:")
                # print(output)
                # print(output['pred_category'])
                st.write(f"**Task type:** {task_type}")
                st.write(f"**Category:** {output['pred_category']} (Confidence: {output['pred_category_score']:.2f})")
                st.write(f"**Sub Category:** {output['pred_sub_category']} (Confidence: {output['pred_sub_category_score']:.2f})")
                st.write(f"**Assignment group:** {output['pred_assignment_group']} (Confidence: {output['pred_assignment_group_score']:.2f})")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.warning("Please enter an issue description.")

if __name__ == "__main__":
    main()
