import streamlit as st
import pandas as pd
from joblib import load
from sentence_transformers import SentenceTransformer

Model/gradient_boosting_model.joblib

# Function to load the models, I have only used SVM and Gradient Boosting since they gave the best results
def load_models():
    try:
        svm_model = load('svm_multioutput_model.joblib')
        gradient_boosting_model = load('gradient_boosting_model.joblib')
        
        print("All models loaded successfully.")
        return svm_model, gradient_boosting_model
    except Exception as e:
        print(f"An error occurred while loading the models: {e}")
        return None, None


# Function to generate embeddings for an input word
def generate_embeddings(input_word):
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode([input_word], show_progress_bar=False)

# Function to make predictions
def make_predictions(models, input_features):
    predictions = {}
    feature_names = [str(i) for i in range(len(input_features))]
    input_features_df = pd.DataFrame([input_features], columns=feature_names)
    for name, model in models.items():
        pred = model.predict(input_features_df)
        predictions[name] = pred[0]  # Assuming single prediction
    return predictions

models = {}

# Load models
svm_model, gradient_boosting_model = load_models()
if not all([svm_model, gradient_boosting_model]):
    print("One or more models could not be loaded.")
else:
    models = {
    'SVM': svm_model,
    'Gradient Boosting': gradient_boosting_model
    }

# Streamlit app
st.title('Linguistic Dimensions Prediction')

# Input from user, # [0] in input features is to get the vector from the list
input_word = st.text_input("Enter a word or phrase:")

if st.button('Predict'):
    if input_word:
        input_features = generate_embeddings(input_word)
        predictions = make_predictions(models, input_features[0])  

        # Display predictions
        for model_name, pred in predictions.items():
            st.write(f"Predictions by {model_name}:")
            st.write(f"Valence: {pred[0]}, Arousal: {pred[1]}, Dominance: {pred[2]}, Concreteness: {pred[3]}")
    else:
        st.write("Please enter a word or phrase.")
    