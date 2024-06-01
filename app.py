import logging

import numpy as np
import pandas as pd
import streamlit as st
from transformers import BertModel, BertTokenizer

import weaviate
from helpers.helpers import (aggregate_embeddings, cosine_similarity,
                             fetch_diagnosis_embeddings,
                             generate_symptom_embedding)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Initialize Weaviate client
client = weaviate.Client("http://weaviate:8080")

# Load possible symptoms from your dataset
symptoms_df = pd.read_csv("data/Symptom-severity.csv")
possible_symptoms = symptoms_df["Symptom"].tolist()

# Load main_df, precautions and descriptions data
main_df = pd.read_csv("data/dataset.csv")
precautions_df = pd.read_csv("data/symptom_precaution.csv")
descriptions_df = pd.read_csv("data/symptom_Description.csv")

# Function to fetch additional diagnosis data


def fetch_additional_data(diagnosis):
    description = descriptions_df.loc[
        descriptions_df["Disease"] == diagnosis, "Description"
    ].values
    precautions = precautions_df.loc[
        precautions_df["Disease"] == diagnosis,
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"],
    ].values
    description = (
        description[0] if len(description) > 0 else "No description available."
    )
    precautions = (
        precautions[0] if len(precautions) > 0 else ["No precautions available."]
    )
    return description, precautions


# Function to fetch severity from main_df


def fetch_severity(diagnosis):
    severity = main_df.loc[
        main_df["Disease"] == diagnosis, "custom_severity_score"
    ].values
    return severity[0] if len(severity) > 0 else "Unknown"


# Streamlit UI
st.title("Healthcare Assistant")

st.markdown(
    """
    **Disclaimer: This tool is for informational purposes only.
    It is not a substitute for professional medical advice, or treatment.
    Always seek the advice of a qualified medical professional.**
    """
)

# Dropdown menu for symptoms
selected_symptoms = st.multiselect("Select symptoms:", possible_symptoms)

if st.button("Find Diagnosis"):
    if selected_symptoms:
        symptom_embeddings = [
            generate_symptom_embedding(symptom) for symptom in selected_symptoms
        ]
        aggregated_embedding = aggregate_embeddings(symptom_embeddings)

        # Fetch diagnosis embeddings from Weaviate
        diagnosis_data = fetch_diagnosis_embeddings(client)

        # Calculate similarities
        similarities = []
        for diagnosis in diagnosis_data:
            diagnosis_embedding = np.array(diagnosis["embedding"])
            similarity = cosine_similarity(aggregated_embedding, diagnosis_embedding)
            # Fetch severity from main_df
            severity = fetch_severity(diagnosis["diagnosis"])
            similarities.append((diagnosis["diagnosis"], similarity, severity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter for top 5 unique diagnoses
        unique_diagnoses = {}
        top_diagnoses = []
        for diagnosis, similarity, severity in similarities:
            if diagnosis not in unique_diagnoses:
                unique_diagnoses[diagnosis] = similarity
                description, precautions = fetch_additional_data(diagnosis)
                top_diagnoses.append(
                    (diagnosis, similarity, severity, description, precautions)
                )
                if len(top_diagnoses) == 5:
                    break

        # Display results
        if top_diagnoses:
            st.write("Possible Diagnoses:")
            for (
                diagnosis,
                similarity,
                severity,
                description,
                precautions,
            ) in top_diagnoses:
                st.write(f"### {diagnosis} ({similarity:.0%} symptom match)")
                st.write(f"**Severity:** {severity}/10")
                st.write(f"**Description:** {description}")
                st.write("**Precautions:**")
                for precaution in precautions:
                    st.write(f"- {precaution}")
        else:
            st.write("No similar diagnoses found.")
    else:
        st.write("Please select symptoms to find a diagnosis.")
else:
    st.write("Please select symptoms to find a diagnosis.")
