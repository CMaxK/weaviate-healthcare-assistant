import logging

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer

import weaviate
from helpers.helpers import aggregate_diagnosis_embeddings

# Load BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")


def generate_diagnosis_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


# Load main_df
main_df = pd.read_csv("data/dataset.csv")

# Initialize Weaviate client
client = weaviate.Client("http://weaviate:8080")

# Ensure schema for Diagnosis class exists
try:
    client.schema.get("Diagnosis")
    logging.info("Diagnosis class already exists.")
except weaviate.exceptions.UnexpectedStatusCodeException:
    logging.info("Creating Diagnosis class.")
    schema = {
        "classes": [
            {
                "class": "Diagnosis",
                "properties": [
                    {"name": "diagnosis", "dataType": ["string"]},
                    {"name": "severity", "dataType": ["int"]},
                    {"name": "embedding", "dataType": ["number[]"]},
                ],
            }
        ]
    }
    client.schema.create(schema)

for index, row in main_df.iterrows():
    symptoms = [
        row[f"Symptom_{i}"] for i in range(1, 18)
        if pd.notna(row[f"Symptom_{i}"])
    ]
    symptom_embeddings = [
        generate_diagnosis_embedding(symptom.strip()) for symptom in symptoms
    ]
    diagnosis_embedding = aggregate_diagnosis_embeddings(symptom_embeddings)

    properties = {
        "diagnosis": row["Disease"],
        "severity": int(row["severity_tally"]),
        "embedding": diagnosis_embedding,
    }
    client.data_object.create(properties, "Diagnosis")

    # Store the diagnosis embedding back in main_df for further use (optional)
    main_df.at[index, "diagnosis_embedding"] = str(diagnosis_embedding)


# Save the updated dataframe
main_df.to_csv("data/dataset.csv", index=False)

logging.info("Diagnosis embeddings generated and stored.")
