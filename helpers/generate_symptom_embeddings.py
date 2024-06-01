import pandas as pd
from transformers import BertTokenizer, BertModel
import weaviate
import logging
from helpers.helpers import generate_symptom_embedding, ensure_correct_format

# Load BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Load severity_df
severity_df = pd.read_csv('data/Symptom-severity.csv')

# Initialize Weaviate client
client = weaviate.Client("http://weaviate:8080")

# Ensure schema for Symptom class exists
try:
    client.schema.get("Symptom")
    logging.info("Symptom class already exists.")
except:
    logging.info("Creating Symptom class.")
    schema = {
        "classes": [
            {
                "class": "Symptom",
                "properties": [
                    {"name": "symptom", "dataType": ["string"]},
                    {"name": "weight", "dataType": ["int"]},
                    {"name": "embedding", "dataType": ["number[]"]},
                ],
            }
        ]
    }
    client.schema.create(schema)

# Generate embeddings and store in Weaviate
severity_df['embedding'] = severity_df['Symptom'].apply(
    generate_symptom_embedding)
ensure_correct_format(severity_df, 'embedding')

for index, row in severity_df.iterrows():
    properties = {
        "symptom": row['Symptom'],
        "severity": row['weight'],
        "embedding": row['embedding']
    }
    client.data_object.create(properties, "Symptom")

logging.info("Symptom embeddings stored in Weaviate.")

severity_df.to_csv("data/Symptom-severity.csv", index=False)
