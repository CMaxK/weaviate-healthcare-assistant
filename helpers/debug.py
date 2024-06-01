import pandas as pd
from transformers import BertTokenizer, BertModel
import weaviate
from helpers.helpers import generate_symptom_embedding


def query_diagnosis_debug(client, embedding):
    print(f"Debug: Querying with embedding: {embedding}")

    query = (
        client.query
        .get("Diagnosis", ["diagnosis", "severity", "embedding"])
        .with_near_vector({"vector": embedding})
        .with_additional(["distance"])
        .do()
    )

    print(f"Debug: Query result: {query}")

    return query


# Load BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Initialize Weaviate client
client = weaviate.Client("http://weaviate:8080")

# Check if any diagnosis data is present in Weaviate
diagnosis_check = client.query.get(
    "Diagnosis", ["diagnosis", "severity", "embedding"]).do()
print(f"Debug: Existing diagnosis data in Weaviate: {diagnosis_check}")

# Example user input
user_symptom = "sore throat"

# Generate embedding for the user input
user_embedding = generate_symptom_embedding(user_symptom)
print(
    f"Debug: Generated embedding for user symptom '{user_symptom}': {user_embedding}")

# Query the diagnosis
response = query_diagnosis_debug(client, user_embedding)

print("Query result:", response)

# Handle the response
diagnoses = response['data']['Get']['Diagnosis']
if not diagnoses:
    print("No relevant diagnosis found.")
else:
    for diagnosis in diagnoses:
        print(
            f"Diagnosis: {diagnosis['diagnosis']}, Severity: {diagnosis['severity']}, Distance: {diagnosis['_additional']['distance']}")
