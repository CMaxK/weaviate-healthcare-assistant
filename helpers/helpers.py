import logging

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

import weaviate

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Weaviate client
client = weaviate.Client("http://weaviate:8080")

# Load BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")


def fetch_diagnosis_embeddings(client):
    query = """
    {
      Get {
        Diagnosis {
          diagnosis
          embedding
        }
      }
    }
    """
    response = client.query.raw(query)
    if (
        response
        and "data" in response
        and "Get" in response["data"]
        and "Diagnosis" in response["data"]["Get"]
    ):
        return response["data"]["Get"]["Diagnosis"]
    return []


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def generate_symptom_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embedding


def aggregate_embeddings(embeddings):
    if len(embeddings) == 0:
        return []
    aggregated_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
    return aggregated_embedding


def aggregate_diagnosis_embeddings(embeddings):
    return torch.mean(torch.tensor(embeddings), dim=0).tolist()


def ensure_correct_format(df, embedding_column):
    correct_embeddings = []
    for embedding in df[embedding_column]:
        # Flatten the embedding if it's nested
        if isinstance(embedding[0], list):
            embedding = [item for sublist in embedding for item in sublist]

        # Ensure all elements are floats
        embedding = [float(x) for x in embedding]

        correct_embeddings.append(embedding)

    # Check dimensionality
    dim = len(correct_embeddings[0])
    for embedding in correct_embeddings:
        if len(embedding) != dim:
            raise ValueError("Inconsistent embedding dimensionality")

    df[embedding_column] = correct_embeddings
