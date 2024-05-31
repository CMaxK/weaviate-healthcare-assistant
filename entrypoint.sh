#!/bin/bash

# Set PYTHONPATH to include the current directory - allows finding of the helpers module needed
# for the processing scripts
export PYTHONPATH=/app

# Wait for Weaviate to be ready
echo "Checking network and services..."
until nc -z weaviate 8080; do
  echo 'Waiting for Weaviate...'
  sleep 5
done

echo "Weaviate is ready!"

echo "Running preprocessing script..."
python3 -m helpers.preprocessing

echo "Running generate symptom embeddings script..."
python3 -m helpers.generate_symptom_embeddings

echo "Running generate diagnosis embeddings script..."
python3 -m helpers.generate_diagnosis_embeddings

# Start the Streamlit app
streamlit run app.py
