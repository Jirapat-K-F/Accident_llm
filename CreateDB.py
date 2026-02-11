import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import pandas as pd
from utils.reader import Reader
from utils.indexing import Indexing

folder_path = 'C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset'
folder_path_train = folder_path+'/train'
res_path ='C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/DataPrep/res/database'


# Use persistent client to save database to disk
client = chromadb.PersistentClient(path=res_path)
collection = client.get_or_create_collection(
    name="accident_report_summaries",
    metadata={"hnsw:space": "cosine"} 
)

# Get existing IDs to avoid reprocessing
try:
    existing_ids = collection.get()['ids']
    print(f"Found {len(existing_ids)} existing documents in collection")
except:
    existing_ids = []
    print("No existing documents found in collection, 0 documents found.")

indexer = Indexing(model_name="llama3")
reader = Reader()
count = 0
# Get all files in the folder
for filename in os.listdir(folder_path_train):
    file_path = os.path.join(folder_path_train, filename)
    
    # Only process files (not subdirectories)
    if os.path.isfile(file_path):
        # Skip if already processed
        if filename in existing_ids:
            print(f"Skipping already processed file: {filename}")
            continue
            
        try:
            content = reader.read(file_path)
            summary = indexer.summarize_with_llm(content)

            with open(res_path+'/'+filename + '.txt', 'w', encoding='utf-8') as f:
                f.write(summary)    # simulation return

            collection.add(
                documents=[summary],
                metadatas=[{"filename": filename}],
                ids=[filename]
            )
            count += 1
            print(f"Processed and added to ChromaDB: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"Processing complete. {count} new documents added to collection.")