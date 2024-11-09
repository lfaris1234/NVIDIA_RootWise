import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from openai import OpenAI

# Initialize global variables
query_engine = None
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)
rag_data = []  # Placeholder for dynamically updated RAG data, i think i want it written to a file actually

def initialize_rag(file_path):
    global query_engine
    
    system_data_dir = "./system_data"
    
    # Ensure the system_data directory exists
    if not os.path.exists(system_data_dir):
        os.makedirs(system_data_dir)
        return "No data found. The system_data directory has been created but is empty."

    try:
        # Load all .txt files in the system_data directory
        documents = []
        for file_name in os.listdir(system_data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(system_data_dir, file_name)
                documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return "No .txt files found in the system_data directory."

        # Create or rebuild the index with loaded documents
        storage_context = StorageContext.from_defaults(persist_dir=system_data_dir)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        query_engine = index.as_query_engine()

        return "Query engine initialized successfully using files from system_data."
    
    except Exception as e:
        print(f"\n\n {e} \n\n")
        query_engine = None  # Reset query engine on failure
        return f"Failed to initialize query engine: {str(e)}"

# Function to handle file inputs for RAG
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents and create the index
def load_documents(file_objs):
    global index, query_engine 
    try:
        if not file_objs:
            return "No files selected."
        
        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found in the selected files."

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        query_engine = index.as_query_engine()
        return "Documents loaded successfully!"
    
    except Exception as e:
        return f"Error loading documents: {str(e)}"

# Add user inputs to the RAG database
def add_to_rag(season, ingredients, restrictions):
    global rag_data, query_engine

    new_entry = {
        "season": season,
        "ingredients": ingredients.split(','),
        "restrictions": restrictions.split(',')
    }
    rag_data.append(new_entry)

    # Simulate RAG database update
    documents = [
        f"Season: {new_entry['season']}, Ingredients: {', '.join(new_entry['ingredients'])}, Restrictions: {', '.join(new_entry['restrictions'])}"
    ]

    if not documents:
            return f"No new data here." 
    
    try:
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=False)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        query_engine = index.as_query_engine()

        return "Input added to RAG database!"
    
    except Exception as e:
        return f"Error updating rag: {str(e)}"


# Chat function for interactive conversation
def chat(message, history):
    global query_engine
    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role":"user","content":f"you are a helpful chatbot, respond to this message:{message} the best you can"}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
        )
    
    reply = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            reply += chunk.choices[0].delta.content

    if query_engine is None:
        return history + [(message, f"I don't have any documents to reference, but here's my best answer: {reply}")]

    try:
        response = query_engine.query(message)
        return history + [(message, response)]
    except Exception as e:
        return history + [(message, f"An error occurred: {str(e)}")]
    
# Function to stream responses 
def stream_response(message, history) :
    global query_engine 
    if query_engine is None:
        yield history + [("Please load documents first.", None)]
        return
    try:
        response = query_engine. query (message)
        partial_response = ""
        for text in response. response_gen:
            partial_response += text
            yield history + [(message, partial_response)]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

# Gradio app setup
with gr.Blocks() as demo:

    file_upload = gr.File(label="Upload Documents", file_types=[".txt", ".pdf"])
    load_button = gr.Button("Load Documents")
    load_button.click(load_documents, inputs=[file_upload], outputs=gr.Textbox(label="Status"))

    season_input = gr.Textbox(label="Season")
    ingredients_input = gr.Textbox(label="Ingredients (comma-separated)")
    restrictions_input = gr.Textbox(label="Dietary Restrictions (comma-separated)")

    add_rag_button = gr.Button("Add to RAG Database")
    add_rag_button.click(add_to_rag, inputs=[season_input, ingredients_input, restrictions_input], outputs=gr.Textbox(label="RAG Status"))

    chat_input = gr.Textbox(label="Ask me anything")
    chat_history = gr.Chatbot()
    
    chat_input.submit(chat, inputs=[chat_input, chat_history], outputs=chat_history)

__name__ = "__main__"
# Launching Gradio interface
if __name__ == "__main__":
  initialize_rag('./system_data')
  demo.queue()
  demo.launch()
  print("herelo")
