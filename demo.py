from llama_index import SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext, GPTVectorStoreIndex, load_index_from_storage, StorageContext
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display
import streamlit as st
from dotenv.main import load_dotenv
load_dotenv()

st.title('Jellyfish Demo')


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context)

    index.set_index_id("vector_index")
    index.storage_context.persist('./storage')

    return index


def ask_ai():
    storage_context = StorageContext.from_defaults(
        persist_dir='storage')
    index = load_index_from_storage(
        storage_context, index_id="vector_index")
    # while True:

    question = st.text_input(
        "What would you like to ask the Jellyfish Chatbot?")
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    # display(Markdown(f"Response: <b>{response.response}</b>"))
    st.write(response.response)


construct_index("context_data/data")
ask_ai()
