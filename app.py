import os

import streamlit as st
from dotenv import load_dotenv
#from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from PIL import Image

import futils

from assistant import Assistant
from blip2 import generate_embeddings
from classifier import Classifier
from langchain_community.vectorstores import FAISS # Import FAISS


#load_dotenv("env/connection.env")
load_dotenv()

llms=futils.get_llms()

# Define the path where the FAISS index will be saved
FAISS_INDEX_PATH = "faiss_image_index"
hf_embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base")
try:
    faiss_db=None
    
    faiss_db = FAISS.load_local(f'{FAISS_INDEX_PATH}', hf_embeddings, allow_dangerous_deserialization=True)
    print(f'Loaded FAISS index with {faiss_db.index.ntotal} vectors.')
    
except Exception as e:
     print(f"Error loading Index: {e}")
     exit()

# CONNECTION_STRING = PGVector.connection_string_from_db_params(
#     driver=os.getenv("DRIVER"),
#     host=os.getenv("HOST"),
#     port=os.getenv("PORT"),
#     database=os.getenv("DATABASE"),
#     user='admin', #os.getenv("USERNAME"),
#     password=os.getenv("PASSWORD"),
# )

# vector_db = PGVector(
#     embeddings=None,#HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base"),  # does not matter for our use case
#     collection_name="fashion",
#     connection=CONNECTION_STRING,
#     use_jsonb=True,
# )

model = llms[1] #ChatVertexAI(model_name=os.getenv("MODEL_NAME"), project=os.getenv("PROJECT_ID"), temperarture=0.0)
classifier = Classifier(model)
assistant = Assistant(model)

st.title("Welcome to WaaW's Fashion Assistant")

user_input = st.text_input("Hi, I an WaaW Fashion Assistant. How can I help you today?")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Submit"):

    # understand what the user is asking for
    classification = classifier.classify(user_input)
    print("Classification    ",classification)

    if uploaded_file:

        image = Image.open(uploaded_file)
        image.save("input_image.jpg")
        embedding = generate_embeddings(image=image)

    else:

        # create text embeddings in case the user does not upload an image
        embedding = generate_embeddings(text=user_input)

    # create a list of items to be retrieved and the path
    retrieved_items = []
    retrieved_items_path = []
    # for item in classification.category:
    #     clothes = vector_db.similarity_search_by_vector(
    #         embedding, k=classification.number_of_items, filter={"category": {"$in": [item]}}
    #     )
    #     for clothe in clothes:
    #         retrieved_items.append({"bytesBase64Encoded": futils.encode_image_to_base64(clothe.page_content)})
    #         retrieved_items_path.append(clothe.page_content)

    if len(classification.category)>0:
        filter_set={"category": {"$in": classification.category}}
    else:
        filter_set={}
    if classification.number_of_items>0: num=  classification.number_of_items 
    else: num=5

    clothes = faiss_db.similarity_search_by_vector(
        embedding, k=num, filter=filter_set
    )
    for clothe in clothes:
        retrieved_items.append({"bytesBase64Encoded": futils.encode_image_to_base64(clothe.page_content)})
        retrieved_items_path.append(clothe.page_content)

    # get assistant's recommendation
    assistant_output = assistant.get_advice(user_input, retrieved_items, len(retrieved_items))
    st.write(assistant_output.answer)
    if len(retrieved_items) >0:
        cols = st.columns(len(retrieved_items))
        for col, retrieved_item in zip(cols, retrieved_items_path):
            print(retrieved_item)
            col.image(retrieved_item)

    user_input = st.text_input("")

else:
    st.warning("Please provide text.")






    