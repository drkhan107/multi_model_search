import os

import streamlit as st
from dotenv import load_dotenv
#from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from PIL import Image

import futils

from assistant import Assistant
# from blip2 import generate_embeddings
from clip import generate_embeddings
from classifier import Classifier
from langchain_community.vectorstores import FAISS # Import FAISS


#load_dotenv("env/connection.env")
load_dotenv()

llms=futils.get_llms()

# Define the path where the FAISS index will be saved
FAISS_INDEX_PATH = "faiss_screenshots_index"
FAISS_INDEX_PATH = "faiss_image_clip_index"
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

classify=True

st.title("Welcome to WaaW's Fashion Assistant")

user_input = st.text_input("Hi, I an WaaW Fashion Assistant. How can I help you today?")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Submit"):

    # understand what the user is asking for
    if classify:
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

    if classify and len(classification.category)>0:
        filter_set={"category": {"$in": classification.category}}
    else:
        filter_set={}
    if classify and classification.number_of_items>0: num=  classification.number_of_items 
    else: num=11

    clothes = faiss_db.similarity_search_by_vector(
        embedding, k=num, filter=filter_set
    )
    for clothe in clothes:
        retrieved_items.append({"bytesBase64Encoded": futils.encode_image_to_base64(clothe.page_content)})
        retrieved_items_path.append(clothe.page_content)

    # get assistant's recommendation
    assistant_output = assistant.get_advice(user_input, retrieved_items, len(retrieved_items))
    st.write(assistant_output.answer)
    



    if 'selected_image_path' not in st.session_state:
        st.session_state.selected_image_path = None

    # --- Display Image Grid ---
    if len(retrieved_items_path) > 0:
        items_per_row = 3
        num_items = len(retrieved_items_path)

        st.write("Click the 'View' button below an image to see it full size.") # User guidance

        # Iterate through the list in steps of items_per_row (3)
        for i in range(0, num_items, items_per_row):
            # Create a new row with items_per_row columns for each chunk
            cols = st.columns(items_per_row)

            # Iterate through the columns in the current row
            # This handles the case where the last row might have fewer than 3 images
            for j in range(items_per_row):
                # Calculate the index of the item in the original list
                item_index = i + j

                # Check if the item_index is within the bounds of the list
                if item_index < num_items:
                    image_path = retrieved_items_path[item_index]

                    # Display the thumbnail in the current column
                    # Added a caption for context
                    cols[j].image(image_path, caption=f"Image {item_index+1}")

                    # Add a button below the image
                    # Use a unique key for each button in the loop
                    # When clicked, update session state
                    if cols[j].button(f"View", key=f"view_button_{item_index}"):
                        st.session_state.selected_image_path = image_path
                        # st.rerun() # Optional: Rerun immediately to show full image.
                                # Often not needed if the full image display is checked *after* this loop.

    # --- Display Full Size Image if selected ---
    if st.session_state.selected_image_path is not None:
        st.subheader("Full Size Image")

        # Add a button to close the full view
        if st.button("Close Full View"):
            st.session_state.selected_image_path = None
            # st.rerun() # Rerun to hide the full image

        # Display the selected full-size image
        # use_column_width=True makes it fill the column width, adjust as needed
        st.image(st.session_state.selected_image_path, use_column_width=True)


    user_input = st.text_input("")

else:
    st.warning("Please provide text.")






    