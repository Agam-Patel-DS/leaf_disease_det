import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import streamlit as st
from langchain_community.document_loaders.onedrive_file import CHUNK_SIZE
from langchain_groq import ChatGroq
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Leaf Disease Detection",  # App name in the browser tab
    page_icon="L",              # Icon in the browser tab
    layout="centered",           # Layout of the app ("centered" or "wide")
    initial_sidebar_state="expanded"  # Sidebar state ("expanded" or "collapsed")
)

st.title("Leaf Disease Detection")
from src.prediction import prediction_config, predict
pred_params=yaml.safe_load(open("params.yaml"))["prediction"]

config=prediction_config(pred_params)

pred=predict(config)

from PIL import Image


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Special Features", "Detect", "AI Chat", "Models Used", "Contributors"])
with tab1:
    st.header("Project Overview")
    st.write("""
        This project is designed to detect leaf diseases in plants using machine learning.
        The model classifies images of plant leaves into categories such as healthy or diseased.
        It uses a deep learning model trained on a dataset of plant images. 
        The goal of this project is to assist farmers and researchers in identifying diseases in crops 
        efficiently, helping in better crop management and early intervention to prevent crop damage.
    """)
    
    st.subheader("Model Workflow")
    st.write("""
        1. **Input**: User uploads an image of a plant leaf.
        2. **Preprocessing**: The image is resized and normalized for the model.
        3. **Prediction**: The trained model classifies the leaf as healthy or diseased.
        4. **Output**: The result is displayed, including the type of disease (if any).
    """)

    st.subheader("Technologies Used")
    st.write("""
        - **Machine Learning**: Convolutional Neural Networks (CNN)
        - **Libraries**: TensorFlow, Keras, Streamlit
        - **Data**: Plant disease dataset with images of healthy and diseased leaves.
    """)
    st.markdown("---")
    st.markdown("""<p style='text-align: center;'><b>Copyrights</b> @ Agam Patel, Aman Agnihotri, Anushri Tiwari</p>""", unsafe_allow_html=True)

with tab2:
    st.header("Special Features of the Project")
    
    # Feature 1: Multi-Model Support
    st.markdown("#### 1. Multi-Model Support")
    st.write("""
        The application supports multiple models, allowing for flexible deployment. 
        It can easily switch between different trained models to detect various leaf diseases.
        This ensures that users get the best possible result depending on the type of leaf or disease.
    """)
    
    # Feature 2: AI Assistance
    st.markdown("#### 2. AI Assistance")
    st.write("""
        The AI-powered assistant provides real-time predictions and suggestions. 
        It is designed to help users by offering disease identification, 
        preventive measures, and even guidance on how to treat affected plants.
    """)

    # Feature 3: Accuracy
    st.markdown("#### 3. High Accuracy")
    st.write("""
        Our model achieves an accuracy rate of 92%, which ensures high confidence in the predictions.
        This makes it a reliable tool for farmers and agricultural experts to monitor and manage plant health.
    """)

    # Feature 4: Easy to Use
    st.markdown("#### 4. Easy to Use")
    st.write("""
        The user interface is designed to be intuitive and easy to navigate. 
        Uploading an image, making predictions, and viewing results are all done with a few clicks.
        It is suitable for both beginners and experts in agriculture.
    """)
    
    # Feature 5: Real-Time Predictions
    st.markdown("#### 5. Real-Time Predictions")
    st.write("""
        Get immediate predictions on the health status of the leaf. 
        The model processes the uploaded images and provides results instantly, saving time and effort.
    """)
    st.markdown("---")
    st.markdown("""<p style='text-align: center;'><b>Copyrights</b> @ Agam Patel, Aman Agnihotri, Anushri Tiwari</p>""", unsafe_allow_html=True)

with tab3:
    st.header("Upload the Image here!")
    image = st.file_uploader(
        "Upload an Image", 
        type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"]  # Specify accepted file types
    )
    
    if image is not None:
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success("Image uploaded successfully!")
        options = ['Best Model', 'Mobile_Net', 'Resnet', 'Inceptionnet', 'Densenet']

        # Create a dropdown with a default value (index 1 means 'Option 2' will be selected by default)
        selected_option = st.selectbox("Select an option", options, index=1)
        image = Image.open(image)
        disease=pred.predict_image(image,selected_option)
        if st.button("Predict"):
            st.warning(f"The Disease is :")
            st.markdown(f"###### {disease}")
    else:
        st.warning("Please upload an image file.")

    st.markdown("---")

    st.subheader("Steps to Use the Leaf Disease Detection App")
    
    # Step 1: Upload an Image
    st.markdown("##### 1. Upload an Image of the Leaf")
    st.write("""
        Start by uploading an image of the plant leaf you want to analyze. 
        Ensure that the image is clear, well-lit, and focused on the leaf for accurate results.
    """)
    
    # Step 2: Select Model (If applicable)
    st.markdown("##### 2. Select a Model (Optional)")
    st.write("""
        If you have multiple models to choose from (e.g., for different plant species or disease types), 
        select the model that best suits your needs from the dropdown list or options.
    """)
    
    # Step 3: Click "Predict"
    st.markdown("##### 3. Click 'Predict'")
    st.write("""
        After uploading the image and selecting the model, click the "Predict" button to run the prediction. 
        The model will analyze the leaf image and identify any potential diseases.
    """)

    # Step 4: View Results
    st.markdown("##### 4. View Results")
    st.write("""
        Once the prediction is complete, the result will be displayed on the screen. 
        It will include the type of disease (if any) and any recommendations for treatment or prevention.
    """)

    # Step 5: Take Action
    st.markdown("##### 5. Take Action Based on Results")
    st.write("""
        Based on the output from the model, follow the suggested actions for managing the identified disease. 
        You can also download the report or save the results for further reference.
    """)

    st.markdown("---")
    st.markdown("""<p style='text-align: center;'><b>Copyrights</b> @ Agam Patel, Aman Agnihotri, Anushri Tiwari</p>""", unsafe_allow_html=True)


with tab4:
    st.subheader("AI Assistance")

    user_input = st.text_area(
        label="If you have any query regarding the disease please ask me!",
        height=150,  # Adjust height for larger input area
        placeholder="Type or paste your question here..."
    )
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
    groq_api_key=os.getenv("GROQ_API_KEY")

    llm=ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    prompt=ChatPromptTemplate.from_template(
      """
      You are an expert leaf disease recognition system. Asnwer the questions 
      in a helpful way based on the provided context only.
      Please provide the most accurate response based on the question.
      <context>
      {context}
      <context>
      Question:{input}
      """
    )

    def create_vector_embeddings():
      if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("pdfs") #Data Ingestion
        st.session_state.docs=st.session_state.loader.load() #Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    user_prompt=user_input
    
    st.warning("Wait for the Vector Embeddings: Send Button will come after finishing.")
    create_vector_embeddings()

    import time

 
      

    if st.button("Send"):
      if user_prompt:
        document_chain=create_stuff_documents_chain(llm, prompt)
        retriever=st.session_state.vectors.as_retriever()
        retriever_chain=create_retrieval_chain(retriever, document_chain)
        start=time.process_time()
        response=retriever_chain.invoke({"input":user_prompt})
        print(f"Response time = {time.process_time()-start}")
        st.write(response["answer"])

    st.markdown("---")
    st.markdown("""<p style='text-align: center;'><b>Copyrights</b> @ Agam Patel, Aman Agnihotri, Anushri Tiwari</p>""", unsafe_allow_html=True)


with tab5:
    st.header("Models Used in Leaf Disease Detection")
    
    # MobileNet Model
    st.subheader("MobileNet - Accuracy: 96.3%")
    st.write("""
        MobileNet is a lightweight, efficient model designed for mobile and embedded vision applications. 
        It uses depthwise separable convolutions, significantly reducing the number of parameters while maintaining 
        good accuracy, making it ideal for real-time applications like leaf disease detection.
    """)
    
    # Add MobileNet Architecture Image
    #st.image("Linkedin Banner.png", caption="MobileNet Architecture", use_column_width=True)

    # ResNet Model
    st.subheader("ResNet - Accuracy: 84%")
    st.write("""
        ResNet (Residual Networks) is a deep neural network architecture that uses residual connections 
        to prevent vanishing gradients. It is particularly useful for training very deep networks, making it 
        a strong choice for complex image classification tasks.
    """)
    
    # Add ResNet Architecture Image
    #st.image("Linkedin Banner.png", caption="ResNet Architecture", use_column_width=True)

    # DenseNet Model
    st.subheader("DenseNet - Accuracy: 93%")
    st.write("""
        DenseNet (Densely Connected Convolutional Networks) is an architecture where each layer is connected 
        to every other layer. This helps with better feature reuse and gradient flow, improving the model’s 
        performance, particularly on smaller datasets.
    """)
    
    # Add DenseNet Architecture Image
    #st.image("Linkedin Banner.png", caption="DenseNet Architecture", use_column_width=True)

    st.markdown("---")
    st.markdown("""<p style='text-align: center;'><b>Copyrights</b> @ Agam Patel, Aman Agnihotri, Anushri Tiwari</p>""", unsafe_allow_html=True)



# with tab6:
#     st.header("Contributors to the Leaf Disease Detection Project")
    
#     # Contributor 1: Alice
#     col1, col2 = st.columns([3, 2])  # Create two columns with a ratio of 3:2
    
#     with col1:
#         st.subheader("Agam Patel - Project Lead & Data Scientist")
#         st.write("""
#             Agam led the project and was responsible for collecting and preprocessing the dataset. 
#             He also trained the models (MobileNet, ResNet, DenseNet) and optimized their performance. 
#             Agam's efforts ensured that the models performed at a high accuracy, providing reliable results for disease detection.
#         """)
#         st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/alice-profile/)")  # Replace with actual link
    
    
#     with col2:
#         image = Image.open("person.jpg")  # Replace with the correct path for the image
#         st.image(image, caption="Agam Patel", use_column_width=True)
    
#     # Contributor 2: Bob
#     col1, col2 = st.columns([3, 2])  # Create two columns with a ratio of 3:2
    
#     with col1:
#         st.subheader("Aman Agnihotri - Full Stack Developer")
#         st.write("""
#             Aman developed the Streamlit app and the user interface. He integrated the models into the app 
#             and ensured that users could easily upload images and view predictions. Aman also worked on making 
#             the app intuitive and user-friendly.
#         """)
#         st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/alice-profile/)")  # Replace with actual link
    
    
#     with col2:
#         image = Image.open("person.jpg")  # Replace with the correct path for the image
#         st.image(image, caption="Bob", use_column_width=True)
    
#     # Contributor 3: Claire
#     col1, col2 = st.columns([3, 2])  # Create two columns with a ratio of 3:2
    
#     with col1:
#         st.subheader("Anushri Tiwari - Research & Model Development")
#         st.write("""
#             Anushri was responsible for researching and selecting the best model architectures for this project. 
#             She fine-tuned the models to improve their accuracy and ensured that the models could handle 
#             various types of leaf disease images effectively.
#         """)
#         st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/alice-profile/)")  # Replace with actual link
    
#     with col2:
#         image = Image.open("person.jpg")  # Replace with the correct path for the image
#         st.image(image, caption="Claire", use_column_width=True)


#     st.markdown("---")
#     st.markdown("""<p style='text-align: center;'><b>Copyrights</b> @ Agam Patel, Aman Agnihotri, Anushri Tiwari</p>""", unsafe_allow_html=True)
