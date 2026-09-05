#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#HW1

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader #Library used to read PDF files

# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

def read_pdf(file) -> str:  #Reads a PDF file into a single string
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

@st.cache_data  #Caches result
def is_valid_key(key: str) -> bool:  #Validation function
    try:
        OpenAI(api_key=key).models.list()  #Checks key
        return True
    except Exception:
        return False

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
elif not is_valid_key(openai_api_key):  #Validate the API key when entered
    st.error("Invalid API key. Please try again.")  #Error displayed
else:
    st.success("Access granted!")  #Confirmation

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)",
        type=("txt", "pdf"),  #Only .txt and .pdf
    )
    
    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        #***
        file_extension = uploaded_file.name.split('.')[-1]  #Grabs whatever follows the last dot
        if file_extension == 'txt': #For .txt files
            document = uploaded_file.read().decode()  #Decodes the raw bytes into a string
        elif file_extension == 'pdf': #For .pdf files
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")  #Error displayed
            st.stop()  #Stops the run

        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        model_options = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"]  #Available models
        selected_model = st.selectbox("Model",
            model_options,
            index=None, #Nothing preselected
            placeholder="Choose a model", #Shown while the selectbox is empty
        )

        if selected_model: #No generation until user sleects a model

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)
