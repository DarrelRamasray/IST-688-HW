#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#HW2

import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

def read_url_content(url):  #Reads a web page into a single string
    try:
        response = requests.get(url)
        response.raise_for_status()  #Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")  #Error displayed #st.error(f"Error reading {url}: {e}")
        return None

def escape_dollars(stream): #Escapes $ so Streamlit does not read it as LaTeX
    for chunk in stream:
        text = chunk.choices[0].delta.content or "" #Pulls the text out of each chunk
        yield text.replace("$", "\\$")

st.sidebar.header("**Settings:**")
st.sidebar.caption("Configure Output Format & AI Model")

#Output Type
st.sidebar.subheader(":material/translate: Language") #Section heading
st.sidebar.caption("Select Language")  #Caption

language = st.sidebar.selectbox("Language", ["English", "Mandarin Chinese", "Hindi", "Spanish", "French"],
    index=0, #English preselected
    label_visibility="collapsed",
)

#Summary Type
st.sidebar.subheader(":material/description: Specify Output Format") #Section heading
st.sidebar.caption("Select type of summary") #Caption

summary_type = st.sidebar.selectbox("Specify Output Format", ["100-Word Summary", "2 Paragraph Summary", "5-Bullet Summary"],
    index=None,
    placeholder="Choose a format",
    label_visibility="collapsed",
)

st.sidebar.divider() #Separates the model section

st.sidebar.subheader(":material/computer: Model Selection") #Section heading

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)
basic_model = "gpt-5.4-nano" #Model used by default
advanced_model = "gpt-5.4-mini" #Model used when the box is checked
selected_model = advanced_model if use_advanced else basic_model #Model selection sent to the API

st.sidebar.caption("_Now using model GPT-5.4 Mini_" if use_advanced else "_You are using model GPT-5.4 Nano_") #Shows which model is active

generate = st.sidebar.button("Generate Summary", type="primary") #Nothing is sent to the API until this is clicked

if st.sidebar.button("Clear Cache"): #Clears the cached key validation
    st.cache_data.clear()

inputs_ready = bool(summary_type) #A model is always set

#Show title and description.
st.title(":blue[HW 2:] :grey[Deep] Scan Protocol") #Title
st.write("Enter a URL below, then select summary format and model. ")

if generate and not inputs_ready: #Error shown when either sidebar selection is missing
    st.error("Error! Please choose a summary format before generating.")

@st.cache_data #Caches result
def is_valid_key(key: str) -> bool: #Validation function
    try:
        OpenAI(api_key=key).models.list() #Checks key
        return True
    except Exception:
        return False

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")  #Key read from .streamlit/secrets.toml (or App settings > Secrets)

summary_instructions = {
    "100-Word Summary": "Summarize the document in about 100 words.",
    "2 Paragraph Summary": "Summarize the document in 2 connecting paragraphs.",
    "5-Bullet Summary": "Summarize the document in 5 bullet points.",
} #Maps selection to the instruction sent to the LLM

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
elif not is_valid_key(openai_api_key): #Validate the API key when entered
    st.error("Invalid API key. Please try again.") #Error displayed
else:
    st.success("Access granted!") #Confirmation
    client = OpenAI(api_key=openai_api_key)
    url = st.text_input("Enter a URL",placeholder="https://example.com",) #URL is entered at the top of the page

    if url and generate and inputs_ready: #Runs once a URL is entered and selections are made
        #Process the URL and the sidebar selections.
        document = read_url_content(url) #Pulls the text off the page
        if not document: #Nothing usable came back
            st.stop() #Stops the run
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {summary_instructions[summary_type]} Write the entire summary in {language}.", #Summary format and output language are now the instruction
            }
        ]
        
        #messages = [
        #    {
        #        "role": "system",
        #        "content": f"You are a summarizer. Write your entire response in {language}, regardless of the language of the source document.",
        #    },
        #    {
        #        "role": "user",
        #        "content": f"Here's a document: {document} \n\n---\n\n {summary_instructions[summary_type]}",
        #    }
        #]

        if selected_model: #No generation until user selects a model
            #Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=True,
            )
            #Stream the response to the app using `st.write_stream`.
            st.write_stream(escape_dollars(stream))