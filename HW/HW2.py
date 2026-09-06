#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#HW2

import streamlit as st
from openai import OpenAI
from anthropic import Anthropic #Second LLM provider
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

def stream_text(provider, key, model, messages):
    if provider == "OpenAI":
        client = OpenAI(api_key=key)
        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
            yield chunk.choices[0].delta.content or "" #Pulls the text out of each chunk
    else:
        client = Anthropic(api_key=key)
        with client.messages.stream(model=model, max_tokens=1024, messages=messages) as stream:
            for text in stream.text_stream: #Anthropic yields plain text already
                yield text

def escape_dollars(stream): #Escapes $ so Streamlit does not read it as LaTeX
    for text in stream:
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

#st.sidebar.divider() #Separates the model section

st.sidebar.subheader(":material/computer: Model Selection") #Section heading
st.sidebar.caption("Select LLM provider") #Caption
provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic"],
    index=0, #OpenAI preselected
    label_visibility="collapsed",
)

models = {
    "OpenAI": {"basic": "gpt-5.4-nano", "advanced": "gpt-5.4-mini",
               "basic_label": "GPT-5.4 Nano", "advanced_label": "GPT-5.4 Mini"},
    "Anthropic": {"basic": "claude-haiku-4-5-20251001", "advanced": "claude-sonnet-5",
                  "basic_label": "Claude Haiku 4.5", "advanced_label": "Claude Sonnet 5"},
} #Maps each provider to its basic and advanced model

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)
tier = "advanced" if use_advanced else "basic" #Which tier the checkbox points to
selected_model = models[provider][tier] #Model selection sent to the API
selected_label = models[provider][tier + "_label"]

st.sidebar.caption(f"_You are using model {selected_label}_") #Shows which model is active

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

def is_valid_key(provider: str, key: str) -> bool: #Validation function, now checks against the selected provider
    try:
        if provider == "OpenAI":
            OpenAI(api_key=key).models.list() #Checks key
        else:
            Anthropic(api_key=key).models.list() #Checks key
        return True
    except Exception:
        return False

secret_names = {"OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY"} #Maps each provider to its secrets.toml entry
api_key = st.secrets.get(secret_names[provider], "")  #Key read from .streamlit/secrets.toml (or App settings > Secrets)

summary_instructions = {
    "100-Word Summary": "Summarize the document in about 100 words.",
    "2 Paragraph Summary": "Summarize the document in 2 connecting paragraphs.",
    "5-Bullet Summary": "Summarize the document in 5 bullet points.",
} #Maps selection to the instruction sent to the LLM

if not api_key:
    st.info(f"Please add your {provider} API key to continue.", icon="🗝️")
elif not is_valid_key(provider, api_key): #Validate the API key when the provider changes
    st.error(f"Invalid {provider} API key. Please try again.") #Error displayed
else:
    st.success("Access granted!") #Confirmation
    url = st.text_input("Enter a URL",placeholder="https://example.com",)

    if url and generate and inputs_ready: #Runs once a URL is entered and selections are made
        document = read_url_content(url) #Pulls the text off the page
        if not document: #Nothing usable came back
            st.stop() #Stops the run
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {summary_instructions[summary_type]} Write the entire summary in {language}.", #Summary format and output language are now the instruction
            }
        ]

        if selected_model:
            st.write_stream(escape_dollars(stream_text(provider, api_key, selected_model, messages))) #Streams from whichever provider is selected
