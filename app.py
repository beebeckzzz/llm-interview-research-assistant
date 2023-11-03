# Streamlit
import streamlit as st

# LLMs
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# YouTube
from langchain.document_loaders import YoutubeLoader

#Scraping
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()

# Get your API keys set
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', False)

# Load up your LLM
def load_LLM(openai_api_key, model_name):
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key, max_tokens=2000, model_name=model_name)
    return llm

# Function to change our long text about a person into documents
def split_text(user_information):
    # First we make our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)

    # Then we split our user information into different documents
    docs = text_splitter.create_documents([user_information])

    return docs

# Prompts - We'll do a dynamic prompt based on the option the users selects
# We'll hold different instructions in this dictionary below
response_types = {
    'Interview Questions' : """
        Your goal is to generate interview questions that we can ask them
        Please respond with list of a 20 interview questions based on the topics above
    """,
    '1-Page Summary' : """
        Your goal is to generate a 1 page summary about them
        Please respond with a few short paragraphs that would prepare someone to talk to this person
    """
}

map_prompt = """You are a helpful AI bot that aids a user in research.
Below is information about a person named {persons_name}.
Information will include tweets, interview transcripts, and blog posts about {persons_name}
Use specifics from the research when possible

{response_type}

% START OF INFORMATION ABOUT {persons_name}:
{text}
% END OF INFORMATION ABOUT {persons_name}:

YOUR RESPONSE:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "persons_name", "response_type"])

combine_prompt = """
You are a helpful AI bot that aids a user in research.
You will be given information about {persons_name}.
Do not make anything up, only use information which is in the person's context

{response_type}

% PERSON CONTEXT
{text}

% YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "persons_name", "response_type"])

# Start Of Streamlit page
st.set_page_config(page_title="Research Assistant", page_icon=":robot:")

# Sidebar for OpenAI model selection and API key input
st.sidebar.header("OpenAI Settings")
model_name = st.sidebar.selectbox("Model", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"))

if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

# Instructions
instructions_expander = st.expander("Instructions")
with instructions_expander:
    st.markdown("""
    1. **Person's Name**: Enter the name of the person you want to research.
    2. **Tweets**: Paste the tweets of the person you want to research. You can copy the tweets directly from Twitter.
    3. **LinkedIn Posts**: Paste the LinkedIn posts of the person you want to research. You can copy the posts directly from LinkedIn.
    4. **YouTube URLs**: Enter the URLs of YouTube videos related to the person you want to research. If you want to enter multiple URLs, separate them with a comma. For example: `https://www.youtube.com/shorts/N37Ku_yfXws, https://www.youtube.com/shorts/rkSBhVuYXws`.
    5. **Web Page URLs**: Enter the URLs of web pages related to the person you want to research. If you want to enter multiple URLs, separate them with a comma. Make sure to include `https://` in the URLs. For example: `https://www.tesla.com/elon-musk`.
    6. **Output Type**: Select the type of output you want. You can choose between 'Interview Questions' and '1-Page Summary'.
    7. Click the "Generate Output" button.
    
    The app will then generate the requested output based on the provided information.
    """)

# Output type selection by the user
output_type = st.radio(
    "Output Type:",
    ('Interview Questions', '1-Page Summary'))

# Collect information about the person you want to research
person_name = st.text_input(label="Person's Name",  placeholder="Ex: Elon Musk", key="persons_name")
tweets = st.text_area(label="Tweets", placeholder="Paste the person's tweets here", key="tweets_input")
linkedin_posts = st.text_area(label="LinkedIn Posts", placeholder="Paste the person's LinkedIn posts here", key="linkedin_posts_input")
youtube_videos = st.text_input(label="YouTube URLs (Use , to seperate videos)",  placeholder="Ex: https://www.youtube.com/shorts/N37Ku_yfeso, https://www.youtube.com/shorts/rkSBhVuYXws", key="youtube_user_input")
webpages = st.text_input(label="Web Page URLs (Use , to seperate urls. Must include https://)",  placeholder="https://www.tesla.com/elon-musk", key="webpage_user_input")

# Output
st.markdown(f"### {output_type}:")

# Get URLs from a string
def parse_urls(urls_string):
    """Split the string by comma and strip leading/trailing whitespaces from each URL."""
    return [url.strip() for url in urls_string.split(',')]

# Get information from those URLs
def get_content_from_urls(urls, content_extractor):
    """Get contents from multiple urls using the provided content extractor function."""
    return "\n".join(content_extractor(url) for url in urls)

def pull_from_website(url):
    st.write("Getting webpages...")
    # Doing a try in case it doesn't work
    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return
    
    # Put your response in a beautiful soup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get your text
    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)
     
    return text


# Pulling data from YouTube in text form
def get_video_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript

button_ind = st.button("*Generate Output*", type='secondary', help="Click to generate output based on information")

# Checking to see if the button_ind is true. If so, this means the button was clicked and we should process the links
if button_ind:
    if not (tweets or linkedin_posts or youtube_videos or webpages):
        st.warning('Please provide tweets, LinkedIn posts, YouTube video URLs or website links', icon="⚠️")
        st.stop()

    if not OPENAI_API_KEY:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

    # Combine tweets, LinkedIn posts, and YouTube video transcripts into a single string
    video_text = get_content_from_urls(parse_urls(youtube_videos), get_video_transcripts) if youtube_videos else ""
    website_data = get_content_from_urls(parse_urls(webpages), pull_from_website) if webpages else ""

    user_information = "\n".join([tweets, linkedin_posts, video_text, website_data])

    user_information_docs = split_text(user_information)

    # Calls the function above
    llm = load_LLM(openai_api_key=OPENAI_API_KEY, model_name=model_name)

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template,
                                 # verbose=True
                                 )
    
    st.write("Sending to LLM...")


    # Here we will pass our user information we gathered, the persons name and the response type from the radio button
    output = chain({"input_documents": user_information_docs, # The seven docs that were created before
                    "persons_name": person_name,
                    "response_type" : response_types[output_type]
                    })

    st.markdown(f"#### Output:")
    st.write(output['output_text'])


