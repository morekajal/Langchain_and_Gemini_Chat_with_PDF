## Conversational QnA Chatbot

import streamlit as st

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI

## Streamlit UI
st.set_page_config(page_title = "Conversational QnA Chatbot")
st.header("Hey, I am your Fitness Advisor to help you!!")

from dotenv import load_dotenv
load_dotenv()                    # while deplyoing the app on hugging face, we can comment out this 2 dotenv code
import os

chat = ChatOpenAI(openai_api_key=os.getenv("OPEN_API_KEY"), temperature=0.3)

# define function
# We need to store the 3 messages - human, system and AI in sessions so that entire chatbot will be able to remember the context
# Streamlit has session_state to maintain these sessions

# If some session['key'] is not present, we to provide some response
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content = "You are a Fitness Advisor")
    ]

## Function to load openAI model and get responses
def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    # once we get the answer append over AIMessage
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))

    return answer.content

input = st.text_input("Input : ", key = "input")
response = get_chatmodel_response(input)

submit = st.button('Ask your question')

# if ask button is clicked

if submit:
    st.subheader(" The Response is : ")
    st.write(response)