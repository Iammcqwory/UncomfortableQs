import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_model():
    table_qa = pipeline('table-question-answering', model='google/tapas-large-finetuned-wtq')
    return table_qa

def process_data(table_qa, table_data):
    response = table_qa(table=table_data) 
    return response

table_data = [
    ['What is your biggest weakness?', 'What are your bad traits?'],
    ['What is your biggest fear?', 'What scares you the most?'],
    ['What did you do last night?', 'Where were you last night?'],
    ['Do you love me?', 'Do you have feelings for me?'],
    ['Who are you texting?', 'Who are you talking to?'],
    ['Why did you break up with your ex?', 'Why did you end your last relationship?'],
    ['Who sent you this message?', 'Who sent you this text?'],
    ['What are you thinking about me?', 'What are you secretly thinking about me?'],
    ['Do you want to be friends?', 'Do you want to be just friends?'],
    ['Do you want to hook up?', 'Do you want to have sex?'],
    ['Do you love me?', 'Do you have feelings for me?'],
    ['Do you want to be friends?', 'Do you want to be just friends?'],
    ['Do you want to hook up?', 'Do you want to have sex?'],
]

table_qa = load_model() 

st.title('QA Bot')
query = st.text_input('Ask a question:')

if query:
    response = process_data(table_qa, [query, table_data])
    st.write(response['answer'])
