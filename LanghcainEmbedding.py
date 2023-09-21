import os

import openai
from langchain import FAISS, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

directory = 'db'
openai_api_key = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = openai_api_key
file = 'data/Disciplined Entrepreneurship.pdf'
openai.api_key = openai_api_key
temperature = 0.3


def embed_pdf():
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    vector_index = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=openai_api_key))
    vector_index.save_local(directory)


def load_embedding():
    vector_store = FAISS.load_local(directory, OpenAIEmbeddings(openai_api_key=openai_api_key))

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, openai_api_key=openai_api_key)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, memory_key='chat_history',
                                             input_key='query', output_key='result', return_messages=True)

    prompt_template = """Give as many details as possible. Anticipate future questions and try to answer them 
    preemptively. This is the current context. Write minimum two paragraphs for each question. 
    {context}
    
    Question: {question}"""
    PROMPT = PromptTemplate.from_template(prompt_template)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def chat_pdf(qa, question):
    response = qa({"query": question})

    for source_doc in response['source_documents']:
        print(f"Found in: {source_doc}")

    return response['result']


if __name__ == '__main__':
    st.title('24 steps to Disciplined Entrepreneurship')

    # textbox for user input
    question = st.text_input("Enter your question about the book:")

    # conversation history
    conversation_history = st.session_state.get("conversation_history", [])

    # embed_pdf()
    qa_interface = load_embedding()

    # Create a slider for adjusting the temperature
    temperature = st.sidebar.slider(
        "Adjust the temperature",
        0.0, 1.0, temperature, 0.01
    )

    # ask questions
    # button
    if st.button("Ask"):
        # query = input("Enter question:\n")
        response = chat_pdf(qa_interface, question)

        # Update the conversation history
        conversation_history.append({"user": question, "AI": response})
        st.session_state.conversation_history = conversation_history

        # Display the conversation history
    for entry in reversed(conversation_history):
        st.write(f"You: {entry['user']}")
        st.write(f"AI: {entry['AI']}")
        st.write("---")
