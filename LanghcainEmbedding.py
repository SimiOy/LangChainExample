import os

from langchain import FAISS, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

directory = 'db'
print(st.secrets['OPENAI_API_KEY'])
os.environ['OPENAI_API_KEY'] = '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'
file = 'data/Disciplined Entrepreneurship.pdf'


def embed_pdf():
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    vector_index = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']))
    vector_index.save_local(directory)


def load_embedding():
    vector_store = FAISS.load_local(directory, OpenAIEmbeddings())

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=os.environ['OPENAI_API_KEY'])
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
    st.title("PDF Chatbot")

    # textbox for user input
    question = st.text_input("Enter your question about the book:")

    # embed_pdf()
    qa_interface = load_embedding()

    # ask questions
    # button
    if st.button("Ask"):
        # query = input("Enter question:\n")
        response = chat_pdf(qa_interface, question)

        st.write(response)
