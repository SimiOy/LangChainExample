from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
import os
# import detectron2

# TODO: These dependencies have to be installed:
# !pip install openai
# !pip install tiktoken
# !pip install unstructured
# !pip install python-magic-bin
# !pip install chromadb
# !pip install "unstructured[local-inference]"
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# !pip install layoutparser[layoutmodels,tesseract]
# !pip install libmagic
# !pip install python-poppler
# !pip install pytesseract

persist_directory = 'db'
# TODO: Change OpenAI API Key here or in some .env file
os.environ['OPENAI_API_KEY'] = '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'


def create_vector_db():
    loader = DirectoryLoader("data/")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


def load_vector_db():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    # Now we can load the persisted database from disk, and use it as normal.
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


if __name__ == '__main__':
    # vector_store = load_vector_db()
    vector_store = create_vector_db()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=os.environ['OPENAI_API_KEY'])
    # memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory,
                                               verbose=True)
    while True:
        query = input("Enter question:\n")
        result = qa({"question": query})
        print(result['answer'])
