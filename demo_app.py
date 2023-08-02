import streamlit as st

import os
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import nltk
from constant import CHROMA_SETTINGS


load_dotenv()


persist_directory = os.environ.get('PERSIST_DIRECTORY')


openai_api=os.getenv("OPENAI_API_KEY")

loader = DirectoryLoader('./data/', glob="**/*.pdf", show_progress=True)#,use_multithreading=True ## unable to use it because of resource
docs = loader.load()

doc_sources = [doc.metadata['source']  for doc in docs]
print(doc_sources)
print(11111111111)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)


# embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS).from_documents(texts,embeddings)



retriever = db.as_retriever()
callbacks = [StreamingStdOutCallbackHandler()]

# llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
llm=OpenAI(openai_api_key=openai_api,temperature=0.2)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer= res['result']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)