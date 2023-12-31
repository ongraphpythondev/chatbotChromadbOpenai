import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dotenv import load_dotenv
import pickle
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
import tiktoken
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens    

load_dotenv()
try: 
    openai_api_key=st.secrets["OPENAI_API_KEY"]
    demo=st.secrets['DEMO']
    max_token_user=st.secrets["MAX_TOKEN_USER"]

except:
    openai_api_key=os.getenv('OPENAI_API_KEY')
    demo=os.getenv('DEMO')
    max_token_user=os.getenv("MAX_TOKEN_USER")

    
def main():
    if demo:
        with st.sidebar:
            
            st.markdown('''
            ## About
            This app is an LLM-powered chatbot built using:
            - [Streamlit](https://streamlit.io/)
            - [LangChain](https://python.langchain.com/)
            - [OpenAI](https://platform.openai.com/docs/models) LLM model

            ''')
            add_vertical_space(5)
        st.header("Chat with PDF 💬 using chromadb")


        # upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type='pdf')

        # st.write(pdf)
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0,
                )
            chunks = text_splitter.split_text(text=text)

            # # embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
            # st.write(chunks)

            embeddings=HuggingFaceEmbeddings()
            if os.path.exists(f"./chroma_db/{store_name}"):
                db = Chroma(persist_directory=f"./chroma_db/{store_name}", embedding_function=embeddings)
                
                st.write('Embeddings Loaded from the Disk')
            else:
                st.write('Create Embeddings...')
                db = Chroma.from_texts(chunks, embeddings, persist_directory=f"./chroma_db/{store_name}")
            
            


            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
            

            # st.write(query)
            # retriever = db.as_retriever()
            
            llm=OpenAI(openai_api_key=openai_api_key,temperature=0)
            user_token=num_tokens_from_string(query, "gpt-3.5-turbo")
            # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            if query:
                if user_token < int(max_token_user):
                    docs = db.similarity_search(query,k=3)
                    print(docs)

                
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    

                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)
                else:
                    st.write(f"EXCEED ALLOCATED PROMPT,\n MAX TOKEN: {max_token_user} \n YOUR TOKEN: {user_token}")    
    else:
        st.header("This App is Private!!!")
if __name__ == '__main__':
    main()