import os
import time
import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Define your URLs here
URLS = [
    "https://python.langchain.com/docs/get_started/introduction",
    "https://www.iresearchnet.com/research-paper-examples/",
    # Add more URLs as needed
]

st.title("Chatbot with Update knowledge Base")

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="gemma-7b-it"
)

# Initialize tools
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. 
    Please provide the most accurate response based on the question
    
    Context:
    {context}
    
    Question: {input}
    
    If the answer is not in the context, please state that clearly.
    """
)

def fetch_wiki_arxiv_data(query):
    """Fetch data from Wikipedia and Arxiv based on the query."""
    documents = []
    
    try:
        # Get Wikipedia data
        wiki_data = wikipedia.run(query)
        if wiki_data:
            documents.append(Document(page_content=wiki_data, metadata={"source": "wikipedia"}))
    except Exception as e:
        st.warning(f"Wikipedia fetch error: {str(e)}")
    
    try:
        # Get Arxiv data
        arxiv_data = arxiv.run(query)
        if arxiv_data:
            documents.append(Document(page_content=arxiv_data, metadata={"source": "arxiv"}))
    except Exception as e:
        st.warning(f"Arxiv fetch error: {str(e)}")
    
    return documents

def vector_embeddings(query=None):
    """Create or update vector embeddings from multiple sources."""
    if "vectors" not in st.session_state:
        # Initialize embeddings
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = []
        
        # Load from predefined URLs
        loader = WebBaseLoader(URLS)
        web_docs = loader.load()
        documents.extend(web_docs)
        
        # Fetch Wikipedia and Arxiv data if query provided
        if query:
            extra_docs = fetch_wiki_arxiv_data(query)
            documents.extend(extra_docs)
        
        # Split documents and create vector store
        if documents:
            split_docs = st.session_state.text_splitter.split_documents(documents)
            st.session_state.vectors = FAISS.from_documents(
                split_docs,
                st.session_state.embeddings
            )
    else:
        # Update existing vector store with only Wiki/Arxiv data
        if query:
            documents = fetch_wiki_arxiv_data(query)
            if documents:
                split_docs = st.session_state.text_splitter.split_documents(documents)
                st.session_state.vectors.add_documents(split_docs)

# Sidebar for initializing and updating knowledge base
with st.sidebar:
    st.header("Knowledge Base Control")
    custom_query = st.text_input("Add custom search query for Wiki/Arxiv:")
    
    if st.button("Initialize/Update Knowledge Base"):
        with st.spinner("Processing knowledge base..."):
            vector_embeddings(query=custom_query)
            st.success("Knowledge base ready!")

# Main chat interface
query = st.text_input("Ask your question:")

if query:
    if "vectors" not in st.session_state:
        st.warning("Please initialize the knowledge base first!")
    else:
        with st.spinner("Searching for answer..."):
            # Create chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever(
                search_kwargs={"k": 4}
            )
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Get response with timing
            start = time.process_time()
            response = retrieval_chain.invoke({'input': query})
            process_time = time.process_time() - start
            
            # Display response
            st.write(response['answer'])
            st.caption(f"Response time: {process_time:.2f} seconds")
            
            # Show sources
            with st.expander("Sources"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    if "source" in doc.metadata:
                        st.caption(f"Source: {doc.metadata['source']}")
                    st.divider()