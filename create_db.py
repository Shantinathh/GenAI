from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('genai_notes.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')
vectorstores = Chroma.from_documents(
    embedding=embedding_model,
    documents=chunks,
    persist_directory='Chroma-dbs'
)

