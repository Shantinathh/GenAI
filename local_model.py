from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


data = PyPDFLoader('genai_notes.pdf')
docs = data.load()


template = ChatPromptTemplate.from_messages(
    [("system","You are an AI which summarizes text"),("human","{data}")]
)

prompt = template.format_messages(data=docs[0].page_content)

model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

response = model.invoke(prompt)

print(response.content)