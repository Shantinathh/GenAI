
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# document loader
loader = TextLoader('notes.txt')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# prompt tempaltes
tempalate = ChatPromptTemplate.from_messages(
    [("system","you are an AI that summarizes text"),("human","{data}")]
)

prompt = tempalate.format_messages(data=chunks)



model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
)

response = model.invoke(prompt)

print(response.text)
