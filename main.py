
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# document loader
loader = TextLoader('notes.txt')
docs = loader.load()

# prompt tempaltes
tempalate = ChatPromptTemplate.from_messages(
    [("system","you are an AI that summarizes text"),("human","{data}")]
)

prompt = tempalate.format_messages(data=docs[0].page_content)



model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
)

response = model.invoke(prompt)

print(response.text)
