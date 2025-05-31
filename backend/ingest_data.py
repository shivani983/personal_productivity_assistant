import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import gdown


# dwonload the env file from the gdrive
file_id = "1UPRDofiylmevla5f59wx4_bNLNbHSWZ8"
url = f"https://drive.google.com/uc?id={file_id}"

output = ".env" 
gdown.download(url,output,quiet = False)

# load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH","db")

if not GROQ_API_KEY:
    raise ValueError("OpenAI API Key not found in .env")

# load documents
documents = [
    Document(page_content = "Meeting notes: Discuss project X deliverables."),
    Document(page_content = "Reminder: Submit report by Friday. "),
    Document(page_content = "Upcoming event: tech conference next Wednesday."),
]


from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")


text_splitter = CharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

vector_db = Chroma.from_documents(docs,embedding = embeddings, persist_directory = CHROMA_DB_PATH)

print("Documents successfully indexed")
