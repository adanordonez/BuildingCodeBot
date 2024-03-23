import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
import tempfile
import whisper
from pytube import YouTube
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore



load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create an instance of the ChatOpenAI model using the OpenAI API key and the "gpt-3.5-turbo" model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")


parser = StrOutputParser()

# Create a chain of the model and the output parser
chain = model | parser


template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = prompt | model | parser



with open("transcription.txt") as file:
    transcription = file.read()

transcription[:100]


loader = TextLoader("transcription.txt")
text_documents = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_splitter.split_documents(text_documents)[:5]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)


embeddings = OpenAIEmbeddings()

print(documents[0])
