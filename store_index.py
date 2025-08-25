from src.helper import (load_pdf_file,
                        text_split,
                        download_hugging_face_embeddings)
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if PINECONE_API_KEY is not None:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Extract text data from pdf iile.
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Check if PINECONE_API_KEY is set
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError(
        "PINECONE_API_KEY environment variable not set. Please set it in your .env file or environment."
    )

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
index_name = "medicalchatbot"

if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=384, metric="cosine", 
                    spec=ServerlessSpec(cloud="aws",
                                        region="us-east-1"))


# Embed each chunk and upsert the embeddings into your pinecone index.
docsearch = PineconeVectorStore.from_documents(documents=text_chunks,
                                               embedding=embeddings,
                                               index_name=index_name)
