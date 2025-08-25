from IPython import embed
from flask import Flask, request, jsonify, render_template
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os


app = Flask(__name__)


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalchatbot"

docsearch = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)

# check as docsearch to fetch relevent docs.
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

api_key = os.environ.get("OPENAI_API_KEY")
print(api_key)
llm = OpenAI(temperature=0.4, max_tokens=500)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", f"{input} Do not repeat the question in the reply."),
])

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form.get("msg")
    else:
        msg = request.args.get("msg")
    if not msg:
        return "No message received.", 400
    print(msg)
    response = rag_chain.invoke({"input": msg})
    print(response['answer'])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

