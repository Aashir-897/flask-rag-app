import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_cors import CORS 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# global vars
db = None
qa = None

@app.route("/")
def index():
    return jsonify({"message":  "PDF Upload and Q&A Service is running."})

@app.route("/upload", methods=["POST"])
def upload_file():
    global db, qa
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text received"}), 400

    from langchain.schema import Document
    documents = [Document(page_content=text)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    os.environ["OPENAI_API_KEY"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    # embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()

    # LLM
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
        openai_api_base="https://router.huggingface.co/v1",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return jsonify({"message": "File uploaded and processed successfully"})


@app.route("/ask", methods=["POST"])
def ask_question():
    global qa
    if qa is None:
        return jsonify({"error": "No PDF uploaded yet"}), 400

    data = request.get_json()
    print("Received data:", data)
    query = data.get("query", "")
    print("Received question:", query)
    if not query:
        return jsonify({"error": "Question is required"}), 400

    result = qa.invoke(query)
    if isinstance(result, dict):
        return jsonify({"answer": result.get("result", str(result))})
    else:
        return jsonify({"answer": str(result)})



if __name__ == "__main__":
    app.run(debug=True)
