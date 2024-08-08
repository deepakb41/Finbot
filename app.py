import numpy as np
import faiss
import os
import tempfile
import shutil
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.schema import AIMessage, Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

app = Flask(__name__)
CORS(app)

# Define paths
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt template for LangChain
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Faiss index
dimension = 768  # Assuming your embeddings are 768-dimensional
index = faiss.IndexFlatL2(dimension)
documents = []

def clear_upload_folder():
    """Clear upload folder and set permissions."""
    if os.path.exists(UPLOAD_FOLDER):
        logger.info(f"Clearing upload folder: {UPLOAD_FOLDER}")
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload folder created: {UPLOAD_FOLDER}")

def initialize_app():
    """Initialize the application by clearing directories."""
    clear_upload_folder()

@app.route('/')
def index():
    try:
        clear_upload_folder()
    except Exception as e:
        logger.error(f"Error during reset on index page load: {e}")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    clear_upload_folder()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        logger.info(f"Uploading file: {file.filename}")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            file_path = temp_file.name
        data = process_pdf(file_path)
        return jsonify({"status": "success", "message": "File uploaded successfully", "chunks": data["chunks"]})
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file deleted: {file_path}")

def process_pdf(file_path):
    """Process the uploaded PDF and create Faiss index."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    logger.info(f"PDF loaded and split into {len(pages)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Document split into {len(chunks)} chunks")

    # Generate embeddings for each chunk
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_function = OpenAIEmbeddings(api_key=api_key)
    
    # Convert text chunks into Documents
    documents_chunked = [Document(page_content=chunk.page_content) for chunk in chunks]
    embeddings = embedding_function.embed_documents(documents_chunked)

    # Convert embeddings to numpy array and add to Faiss index
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    documents.extend(chunks)
    
    return {"chunks": len(chunks)}

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '').lower()

    try:
        # Generate embedding for the query
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_function = OpenAIEmbeddings(api_key=api_key)
        query_embedding = embedding_function.embed_documents([Document(page_content=query_text)])[0].astype('float32').reshape(1, -1)

        # Search Faiss index
        D, I = index.search(query_embedding, k=3)
        results = [documents[i] for i in I[0]]

        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = ChatOpenAI(api_key=api_key)
        response = model.invoke(prompt)

        if isinstance(response, AIMessage):
            response_text = response.content
        else:
            response_text = response["choices"][0]["message"]["content"]

        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return jsonify({"error": "Failed to process query"}), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        clear_upload_folder()
        global index, documents
        index = faiss.IndexFlatL2(dimension)
        documents = []
        return jsonify({"status": "success", "message": "System reset successfully"})
    except Exception as e:
        logger.error(f"Error during reset: {e}")
        return jsonify({"error": "Failed to reset system"}), 500

if __name__ == '__main__':
    try:
        initialize_app()
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Error during app initialization: {e}")
