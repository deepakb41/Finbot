from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import tempfile
import logging
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Use a writable directory for ChromaDB inside the /tmp directory
CHROMA_PATH = os.path.join('/tmp', 'chroma')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to clear ChromaDB directory and create a new one
def clear_chroma_db():
    if os.path.exists(CHROMA_PATH):
        logger.info(f"Clearing ChromaDB directory: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.chmod(CHROMA_PATH, 0o777)  # Ensure the directory is writable
    logger.info(f"ChromaDB directory created and permissions set: {CHROMA_PATH}")

# Function to clear the UPLOAD_FOLDER
def clear_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        logger.info(f"Clearing upload folder: {UPLOAD_FOLDER}")
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload folder created: {UPLOAD_FOLDER}")

# Clear the database and upload folder when the app starts
clear_chroma_db()
clear_upload_folder()

# LangChain Configuration
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to initialize ChromaDB
def initialize_chroma(chunks=None):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_function = OpenAIEmbeddings(api_key=api_key)
        if chunks:
            db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
        else:
            db = Chroma(embedding_function=embedding_function, persist_directory=CHROMA_PATH)
        logger.info("ChromaDB initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Error initializing Chroma: {e}")
        raise

@app.route('/')
def index():
    try:
        clear_chroma_db()
        clear_upload_folder()
    except Exception as e:
        logger.error(f"Error during reset on index page load: {e}")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    clear_chroma_db()  # Clear the database whenever a new file is uploaded
    clear_upload_folder()  # Clear the upload folder whenever a new file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
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

# Function to process the uploaded PDF and create ChromaDB
def process_pdf(file_path):
    try:
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

        db = initialize_chroma(chunks=chunks)

        # Ensure the database file is writable
        for root, dirs, files in os.walk(CHROMA_PATH):
            for file in files:
                os.chmod(os.path.join(root, file), 0o666)
                logger.info(f"Set writable permission for file: {file}")

        return {"chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '').lower()

    try:
        db = initialize_chroma()
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        api_key = os.getenv("OPENAI_API_KEY")
        model = ChatOpenAI(api_key=api_key)
        response_text = model.predict(prompt)  # Use invoke instead of predict

        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return jsonify({"error": "Failed to process query"}), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        clear_chroma_db()
        clear_upload_folder()
        return jsonify({"status": "success", "message": "System reset successfully"})
    except Exception as e:
        logger.error(f"Error during reset: {e}")
        return jsonify({"error": "Failed to reset system"}), 500

if __name__ == '__main__':
    app.run(debug=True)
