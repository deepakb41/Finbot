from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
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

# Use PostgreSQL database URL for ChromaDB
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///tmp/chroma.db')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

# Initialize SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def initialize_chroma(clear_db=False):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_function = OpenAIEmbeddings(api_key=api_key)
        
        if clear_db:
            # Clear the Chroma database
            with engine.connect() as conn:
                conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        
        return Chroma(persist_directory=DATABASE_URL, embedding_function=embedding_function)
    except Exception as e:
        logger.error(f"Error initializing Chroma: {e}")
        raise

@app.route('/')
def index():
    try:
        # Clear and reinitialize ChromaDB
        initialize_chroma(clear_db=True)
    except Exception as e:
        logger.error(f"Error during reset on index page load: {e}")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        # Save file to a temporary location
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                file_path = temp_file.name
            # Process the PDF and create the Chroma DB
            data = process_pdf(file_path)
            return jsonify({"status": "success", "message": "File uploaded successfully", "chunks": data["chunks"]})
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(pages)

        # Initialize ChromaDB
        api_key = os.getenv("OPENAI_API_KEY")
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(api_key=api_key), persist_directory=DATABASE_URL)
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
        response_text = model.predict(prompt)

        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return jsonify({"error": "Failed to process query"}), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        # Clear and reinitialize ChromaDB
        initialize_chroma(clear_db=True)
        return jsonify({"status": "success", "message": "System reset successfully"})
    except Exception as e:
        logger.error(f"Error during reset: {e}")
        return jsonify({"error": "Failed to reset system"}), 500

if __name__ == '__main__':
    app.run(debug=True)
