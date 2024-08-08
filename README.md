# Finbot - AI-Powered Financial Document Analysis Chatbot

Finbot is a tool for analysing and extracting information from complex financial documents. By leveraging LangChain, OpenAI embeddings, and Faiss indexing, Finbot provides users with a robust solution for querying financial data. The integration of Retrieval-Augmented Generation ensures that the answers are both relevant and contextually accurate, making Finbot an invaluable asset for financial analysts, investors, and other stakeholders.

Experience Finbot in action: [Try the Deployed Chatbot](https://web-production-318a.up.railway.app/)

## Key Features
- **AI-Driven Chatbot**: Utilises OpenAI's models to understand and generate human language.
- **LangChain**: Integrated for managing prompts, handling document loading, and embedding text.
- **Retrieval-Augmented Generation (RAG)**: Implements RAG to ensure that the responses are contextually relevant and grounded in the source documents.
- **Faiss Index**: Uses Faiss for efficient similarity search and clustering of dense vectors.
- **User-Friendly Interface**: Provides a seamless user experience with real-time feedback.

## Components

### Flask Web Application
The backend of Finbot is built using Flask, providing the essential tools to handle HTTP requests, manage file uploads, and serve the web interface.

### LangChain
A framework designed to build applications that understand and generate human language, providing tools for managing prompts and integrating with various embeddings and language models.

### OpenAI Embeddings
Converts textual data into high-dimensional vectors that capture semantic meaning and context, crucial for accurately addressing user queries based on the document content.

### Faiss Index
A library for efficient similarity search and clustering of dense vectors, enabling rapid and precise retrieval of relevant document chunks in response to user queries.

## Workflow

1. **File Upload**: Users upload 10-K or 10-Q financial documents in PDF format through the web interface.
2. **Document Processing**: The PDF is processed to extract and split the document into manageable chunks.
3. **Embedding Generation**: The text chunks are converted into embeddings using OpenAI's models.
4. **Faiss Indexing**: The embeddings are stored in a Faiss index for efficient similarity search.
5. **Query Handling**: Generates an embedding for the user query and retrieves the most relevant document chunks.
6. **Answer Generation**: Uses the retrieved chunks to generate a contextually relevant answer.

## Deployment

The application is deployed on Railway. Railway provides a simple way to deploy, manage, and scale web applications with minimal configuration. The necessary configuration files (`Procfile`, `runtime.txt`) and environment variables are set up in the Railway project settings.

- [Deployed Chatbot](https://web-production-318a.up.railway.app/)

## Attribution

The chatbot icon used in this project is sourced from [Flaticon](https://www.flaticon.com/free-icons/chatbot)




