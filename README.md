# Document RAG Chatbot

This repository contains a Document Retrieval Augmented Generation (RAG) Chatbot built with Gradio and Qdrant, utilizing Google's Gemini Flash and Embedding model.

## Overview

The chatboterface allows you to upload documents in various formats(specified below) and ask questions based on the content of those documents. It processes and indexes documents using google text-embedding-004 model and stores them in qdrant vector store, and provides context-aware answers through an interactive chat interface.

## Features

- **Multi-format Document Support**: Handles CSV, PDF, Word, PowerPoint, Markdown, and text files.
- **Batch Processing**: Upload and process up to 5 files simultaneously.
- **Vector Store Integration**: Uses Qdrant for efficient similarity search.
- **Generative AI Integration**: Leverages Google's Generative AI for embeddings and answer generation.
- **Persistent Chat History**: Maintains conversation context for more accurate responses.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Lokesh-Chimakurthi/simple-document-rag.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd simple-document-rag
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   - Set your Google Generative AI API key:

     ```bash
     export GEMINI_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the Application**

   ```bash
   python simple_rag.py
   ```

2. **Access the Interface**

   - Open the provided local URL in your web browser.

3. **Upload Documents**

   - Click on "Upload Documents" and select up to 5 files.

4. **Ask Questions**

   - Type your question in the message box and press Enter.

5. **Clear Chat History**

   - Click on "Clear Chat History" to reset the conversation.

## File Structure

- `app.py`: Sets up the Gradio user interface and handles user interactions.
- `simple_rag.py`: Contains classes for document processing, embedding generation, vector storage, and answer generation.

## Requirements

- Python 3.7 or higher
- `gradio`
- `numpy`
- `PyPDF2`
- `python-docx`
- `python-pptx`
- `markdown`
- `qdrant-client`
- `google-generativeai`

## Supported File Types

- CSV (`.csv`)
- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- PowerPoint Presentations (`.pptx`, `.ppt`)
- Markdown Files (`.md`, `.markdown`)
- Text Files (`.txt`, `.text`)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Gradio](https://gradio.app/) for the user interface components.
- [Qdrant](https://qdrant.tech/) for vector storage solutions.
- [Google Generative AI](https://ai.google.dev/) for AI models and APIs.