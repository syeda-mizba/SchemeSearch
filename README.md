# SchemeSearch
Scheme Search is a AI chatbot which helps you discover Government Schemes For Entrepreneurs

# SchemeSearch, Karnataka Entrepreneur Scheme Finder
SchemeFinder is a simple web app that helps entrepreneurs in Karnataka find relevant government schemes based on their profile.

## How It Works
1. The application allows you to upload official government scheme PDFs through the sidebar. These documents form the knowledge base of the system.
2. When a PDF is uploaded, the app extracts the text using PyPDF. If the file is a scanned document, OCR is automatically applied using Tesseract to read the content.
3. The extracted text is split into smaller overlapping chunks so that information can be searched more accurately.
4. Each chunk is converted into vector embeddings using the Ollama embedding model and stored in a persistent ChromaDB collection.
5. When a user fills out their entrepreneur profile and clicks “Find Schemes,” the profile is converted into an embedding and matched against the stored document chunks.
6. The most relevant chunks are passed to a lightweight chat model, which generates a response strictly based on the retrieved document context.
7. If the requested information is not present in the uploaded documents, the system clearly states that it does not have that information instead of guessing.

## How to Operate the Application
1. Install the required dependencies: streamlit, chromadb, ollama, pypdf, pdf2image, and pytesseract. Make sure Tesseract OCR is installed on your system.
2. Install Ollama and pull the required models specified in the configuration, such as the embedding model and chat model.
3. Run the application using the command `streamlit run <your_script_name>.py`.
4. Open the app in your browser. In the sidebar, upload one or more government scheme PDFs and click “Add Documents” to ingest them into the database.
5. Fill in the Entrepreneur Profile form with details such as age, gender, sector, entrepreneur type, state, and readiness to apply.
6. Click “Find Schemes” to generate matching scheme recommendations based only on the uploaded documents.

## Project Goal
The goal of SchemeFinder is to provide a reliable and document-grounded tool that helps Karnataka entrepreneurs quickly identify relevant government schemes without misinformation or hallucinated details.

