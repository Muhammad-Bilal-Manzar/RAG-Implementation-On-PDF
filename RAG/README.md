# PDF Q&A Web Application

This is a **Streamlit** web application that allows users to upload a PDF document and ask questions about its content. The application utilizes **LangChain**, **Groq API**, and **Retrieval-Augmented Generation (RAG)** to process and extract information from the uploaded PDF and respond with contextually relevant answers.

### Features
- Upload a PDF document.
- Automatically processes the PDF, splits the text into chunks, and embeds the content.
- Uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant content and generate context-aware responses.
- Allows users to ask questions related to the uploaded document.
- Displays answers based on the document's content along with source references.

### Model Used
The application uses the **Llama3 8B 8192** model for generating answers based on the content of the uploaded PDF. This model is powered by the **Groq API**.

## Requirements

To run the application, you'll need to install the necessary dependencies. This can be done by installing the libraries listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
