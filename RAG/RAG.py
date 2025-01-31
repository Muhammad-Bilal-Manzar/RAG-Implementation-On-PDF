import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

def process_pdf(pdf_file):
    """Process PDF and initialize RAG system"""
    try:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=320
        )
        texts = text_splitter.split_documents(pages)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        api_key = os.getenv("GROQ_API_KEY")  
        
        if not api_key:
            raise ValueError("Groq API key is missing. Please set it in the .env file.")
        
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama3-8b-8192",
            temperature=0.1
        )
        
        # Creating RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

def main():
    st.set_page_config(page_title="PDF Q&A ", page_icon="📄")
    st.title("📄 PDF Q&A ")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
    
    if pdf_file is not None and st.session_state.qa_chain is None:
        with st.spinner("Processing PDF..."):
            st.session_state.qa_chain = process_pdf(pdf_file)
            st.session_state.messages = []
            st.success("PDF processed successfully! You can now ask questions.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for doc in message["sources"]:
                        st.write(f"- Page {doc['page']} of {doc['source']}")
    
    if prompt := st.chat_input("Ask a question about the document"):
        if not st.session_state.qa_chain:
            st.warning("Please upload a PDF first!")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"]
                sources = [
                    {
                        "page": doc.metadata["page"],
                        "source": doc.metadata["source"]
                    } 
                    for doc in result["source_documents"]
                ]
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("View Sources"):
                        for doc in sources:
                            st.write(f"- Page {doc['page']} of {doc['source']}")
                        
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
