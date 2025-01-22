import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

def process_pdf(file_path):
    """Process PDF with fallback strategies"""
    try:
        # Try different loaders with fallback
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except:
            loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
            documents = loader.load()
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    st.title("📄 Chat with Your PDF")
    
    # Sidebar for API key and PDF upload
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and not st.session_state.vector_store:
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process PDF
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.vector_store = process_pdf("temp.pdf")
            
            if st.session_state.vector_store:
                st.success("PDF processed successfully!")
            else:
                st.error("Failed to process PDF")

    # Chat interface
    if st.session_state.vector_store:
        # Initialize conversation chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.1),
            st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
        
        # Display chat history
        for query, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                st.write(answer)
        
        # Handle new query
        query = st.chat_input("Ask a question about the PDF:")
        if query:
            # Add user question to history
            st.session_state.chat_history.append((query, ""))
            
            try:
                # Get answer
                result = qa({"question": query})
                answer = result["answer"]
                
                # Update chat history
                st.session_state.chat_history[-1] = (query, answer)
                
                # Rerun to update display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    else:
        st.info("Please upload a PDF and enter your OpenAI API key to begin")

if __name__ == "__main__":
    main()