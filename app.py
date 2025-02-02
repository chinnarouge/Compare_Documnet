import streamlit as st
import difflib
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Define modern dark theme with better contrast
theme = {
    "primary": "#00ADB5",  # Teal
    "secondary": "#FF2E63",  # Pink
    "background": "#1F1F1F",  # Dark gray
    "text": "#EEEEEE",  # Light gray
    "border": "#393E46",  # Gray
    "chat_background": "#2A2A2A",  # Slightly lighter dark gray
    "chat_text": "#FFFFFF",  # White
}

# Apply custom CSS for a professional UI
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {theme['background']};
        color: {theme['text']};
        font-family: 'Arial', sans-serif;
    }}
    .stButton>button, .stFileUploader>div>div>div>button {{
        background-color: {theme['primary']};
        color: white;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        padding: 12px 18px;
        margin-right: 10px;
        transition: 0.3s;
    }}
    .stButton>button:hover, .stFileUploader>div>div>div>button:hover {{
        background-color: {theme['secondary']};
    }}
    .stTextInput>div>div>input {{
        background-color: {theme['border']};
        color: {theme['text']};
        border: 2px solid {theme['primary']};
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
    }}
    .comparison-container {{
        display: flex;
        gap: 20px;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .comparison-box {{
        width: 45%;
        padding: 15px;
        border: 2px solid {theme['border']};
        border-radius: 10px;
        background: {theme['chat_background']};
        color: {theme['chat_text']};
    }}
    .chat-container {{
        background: {theme['chat_background']};
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
    }}
    .chat-container b {{
        color: {theme['primary']};
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🚀 Advanced RAG System with DeepSeek R1 & Ollama")

# Function to load and process PDF
def load_and_process_pdf(uploaded_file, temp_file_name):
    with open(temp_file_name, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PDFPlumberLoader(temp_file_name)
    docs = loader.load()
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    return text_splitter.split_documents(docs)

# Function to compare two documents and highlight differences
def compare_documents(docs1, docs2):
    text1 = " ".join([doc.page_content for doc in docs1])
    text2 = " ".join([doc.page_content for doc in docs2])
    differ = difflib.HtmlDiff()
    return differ.make_file(text1.splitlines(), text2.splitlines())

# Upload PDF files
col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("📄 Upload First PDF", type="pdf")
with col2:
    uploaded_file2 = st.file_uploader("📄 Upload Second PDF", type="pdf")

# Clear files button
if st.button("🆑 Clear Files"):
    uploaded_file1 = None
    uploaded_file2 = None
    st.session_state.chat_history = []
    st.success("Files cleared. Please upload new files.")

# Initialize documents and embeddings
documents1, documents2 = None, None
if uploaded_file1:
    documents1 = load_and_process_pdf(uploaded_file1, "temp1.pdf")
if uploaded_file2:
    documents2 = load_and_process_pdf(uploaded_file2, "temp2.pdf")

# Compare documents if both files are uploaded
if uploaded_file1 and uploaded_file2:
    st.subheader("📊 Side-by-Side Document Comparison")
    diff_html = compare_documents(documents1, documents2)
    st.components.v1.html(diff_html, height=500, scrolling=True)

# Initialize RAG system if at least one document is uploaded
if uploaded_file1 or uploaded_file2:
    # Combine documents if both are uploaded
    combined_documents = documents1 + documents2 if documents1 and documents2 else documents1 or documents2

    # Create embeddings and vector store
    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(combined_documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize Ollama LLM
    llm = Ollama(model="deepseek-r1:1.5b")

    # Define the prompt template
    qa_prompt = """
    Use the provided context to answer the question concisely. If the answer is unknown, say "I don't know". 
    If the question is unrelated, say "This question is unrelated to the document." 
    Context: {context}
    Question: {question}
    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_prompt)
    

    # Configure the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)

    # Configure the document chain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=PromptTemplate(
            input_variables=["page_content"],
            template="Context:\n{page_content}",
        ),
    )

    # Configure the RetrievalQA chain
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        verbose=True,
    )

    # Chat input and button to ask questions
    user_input = st.chat_input("💬 Ask a question about the uploaded documents:")
    if st.button("❓ Ask a Question"):
        user_input = st.text_input("Enter your question:")

    if user_input:
        with st.spinner("Generating response..."):
            response = qa.run(user_input)
            st.session_state.chat_history.append((user_input, response))

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for query, reply in st.session_state.chat_history:
            st.markdown(f"<div class='chat-container'><b>User:</b> {query}<br><b>AI:</b> {reply}</div>", unsafe_allow_html=True)