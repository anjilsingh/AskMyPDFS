import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import uuid

# Set page config
st.set_page_config(page_title="Chat PDF", layout="wide")

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Set it in your .env file.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# Theme settings
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "🌞 Light Mode"):
    st.session_state.dark_mode = not st.session_state.dark_mode

light_theme = {"background": "#F4F4F4", "text": "#333333", "button": "#4CAF50"}
dark_theme = {"background": "#1E1E1E", "text": "#E0E0E0", "button": "#BB86FC"}
theme = dark_theme if st.session_state.dark_mode else light_theme

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {theme["background"]};
            color: {theme["text"]};
        }}
        .stButton>button {{
            background-color: {theme["button"]};
            color: white;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Store Q&A
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Store last question
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Create a unique session ID for this run
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Track if we're downloading
if "is_downloading" not in st.session_state:
    st.session_state.is_downloading = False

# Function to display Q&A history
def display_qa_history():
    if st.session_state.qa_history:
        st.subheader("Q&A History")
        for i, (q, a) in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"**Question {i}:** {q}")
            st.markdown(f"**Answer {i}:** {a}")
            st.markdown("---")

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Setup conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question using the provided context. If the answer is not available, say so.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Generate PDF with proper Q&A format using ReportLab Platypus
def generate_pdf(qa_history, filename="qa_history.pdf"):
    pdf_path = os.path.join("downloads", filename)
    os.makedirs("downloads", exist_ok=True)
    
    # Use ReportLab's SimpleDocTemplate for better text handling
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("<b>Q&A History</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add each Q&A pair with proper formatting
    for index, (question, answer) in enumerate(qa_history, start=1):
        # Format question with bold style
        q_para = Paragraph(f"<b>Question {index}:</b> {question}", styles["Normal"])
        story.append(q_para)
        story.append(Spacer(1, 6))
        
        # Format answer
        a_para = Paragraph(f"<b>Answer {index}:</b> {answer}", styles["Normal"])
        story.append(a_para)
        story.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(story)
    return pdf_path

# Download handler (separated from the main flow)
def handle_download():
    st.session_state.is_downloading = True
    if not st.session_state.qa_history:
        st.error("❌ No Q&A history available.")
        return
        
    pdf_file = generate_pdf(st.session_state.qa_history)
    with open(pdf_file, "rb") as f:
        download_data = f.read()
        
    # Create a download button with a unique key
    download_key = f"download_button_{st.session_state.session_id}"
    st.download_button(
        label="Download Q&A PDF",
        data=download_data,
        file_name="qa_history.pdf",
        mime="application/pdf",
        key=download_key
    )
    st.success("✅ PDF Generated Successfully!")

# Process user question - completely separate from download
def process_question(question):
    # Don't process empty questions
    if not question.strip():
        return
        
    # Don't check for duplicates during download
    if st.session_state.is_downloading:
        st.session_state.is_downloading = False
        return
        
    # Check for duplicate question
    if question.lower().strip() == st.session_state.last_question.lower().strip():
        st.warning("⚠️ You've just asked this question. Please try a different question.")
        return
        
    # Update last question
    st.session_state.last_question = question.lower().strip()
    
    # Process the question
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ FAISS Index Load Error: {e}")
        return
        
    docs = new_db.similarity_search(question, k=5)
    if not docs:
        st.error("⚠️ No relevant context found. Try reprocessing PDFs.")
        return
        
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    answer = response["output_text"]
    
    # Add to history
    st.session_state.qa_history.append((question, answer))
    st.write("Reply:", answer)

# Streamlit UI with completely separated functionality
def main():
    st.header("AskMyPDFs – Chat with Your Documents! 📚")
    
    # Main UI area
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Question input in its own container
        question_container = st.container()
        with question_container:
            # Only show warning in this container
            if "user_question" not in st.session_state:
                st.session_state.user_question = ""
                
            user_question = st.text_input("Upload PDFs & Ask Questions!", key="main_question_input")
            ask_button = st.button("Ask Question")
            
            if ask_button and user_question:
                process_question(user_question)
    
        # Display Q&A history in main area
        display_qa_history()
    
    # Sidebar for document management and downloads
    with st.sidebar:
        st.title("Menu")
        
        # PDF upload section
        upload_section = st.container()
        with upload_section:
            pdf_docs = st.file_uploader("📂 Upload PDFs", accept_multiple_files=True)
            
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("❌ No text extracted. Try another PDF.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ Processing Complete!")
        
        # Completely separate download section with its own button
        download_section = st.container()
        with download_section:
            if st.button("📥 Generate Q&A PDF", key="generate_pdf_button"):
                handle_download()

if __name__ == "__main__":
    main()