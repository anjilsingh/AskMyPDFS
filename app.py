import os
import uuid
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import google.generativeai as genai

st.set_page_config(page_title="AskMyPDFs - Generative AI RAG System", layout="wide")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Set it in your environment or .env file.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.button("Dark Mode" if not st.session_state.dark_mode else "Light Mode"):
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

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "is_downloading" not in st.session_state:
    st.session_state.is_downloading = False

def display_qa_history():
    if st.session_state.qa_history:
        st.subheader("Q&A History")
        for i, (q, a) in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"**Question {i}:** {q}")
            st.markdown(f"**Answer {i}:** {a}")
            st.markdown("---")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question accurately using only the provided context. If the answer is not available in the context, explicitly state that it is not available. Do not fabricate information.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def generate_pdf(qa_history, filename="qa_history.pdf"):
    pdf_path = os.path.join("downloads", filename)
    os.makedirs("downloads", exist_ok=True)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("<b>Q&A History Report</b>", styles["Title"]), Spacer(1, 12)]
    
    for index, (question, answer) in enumerate(qa_history, start=1):
        story.append(Paragraph(f"<b>Question {index}:</b> {question}", styles["Normal"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Answer {index}:</b> {answer}", styles["Normal"]))
        story.append(Spacer(1, 12))
        
    doc.build(story)
    return pdf_path

def handle_download():
    st.session_state.is_downloading = True
    if not st.session_state.qa_history:
        st.error("No Q&A history available to export.")
        return
        
    pdf_file = generate_pdf(st.session_state.qa_history)
    with open(pdf_file, "rb") as f:
        download_data = f.read()
        
    st.download_button(
        label="Download Q&A PDF",
        data=download_data,
        file_name="qa_history.pdf",
        mime="application/pdf",
        key=f"download_button_{st.session_state.session_id}"
    )
    st.success("PDF Generated Successfully!")

def process_question(question):
    if not question.strip():
        return
        
    if st.session_state.is_downloading:
        st.session_state.is_downloading = False
        return
        
    if question.lower().strip() == st.session_state.last_question.lower().strip():
        st.warning("You've just asked this question. Try a different query.")
        return
        
    st.session_state.last_question = question.lower().strip()
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"FAISS Index Load Error: {e}. Please process PDFs first.")
        return
        
    docs = new_db.similarity_search(question, k=5)
    if not docs:
        st.error("No relevant context found in documents.")
        return
        
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    answer = response["output_text"]
    
    st.session_state.qa_history.append((question, answer))
    st.write("Reply:", answer)

def main():
    st.header("AskMyPDFs - Generative AI RAG System")
    
    col1, col2 = st.columns([7, 3])
    
    with col1:
        user_question = st.text_input("Ask a question based on your uploaded documents:", key="main_question_input")
        if st.button("Ask Question") and user_question:
            process_question(user_question)
            
        display_qa_history()
    
    with st.sidebar:
        st.title("Document Manager")
        pdf_docs = st.file_uploader("Upload PDF Documents", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Executing data ingestion & vector embedding..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text extracted from PDFs.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Ingestion & Vector Indexing Complete!")
                    
        if st.button("Generate Q&A PDF Report", key="generate_pdf_button"):
            handle_download()

if __name__ == "__main__":
    main()
