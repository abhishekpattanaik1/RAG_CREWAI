import streamlit as st
import os
from dotenv import load_dotenv
from rag.ingest import build_vectorstore
from crew import run_query

load_dotenv()

st.set_page_config(page_title="Multi-Agent RAG Assistant", layout="wide")
st.title("📚 Multi-Agent RAG Assistant (CrewAI)")

with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file and st.button("Ingest PDF"):
        os.makedirs("./uploads", exist_ok=True)
        pdf_path = os.path.join("./uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Building vector store..."):
            build_vectorstore(pdf_path)
        st.success("PDF ingested and indexed!")

    st.divider()
    st.caption("Ask about the PDF, general web topics, or weather in any city.")

st.header("2. Ask a Question")
query = st.text_input("Your question", placeholder="e.g. Summarize section 2 of the PDF / What's the weather in Hyderabad? / Latest news on X")

if st.button("Run") and query:
    with st.spinner("Manager agent delegating to specialists..."):
        try:
            answer = run_query(query)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")