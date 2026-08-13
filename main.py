import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

st.title("📚 Document RAG Assistant")

# -----------------------------
# Upload PDF
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    # -----------------------------
    # Load PDF
    # -----------------------------

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.write(f"📄 Pages found: {len(documents)}")

    # -----------------------------
    # Split into chunks
    # -----------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    st.write(f"📝 Created {len(chunks)} chunks")

    # -----------------------------
    # Embeddings
    # -----------------------------

    embedding_model = HuggingFaceEmbeddings()

    # -----------------------------
    # Chroma
    # -----------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # -----------------------------
    # Retriever
    # -----------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # -----------------------------
    # Mistral
    # -----------------------------

    llm = ChatMistralAI(
        model="mistral-small-2603",
        temperature=0
    )

    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the provided context, say:

"I could not find the answer in the uploaded document."

Do not use outside knowledge.
"""
        ),
        (
            "human",
            """Context:

{context}

Question:

{question}
"""
        )
    ])

    st.divider()

    # -----------------------------
    # Ask question
    # -----------------------------

    question = st.text_input(
        "Ask a question about your document:"
    )

    if question:

        with st.spinner("Searching the document..."):

            # Retrieve relevant documents
            docs = retriever.invoke(question)

            # Combine retrieved text
            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

        with st.spinner("Generating answer..."):

            # Create prompt
            final_prompt = prompt.invoke({
                "context": context,
                "question": question
            })

            # Get answer
            response = llm.invoke(final_prompt)

        st.subheader("🤖 AI Answer")

        st.write(response.content)

        # -----------------------------
        # Sources
        # -----------------------------

        with st.expander("📄 View sources"):

            for i, doc in enumerate(docs):

                st.write(f"### Source {i + 1}")

                st.write(doc.page_content)

                page = doc.metadata.get("page")

                if page is not None:
                    st.caption(f"Page {page + 1}")

    # Delete temporary file
    os.unlink(pdf_path)