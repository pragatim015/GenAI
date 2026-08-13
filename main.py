import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import hashlib

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DocuMind",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ================================
       GLOBAL
    ================================= */

    .stApp {
        background-color: #0b0b0b;
        color: #f5f5f5;
    }

    .main {
        background-color: #0b0b0b;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ================================
       SIDEBAR
    ================================= */

    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #2a2a2a;
    }

    section[data-testid="stSidebar"] * {
        color: #eeeeee;
    }


    /* ================================
       HEADINGS
    ================================= */

    h1 {
        color: #ffffff !important;
        font-size: 42px !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #ffffff !important;
    }

    p {
        color: #b5b5b5;
    }


    /* ================================
       HEADER
    ================================= */

    .hero {
        padding: 25px 30px;
        border: 1px solid #292929;
        border-radius: 18px;
        background: linear-gradient(
            145deg,
            #151515,
            #0f0f0f
        );
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #9d9d9d;
        font-size: 16px;
    }


    /* ================================
       UPLOAD CARD
    ================================= */

    .upload-card {
        background-color: #141414;
        border: 1px solid #292929;
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
    }


    /* ================================
       FILE UPLOADER
    ================================= */

    [data-testid="stFileUploader"] {
        background-color: #151515;
        border: 1px dashed #444444;
        border-radius: 14px;
        padding: 10px;
    }

    [data-testid="stFileUploader"] section {
        background-color: transparent;
    }


    /* ================================
       CHAT
    ================================= */

    [data-testid="stChatMessage"] {
        background-color: #151515;
        border: 1px solid #292929;
        border-radius: 16px;
        padding: 8px 15px;
        margin-bottom: 12px;
    }


    /* User message */
    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background-color: #202020;
    }


    /* ================================
       CHAT INPUT
    ================================= */

    [data-testid="stChatInput"] {
        border: 1px solid #333333;
        border-radius: 16px;
        background-color: #151515;
    }

    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background-color: #151515 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
    }


    /* ================================
       BUTTONS
    ================================= */

    .stButton > button {
        background-color: #ffffff;
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #dcdcdc;
        color: #000000;
    }


    /* ================================
       INFO / SUCCESS
    ================================= */

    .stAlert {
        background-color: #171717;
        border: 1px solid #333333;
        color: #eeeeee;
        border-radius: 12px;
    }


    /* ================================
       DIVIDER
    ================================= */

    hr {
        border-color: #292929 !important;
    }


    /* ================================
       SOURCE CARD
    ================================= */

    .source-card {
        background-color: #111111;
        border: 1px solid #292929;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }

    .source-title {
        color: #ffffff;
        font-weight: 600;
    }

    .source-page {
        color: #777777;
        font-size: 13px;
    }


    /* ================================
       STATUS CARD
    ================================= */

    .status-card {
        background-color: #161616;
        border: 1px solid #292929;
        border-radius: 14px;
        padding: 18px;
        margin-top: 15px;
    }

    .status-title {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .status-text {
        color: #888888;
        font-size: 14px;
    }


    /* ================================
       METRICS
    ================================= */

    .metric-card {
        background-color: #151515;
        border: 1px solid #292929;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .metric-number {
        color: #ffffff;
        font-size: 25px;
        font-weight: 700;
    }

    .metric-label {
        color: #777777;
        font-size: 13px;
    }


    /* ================================
       REMOVE STREAMLIT BRANDING
    ================================= */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background-color: transparent !important;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = ChatMistralAI(
        model="mistral-small-2603",
        temperature=0
    )


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful document assistant.

Use ONLY the information provided in the context.

Do NOT use outside knowledge.

If the answer cannot be found in the context, respond exactly:

"I could not find the answer in the uploaded document."

Keep the answer clear and concise.
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


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:26px;
            font-weight:700;
            color:white;
            margin-bottom:5px;
        ">
            ◈ DocuMind
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:#777777;
            font-size:13px;
            margin-bottom:25px;
        ">
            AI-powered document assistant
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📄 Document")

    if st.session_state.file_name:

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">
                    {st.session_state.file_name}
                </div>

                <div class="status-text">
                    {st.session_state.page_count} pages
                    <br>
                    {st.session_state.chunk_count} chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-card">
                <div class="status-title">
                    No document
                </div>

                <div class="status-text">
                    Upload a PDF to begin.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### ⚙️ RAG Settings")

    st.caption("Retrieval method")

    st.code("MMR", language=None)

    st.caption("Retrieved chunks")

    st.code("4", language=None)

    st.divider()

    st.markdown(
        """
        <div style="
            color:#555555;
            font-size:12px;
            text-align:center;
            padding-top:20px;
        ">
            Powered by LangChain + Mistral
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            Document Intelligence
        </div>

        <div class="hero-subtitle">
            Upload a document, search its content,
            and get accurate answers using AI.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    """
    <div class="upload-card">

        <h3>📤 Upload your document</h3>

        <p>
            Supported format: PDF
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    current_file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    if st.session_state.file_hash != current_file_hash:

        st.session_state.file_hash = current_file_hash
        st.session_state.file_name = uploaded_file.name

        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.messages = []

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = temp_file.name

        try:

            # -------------------------------------------------
            # LOAD PDF
            # -------------------------------------------------

            with st.spinner("Reading document..."):

                loader = PyPDFLoader(pdf_path)

                documents = loader.load()

            st.session_state.page_count = len(
                documents
            )


            # -------------------------------------------------
            # SPLIT DOCUMENT
            # -------------------------------------------------

            with st.spinner(
                "Preparing document..."
            ):

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = text_splitter.split_documents(
                    documents
                )

            st.session_state.chunk_count = len(
                chunks
            )


            # -------------------------------------------------
            # EMBEDDINGS
            # -------------------------------------------------

            with st.spinner(
                "Creating embeddings..."
            ):

                embedding_model = get_embedding_model()


            # -------------------------------------------------
            # CHROMA
            # -------------------------------------------------

            with st.spinner(
                "Building knowledge base..."
            ):

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model
                )


            # -------------------------------------------------
            # RETRIEVER
            # -------------------------------------------------

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )


            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever

            st.success(
                "Document is ready to chat with."
            )

        finally:

            if os.path.exists(pdf_path):
                os.unlink(pdf_path)


# =========================================================
# DOCUMENT METRICS
# =========================================================

if st.session_state.retriever is not None:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {st.session_state.page_count}
                </div>

                <div class="metric-label">
                    PAGES
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {st.session_state.chunk_count}
                </div>

                <div class="metric-label">
                    CHUNKS
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-number">
                    MMR
                </div>

                <div class="metric-label">
                    RETRIEVAL
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# CHAT
# =========================================================

if st.session_state.retriever is not None:

    st.divider()

    st.markdown(
        "### 💬 Ask your document"
    )

    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # -----------------------------------------------------
    # QUESTION
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask anything about your document..."
    )


    if question:

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.write(question)


        # -------------------------------------------------
        # RETRIEVE
        # -------------------------------------------------

        with st.spinner(
            "Searching the document..."
        ):

            docs = st.session_state.retriever.invoke(
                question
            )


        # -------------------------------------------------
        # CONTEXT
        # -------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )


        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------

        final_prompt = prompt.invoke({
            "context": context,
            "question": question
        })


        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        with st.spinner(
            "Thinking..."
        ):

            response = st.session_state.llm.invoke(
                final_prompt
            )

        answer = response.content


        # -------------------------------------------------
        # ASSISTANT
        # -------------------------------------------------

        with st.chat_message("assistant"):

            st.write(answer)


        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        with st.expander(
            "📄 View sources"
        ):

            for i, doc in enumerate(docs):

                page = doc.metadata.get(
                    "page"
                )

                page_text = (
                    f"Page {page + 1}"
                    if page is not None
                    else "Page unavailable"
                )

                st.markdown(
                    f"""
                    <div class="source-card">

                        <div class="source-title">
                            Source {i + 1}
                        </div>

                        <div class="source-page">
                            {page_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    doc.page_content
                )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.markdown(
        """
        <div style="
            margin-top:50px;
            text-align:center;
            padding:45px;
            border:1px solid #252525;
            border-radius:18px;
            background:#111111;
        ">

            <div style="
                font-size:45px;
                margin-bottom:15px;
            ">
                ◈
            </div>

            <div style="
                color:#ffffff;
                font-size:22px;
                font-weight:600;
            ">
                Your document is waiting
            </div>

            <div style="
                color:#777777;
                margin-top:8px;
            ">
                Upload a PDF above to start
                asking questions.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )