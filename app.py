# IMPORTS
import streamlit as st
import chromadb
import ollama
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract


# CONFIG
CHROMA_PATH = "./chroma_db"
COLLECTION = "entrepreneur_schemes"
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "phi3:mini"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
N_RESULTS = 5
DIST_CUTOFF = 0.75

# PAGE CONFIG
st.set_page_config(
    page_title="SchemeFinder",
    page_icon="logo.jpeg",
    layout="wide"
)


# UI THEME
st.markdown("""
<style>
.stApp { background-color: #EAF4F4; font-family: "Inter", sans-serif; }

section[data-testid="stSidebar"] { background-color: #0F5C5C; }
section[data-testid="stSidebar"] * { color: white !important; }

[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 2px dashed #0F5C5C;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div {
    color: #000000 !important;
    font-weight: 600;
}

[data-testid="stFileUploader"] button {
    background-color: black !important;
    color: white !important;
    border-radius: 6px;
    font-weight: 600;
}

.profile-title {
    color: #064E3B;
    font-weight: 800;
    font-size: 22px;
}

.block-container {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
}

.stButton > button {
    background-color: #0F5C5C;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #064E3B;
}

.success-box {
    background-color: #E6F7F0;
    border-left: 6px solid #2BB673;
    padding: 20px;
    border-radius: 8px;
    color: #064E3B;
}
</style>
""", unsafe_allow_html=True)

# HEADER
col1, col2 = st.columns([1,5])
with col1:
    st.image("logo.jpeg", width=85)
with col2:
    st.markdown("Karnataka Entrepreneur Scheme Finder")
    st.caption("Find government schemes tailored for Karnataka entrepreneurs")

st.divider()


# CHROMA DB
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION)

collection = get_collection()

# FUNCTIONS
def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded_file)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text
        except:
            pass
        st.warning("Using OCR for scanned PDF...")
        images = convert_from_bytes(uploaded_file.read())
        return "".join(pytesseract.image_to_string(img) for img in images)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]

def ingest(file, collection):
    text = read_file(file)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        collection.upsert(
            documents=[chunk],
            embeddings=embed([chunk]),
            metadatas=[{"source": file.name}],
            ids=[f"{file.name}_{i}"],
        )
    return len(chunks)

def retrieve(query, collection):
    if collection.count() == 0:
        return []
    results = collection.query(
        query_embeddings=[embed([query])[0]],
        n_results=min(N_RESULTS, collection.count()),
        include=["documents","distances"],
    )
    return [d for d,dist in zip(results["documents"][0], results["distances"][0]) if dist < DIST_CUTOFF]


def build_prompt(profile, chunks):

    context = "\n\n".join(chunks)

    return f"""
You are a  Government Schemes Advisor.

Use ONLY the provided document context.
Do NOT guess or invent details.

If scheme information is missing, respond exactly:
"I do not have this information in the available scheme documents."

USER PROFILE:
{profile}

DOCUMENT CONTEXT:
{context}

Return the answer in a SIMPLE readable format:

• Scheme Name: <name>
Purpose: <what the scheme supports>
Eligibility: Explain who can apply and how the user's profile matches or does not match.
Benefits:
○ <benefit>
○ <benefit>
○ <benefit>

Website Link: <official link if present, otherwise write "Not available">

Keep it natural and professional.
Do not use rigid templates.
Do not add extra sections.
"""


# SIDEBAR
with st.sidebar:
    st.markdown("### 📂 Scheme Database")

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if files and st.button("Add Documents"):
        for file in files:
            with st.spinner(f"Ingesting {file.name}..."):
                count = ingest(file, collection)
            st.success(f"{file.name} → {count} chunks added")

    st.write(f"Total Chunks: {collection.count()}")


# PROFILE FORM
st.markdown('<div class="profile-title">📋 Entrepreneur Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70)
    gender = st.selectbox("Gender", ["Male","Female","Other"])
    sector = st.selectbox("Sector",
        ["Technology","Social Innovation","Product Development","Service Innovation"])

with col2:
    entrepreneur_type = st.selectbox("Entrepreneur Type",
        ["Startup","MSME","Student Entrepreneur","Women Entrepreneur"])
    state = st.selectbox("State", ["Karnataka","Other"])
    ready = st.checkbox("Ready to apply immediately?")


# SEARCH
if st.button("Find Schemes"):

    profile = f"""
State: {state}
Age: {age}
Gender: {gender}
Sector: {sector}
Type: {entrepreneur_type}
Ready: {ready}
"""

    chunks = retrieve(profile, collection)

    with st.spinner("Searching schemes..."):
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": build_prompt(profile, chunks)}],
            stream=True,
        )
        response = ""
        for chunk in stream:
            if "message" in chunk:
                response += chunk["message"]["content"]

    st.markdown("Matching Schemes")
    st.markdown(f'<div class="success-box">{response}</div>', unsafe_allow_html=True)