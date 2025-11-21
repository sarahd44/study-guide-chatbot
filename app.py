import os

import streamlit as st
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

# 1. Load environment variables (gets OPENAI_API_KEY from .env)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Put it in your .env file.")

# 2. Build the vector store from your HTML file
@st.cache_resource
def build_vectorstore():
    # Safely read the HTML file with utf-8 and ignore bad characters
    with open("docs/study_guide.html", "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    # Use BeautifulSoup to extract text from the HTML
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")

    # Wrap the text in a LangChain Document
    docs = [Document(page_content=text, metadata={"source": "study_guide.html"})]

    # Split into smaller chunks so the model can handle them
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Turn chunks into embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Set up the LLM via LangChain
llm = ChatOpenAI(
    model="gpt-4o-mini",   # you can change the model name if needed
    temperature=0,
    api_key=openai_api_key,
)

def get_answer_and_sources(question: str):
    # 1. Retrieve relevant chunks from the vector store (new LangChain style)
    docs = retriever.invoke(question)

    # 2. Build a context string from those chunks
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Create a prompt that tells the model how to use the context
    prompt = (
        "You are helping a college student study using a course study guide.\n"
        "Use ONLY the following context to answer the question. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in a clear, concise way appropriate for a student."
    )

    # 4. Call the LLM
    response = llm.invoke(prompt)

    # 5. Extract the text content
    answer_text = getattr(response, "content", str(response))

    return answer_text, docs


# 4. Streamlit UI (the "face" of your chatbot)
st.title("Study Guide Q&A Chatbot")
st.write("Ask questions about the HTML study guide in docs/study_guide.html.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Thinking..."):
        answer, sources = get_answer_and_sources(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Where in the document did this come from?")
    for i, doc in enumerate(sources, start=1):
        st.markdown(f"**Source {i}:**")
        st.write(doc.page_content[:400] + "...")



