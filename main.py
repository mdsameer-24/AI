# main.py — copy/paste ready for Option A (LangChain + langchain-google-genai + Streamlit)
import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

# LangGraph / LangChain components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# Messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Tools
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Document loading & splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vectorstore
# use sentence-transformers locally for stable embeddings (no cloud usage)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google LLM integration
# This import expects `langchain-google-genai` to be installed and compatible with langchain
from langchain_google_genai import ChatGoogleGenerativeAI

# === Load env variables ===
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Allow app to run in dev without Google key (but RAG LLM calls will fail)
    st.warning("GOOGLE_API_KEY not set. Add it to .env to call Google Gemini models.", icon="⚠️")

# === Streamlit session state basics ===
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

class State(TypedDict):
    messages: Annotated[list, add_messages]

# persistent in-memory per-file DBs (kept in session_state)
if "per_file_dbs" not in st.session_state:
    st.session_state.per_file_dbs = {}
per_file_dbs = st.session_state.per_file_dbs

# Set up embeddings (local sentence-transformer) - stable & avoids extra Google embedding dependency
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Tool: search, simple math, and document_retrieval from uploaded PDFs
def document_retrieval(query: str) -> str:
    if not per_file_dbs:
        return "No documents indexed yet. Please upload PDFs first."
    all_results = []
    for db in per_file_dbs.values():
        try:
            results = db.similarity_search(query, k=5)
        except Exception:
            results = []
        if results:
            all_results.extend(results)
    return "\n\n".join([doc.page_content for doc in all_results]) if all_results else "No relevant info found."

tools = [
    Tool(name="duckduckgo_search", func=DuckDuckGoSearchRun().run, description="Search the web."),
    Tool(name="add", func=lambda a, b: a + b, description="Add two numbers."),
    Tool(name="multiply", func=lambda a, b: a * b, description="Multiply two numbers."),
    Tool(name="document_retrieval", func=document_retrieval, description="RAG from uploaded PDFs."),
]

# Initialize Google-backed LLM wrapper (LangChain integration)
# NOTE: modern langchain-google-genai uses api args like google_api_key=...
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY if GOOGLE_API_KEY else None,
)

# bind tools to the LLM (so tool-calling works)
llm_with_tools = llm.bind_tools(tools)

# Build a simple LangGraph flow: chatbot -> optionally call tools -> chatbot
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm_with_tools.invoke(state["messages"])]})
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Streamlit UI
if __name__ == "__main__":
    st.set_page_config(page_title="Sameer AI", layout="wide")
    st.title("Sameer AI — LangChain + Google Gemini (copy/paste ready)")

    st.sidebar.header("Upload PDFs (for RAG)")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # prepare session state items
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "user-001"
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = set()

    st.info("Ask me anything. Upload PDFs to let the bot answer from your documents (RAG).", icon="🤖")

    # Handle uploads: only re-index when file list changes
    if uploaded_files:
        current_uploaded = set(file.name for file in uploaded_files)
        if current_uploaded != st.session_state.last_uploaded_files:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            for uploaded_file in uploaded_files:
                # write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp_path = tmp.name
                    tmp.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(tmp_path)
                chunks = loader.load_and_split(splitter)
                # create FAISS index per file
                file_db = FAISS.from_documents(documents=chunks, embedding=embedding_fn)
                per_file_dbs[uploaded_file.name] = file_db
            st.session_state.last_uploaded_files = current_uploaded
            st.success("✅ PDFs uploaded & indexed")

            # compile graph once DBs exist (optional)
            st.session_state.graph = graph_builder.compile(checkpointer=st.session_state.checkpointer)

    if "graph" not in st.session_state:
        st.session_state.graph = graph_builder.compile(checkpointer=st.session_state.checkpointer)
    graph = st.session_state.graph

    # show chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {
            "thread_id": st.session_state.thread_id,
            "checkpoint": st.session_state.checkpointer
        }
        try:
            # invoke LangGraph with a single HumanMessage
            response = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            ai_message = next(
                (m.content.strip() for m in reversed(response["messages"]) if isinstance(m, (AIMessage, ToolMessage)) and getattr(m, "content", None)),
                None
            )
            if ai_message:
                st.session_state.chat_history.append(("assistant", ai_message))
                with st.chat_message("assistant"):
                    st.markdown(ai_message)
        except Exception as e:
            st.error(f"❌ Error: {e}")
