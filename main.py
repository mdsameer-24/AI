# Replace entire previous main.py with this for in-memory Chroma

import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# === Load env variables ===
load_dotenv()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# === State ===
class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = []

# embedding_fn = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.environ["GOOGLE_API_KEY"]
# )
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if "per_file_dbs" not in st.session_state:
    st.session_state.per_file_dbs = {}
per_file_dbs = st.session_state.per_file_dbs

def document_retrieval(query: str) -> str:
    if not per_file_dbs:
        return "No documents indexed yet. Please upload PDFs first."
    all_results = []
    for db in per_file_dbs.values():
        results = db.similarity_search(query)
        if results:
            all_results.extend(results)
    return "\n\n".join([doc.page_content for doc in all_results]) if all_results else "No relevant info found."

tools.extend([
    Tool(name="duckduckgo_search", func=DuckDuckGoSearchRun().run, description="Search the web."),
    Tool(name="add", func=lambda a, b: a + b, description="Add two numbers."),
    Tool(name="multiply", func=lambda a, b: a * b, description="Multiply two numbers."),
    Tool(name="document_retrieval", func=document_retrieval, description="RAG from uploaded PDFs.")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.5,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm_with_tools.invoke(state["messages"])]})
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

if __name__ == "__main__":
    st.set_page_config(page_title="Sameer AI")
    st.title("Sameer AI")
    st.sidebar.header("Upload PDFs")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "user-001"
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = set()

    st.warning("Ask me anything or 📱 On mobile? Tap >> to upload PDFs if it's document-related!", icon="⚠️")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        current_uploaded = set(file.name for file in uploaded_files)
        if current_uploaded != st.session_state.last_uploaded_files:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            for uploaded_file in uploaded_files:
                path = f"temp_{uploaded_file.name}"
                with open(path, "wb") as f:
                    f.write(uploaded_file.read())
                loader = PyPDFLoader(path)
                chunks = loader.load_and_split(splitter)
                file_db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_fn
)
                file_db.save_local(f"faiss_indexes/{uploaded_file.name}")
                per_file_dbs[uploaded_file.name] = file_db
            st.session_state.last_uploaded_files = current_uploaded
            st.success("✅ PDFs uploaded")
            st.session_state.graph = graph_builder.compile(checkpointer=st.session_state.checkpointer)

    if "graph" not in st.session_state:
        st.session_state.graph = graph_builder.compile(checkpointer=st.session_state.checkpointer)

    graph = st.session_state.graph

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
            response = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            ai_message = next(
                (m.content.strip() for m in reversed(response["messages"]) if isinstance(m, (AIMessage, ToolMessage)) and m.content),
                None
            )
            if ai_message:
                st.session_state.chat_history.append(("assistant", ai_message))
                with st.chat_message("assistant"):
                    st.markdown(ai_message)
        except Exception as e:
            # st.error(f"❌ Error: {e}")
            st.error("❌ Oops! Something went wrong. Please try again shortly.")
