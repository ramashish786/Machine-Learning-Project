import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


load_dotenv()

@st.cache_resource
def load_vectorstore():
    loader = TextLoader("../inputs/sample_text.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=25
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever( 
                                    search_type="similarity",
                                    search_kwargs={"k": 4})


retriever = load_vectorstore()


prompt_template = ChatPromptTemplate.from_messages(
    [(
        "system",
     "You are a helpful AI assistant.\n"
     "Answer ONLY using the provided context.\n"
     "If the answer is not in the context, say: 'I don't know.'\n\n"
     "Context:\n{context}"
    ),
    (
        MessagesPlaceholder(variable_name="chat_history")
    ),
    (
      "human","{input}"
    )]
)

llm = ChatOpenAI(model="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))

chain =  prompt_template | llm

def get_response(input_text,chat_history):
    matching_docs = retriever.invoke(input_text)
    st.write(matching_docs)
    context = "\n".join(doc.page_content for doc in matching_docs)

    response = chain.invoke({"input":input_text,"context":context,"chat_history":chat_history})
    return response.content

chat_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("Chatbot")

user_input = st.text_input("You:")


if st.button("Submit") and user_input:
    response = get_response(user_input, st.session_state.chat_history)

    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )
    st.session_state.chat_history.append(
        AIMessage(content=response)
    )
    st.write(response)