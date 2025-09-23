import os 
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Configure logging using the setup_logger utility
logging.basicConfig(filename="logs/app.log",
    filemode="a", 
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from a .env file
load_dotenv()

#load books
folder = "data"
documents = []

for filename in os.listdir(folder):  
    filepath = os.path.join(folder, filename)  
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()
    documents.extend(docs)

# Preview
print(documents[0].metadata)
print(documents[0].page_content[:200])
print(len(documents))

logging.info("Books loaded successfully!")

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# The split_documents method applies the splitting logic.
doc_chunks = text_splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base=os.getenv("OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

def get_faiss_vector_store(documents, embedding_model, persist_directory="db/faiss_index"):
    """
    Create or load a FAISS vector store with persistence, similar to Chroma.
    """
    import os

    if os.path.exists(persist_directory):
        # Reload existing FAISS index
        return FAISS.load_local(
            persist_directory,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        # Create a new one and save it
        vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)
        vector_store.save_local(persist_directory)
        return vector_store

# Initialize FAISS with our texts and the embeddings model.exit
vector_store = get_faiss_vector_store(
    documents=doc_chunks,
    embedding_model=embeddings_model,
    persist_directory="db/faiss_index"
)

logging.info("Vector store from books created successfully!")

# Create maximal marginal relevance (MMR) retriever
mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})


# Initialize the Language Model (LLM) we want to use for answering
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    openai_api_base=os.getenv("OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

# Ask a question!

# Create your prompt

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant who answers questions strictly based on the provided context. "
     "If the context does not contain the answer, respond with 'I don't know.'"),
    ("user", "Context: {context}\n\nQuestion: {question}")
])

question = "What is the basic plot of The Shining?"

retriever = mmr_retriever

# Create the RetrievalQA chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Ask the question using the chain
# The .invoke() method runs the entire chain.
ret_response = chain.invoke(question)

# Print out the responses
print(f"\nQuery: {question}")
#print(f"Specific Answer: {ret_response['result']}")
print(f"Specific Answer: {ret_response}")
logging.info("Query processed successfully!")