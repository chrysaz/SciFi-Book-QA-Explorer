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


class SciFiExplorer:
    """A class for exploring and answering questions about a collection of science fiction books using LLMs and vector search."""

    def __init__(self, data_folder="data", persist_directory="db/faiss_index"):
        """
        Initializes the SciFiExplorer.

        Args:
            data_folder (str): Path to the folder containing .txt book files.
            persist_directory (str): Directory to persist or load the FAISS vector store.
        """
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/app.log",
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        load_dotenv()

        logging.info(f"Loading documents from {data_folder}...")
        self.documents = self._load_documents(data_folder)
        logging.info(f"Loaded {len(self.documents)} documents from {data_folder}")

        logging.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        doc_chunks = text_splitter.split_documents(self.documents)
        logging.info(f"Split into {len(doc_chunks)} chunks.")

        logging.info("Creating embeddings model...")
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=os.getenv("OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        logging.info("Creating or loading FAISS vector store...")
        self.vector_store = self._get_faiss_vector_store(
            documents=doc_chunks,
            embedding_model=embeddings_model,
            persist_directory=persist_directory
        )
        logging.info("Vector store from books created successfully!")

        logging.info("Initializing retriever...")
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )

        logging.info("Initializing LLM (ChatOpenAI)...")
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            openai_api_base=os.getenv("OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant who answers questions strictly based on the provided context. "
            "If the context does not contain the answer, respond with 'I don't know.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _load_documents(self, folder):
        """
        Loads all .txt documents from a folder.

        Args:
            folder (str): Path to the folder containing .txt files.

        Returns:
            list: List of loaded document objects.
        """
        documents = []
        for filename in os.listdir(folder):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(folder, filename)
            logging.info(f"Loading file: {filepath}")
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            logging.info(f"Loaded {len(docs)} docs from {filename}")
            documents.extend(docs)
        return documents

    def _get_faiss_vector_store(self, documents, embedding_model, persist_directory):
        """
        Creates or loads a FAISS vector store from disk.

        Args:
            documents (list): List of document chunks to index.
            embedding_model: Embedding model to use for vectorization.
            persist_directory (str): Directory to persist/load the FAISS index.

        Returns:
            FAISS: The FAISS vector store object.
        """
        if os.path.exists(persist_directory):
            logging.info(f"Loading FAISS vector store from {persist_directory}")
            return FAISS.load_local(
                persist_directory,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            logging.info("Creating new FAISS vector store and saving to disk...")
            vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)
            vector_store.save_local(persist_directory)
            logging.info(f"Saved FAISS vector store to {persist_directory}")
            return vector_store

    def ask(self, question: str) -> str:
        """
        Asks a question and returns the model's answer based on the indexed book context.

        Args:
            question (str): The user's question about the books.

        Returns:
            str: The answer generated by the LLM, based on retrieved context.
        """
        logging.info(f"Received question: {question}")
        return self.chain.invoke(question)


if __name__ == "__main__":
    explorer = SciFiExplorer()

    while True:
        query = input("\nEnter your question about the books (or 'end' to quit): ")
        if query.strip().lower() == "end":
            print("Goodbye!")
            logging.info("Session ended by user.")
            break
        answer = explorer.ask(query)
        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
        logging.info(f"User query: {query}\nAnswer: {answer}")

