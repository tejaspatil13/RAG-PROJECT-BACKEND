import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from logger import logger

load_dotenv()


class Retriever:

    retriever = None

    def get_retriever(self):
        try:
            if Retriever.retriever is not None:
                logger.info("Using Existing Retriever...")
                return Retriever.retriever
            
            
            
            logger.info("Loading Embedding Model...")

            embedding = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )

            logger.info("Loading ChromaDB...")

            vector_db = Chroma(
                persist_directory="chroma_db",
                embedding_function=embedding
            )

            logger.info("Creating Retriever...")

            Retriever.retriever = vector_db.as_retriever(
                search_kwargs={"k": 3}
            )

            logger.info("Retriever Created Successfully")

            return Retriever.retriever

        except Exception as e:
            logger.error(f"Retriever Error: {e}")
            raise