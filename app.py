import os
import subprocess
from dataclasses import dataclass
from typing import Any, List, Optional

import chromadb
import chromadb.api
import pandas as pd
import streamlit as st
from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass
class AppState:
    """Class to manage application state."""

    chroma_path: Optional[str]
    chroma_client: Optional[chromadb.api.ClientAPI]
    current_collection: Optional[chromadb.Collection]
    query_engine: Optional[Any] = None

    def reset_query_engine(self) -> None:
        """Reset the query engine."""
        self.query_engine = None


class ModelManager:
    """Class to handle model-related operations."""

    @staticmethod
    def get_ollama_models() -> List[str]:
        """Get list of available Ollama models."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return [line.split()[0] for line in result.stdout.strip().split("\n")[1:]]
        except Exception:
            return []

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_llm(llm_name: str) -> Ollama:
        """Load Ollama model."""
        return Ollama(model=llm_name, request_timeout=120)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embed_model(embed_name: str) -> HuggingFaceEmbedding:
        """Load embedding model."""
        return HuggingFaceEmbedding(model_name=embed_name)


class ChromaDBManager:
    """Class to handle ChromaDB operations."""

    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def setup_ephemeral_connection(self) -> None:
        """Set up Ephemeral ChromaDB connection."""
        self.app_state.chroma_client = chromadb.EphemeralClient()
        self.setup_collection_selection()

    def setup_persistent_connection(self, chroma_path: str) -> None:
        """Set up Persistent ChromaDB connection."""
        if not os.path.exists(chroma_path):
            st.warning("ChromaDB path does not exist.")
            if st.button("Create ChromaDB"):
                self.app_state.chroma_client = chromadb.PersistentClient(
                    path=chroma_path
                )
                self.setup_collection_selection()
        else:
            try:
                self.app_state.chroma_client = chromadb.PersistentClient(
                    path=chroma_path
                )
                self.setup_collection_selection()
            except Exception as e:
                st.error(f"Error connecting to ChromaDB: {str(e)}")

    def setup_collection_selection(self) -> None:
        collections = self.list_collections()
        collection_name = st.selectbox(
            "Select Collection",
            collections,
            index=None,
            placeholder="",
        )
        self.handle_collection_selection(collection_name)

    def list_collections(self) -> List[str]:
        """Get list of existing collections."""
        if self.app_state.chroma_client:
            return sorted(
                [col.name for col in self.app_state.chroma_client.list_collections()]
            )
        return []

    def handle_collection_selection(self, collection_name: str) -> None:
        """Handle collection selection or creation."""
        # Clear all caches when changing collections
        for key in [
            k for k in st.session_state.keys() if k.startswith("cache_resource")
        ]:
            del st.session_state[key]
        self.app_state.reset_query_engine()

        if collection_name is None:
            self.app_state.current_collection = None
            self._handle_new_collection()
        else:
            self._select_existing_collection(collection_name)

    def _handle_new_collection(self) -> None:
        """Handle creation of new collection."""
        new_collection_name = st.text_input("New Collection Name", placeholder="")
        if st.button("Create Collection") and new_collection_name:
            try:
                self.app_state.current_collection = (
                    self.app_state.chroma_client.create_collection(
                        name=new_collection_name,
                        metadata={"description": "Document collection"},
                    )
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error creating collection: {str(e)}")

    def _select_existing_collection(self, collection_name: str) -> None:
        """Select an existing collection."""
        try:
            self.app_state.current_collection = (
                self.app_state.chroma_client.get_collection(name=collection_name)
            )
        except Exception as e:
            st.error(f"Error getting collection: {str(e)}")


class DocumentManager:
    """Class to handle document operations."""

    @staticmethod
    def _create_index(
        collection: chromadb.Collection, embed_model: HuggingFaceEmbedding
    ) -> VectorStoreIndex:
        """Create a new index for the collection."""
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    @staticmethod
    def get_index(
        collection: chromadb.Collection, embed_model: HuggingFaceEmbedding
    ) -> VectorStoreIndex:
        """Get index for the collection."""
        # Use collection name as cache key
        cache_key = f"index_{collection.name}"

        if cache_key not in st.session_state:
            st.session_state[cache_key] = DocumentManager._create_index(
                collection, embed_model
            )

        return st.session_state[cache_key]

    @staticmethod
    def display_collection_contents(collection: chromadb.Collection) -> None:
        """Display collection contents in a DataFrame."""
        try:
            docs = collection.get()
            if not docs["ids"]:
                st.info("Collection is empty. Please add some documents.")
                return

            data = {
                "ID": docs["ids"],
                "Document": docs["documents"],
                "Metadata": docs["metadatas"],
            }
            df = pd.DataFrame(data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Error displaying collection contents: {e}")

    @staticmethod
    def load_and_insert_documents(
        directory_path: str,
        collection: chromadb.Collection,
        embed_model: HuggingFaceEmbedding,
    ) -> None:
        """Load documents from directory and insert into collection."""
        try:
            if not os.path.exists(directory_path):
                st.error("Directory does not exist!")
                return

            file_reader = SimpleDirectoryReader(directory_path, recursive=True)

            with st.spinner("Loading documents..."):
                documents = file_reader.load_data()
                if not documents:
                    st.warning("No documents found in the specified directory.")
                    return

                # Create new index and insert documents
                index = DocumentManager._create_index(collection, embed_model)
                for doc in documents:
                    index.insert(doc)

                # Update cached index
                cache_key = f"index_{collection.name}"
                st.session_state[cache_key] = index
                st.session_state.app_state.reset_query_engine()
                st.rerun()
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")


class Interface:
    """Class to handle interface operations."""

    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def display_query_interface(self, llm: Ollama, index: VectorStoreIndex) -> None:
        """Display chat-like query interface."""
        # Check if collection is empty
        try:
            collection_empty = self.app_state.current_collection.count() == 0
            if collection_empty:
                st.warning(
                    "⚠️ The collection is empty. Please add some documents before querying."
                )
                return
        except Exception as e:
            st.error(f"Error checking collection: {str(e)}")
            return

        # Only show chat interface if collection has documents
        # Initialize query engine if needed
        if self.app_state.query_engine is None:
            # custome prompt template
            template = (
                "Imagine you are an advanced AI expert in cyber security laws, with access to all current and relevant legal documents, "
                "case studies, and expert analyses. Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
                "Here is some context related to the query:\n"
                "-----------------------------------------\n"
                "{context_str}\n"
                "-----------------------------------------\n"
                "Considering the above information, please respond to the following inquiry with detailed references to applicable laws, "
                "precedents, or principles where appropriate:\n\n"
                "Question: {query_str}\n\n"
                "Answer succinctly, starting with the phrase 'According to cyber security law,' and ensure your response is understandable to someone without a legal background."
            )
            qa_template = PromptTemplate(template)

            self.app_state.query_engine = index.as_query_engine(
                llm=llm,
                similarity_top_k=3,
                text_qa_template=qa_template,
                streaming=True,
            )

        # Chat input
        prompt = st.chat_input("Ask a question...")

        if prompt:
            try:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(prompt)

                with st.chat_message("assistant", avatar="🦙"):
                    with st.spinner("Thinking..."):
                        response = self.app_state.query_engine.query(prompt)
                        st.write_stream(response.response_gen)

                        sources = response.get_formatted_sources()
                        if sources:
                            st.divider()
                            st.caption("Sources:")
                            st.write(sources)
            except Exception as e:
                st.error(f"Error processing query: {e}")
                self.app_state.query_engine = None


def init_app_state() -> AppState:
    """Initialize application state."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState(
            chroma_path=None,
            chroma_client=None,
            current_collection=None,
        )
    return st.session_state.app_state


def main() -> None:
    """Main application function."""
    st.set_page_config(
        page_title="My RAG Chat",
        page_icon="🦙",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title("Chat With Your Docs 💬")

    app_state = init_app_state()
    model_manager = ModelManager()
    chroma_manager = ChromaDBManager(app_state)
    doc_manager = DocumentManager()
    interface = Interface(app_state)

    # Create two columns: sidebar and main area
    with st.sidebar:
        st.header("Configuration")

        # Model Selection
        with st.expander("Model Settings", expanded=True):
            # Track previous model settings
            prev_llm = getattr(st.session_state, "prev_llm", None)
            prev_embed = getattr(st.session_state, "prev_embed", None)

            llm_name = st.selectbox(
                "Ollama Model", options=model_manager.get_ollama_models()
            )
            llm = model_manager.load_llm(llm_name)

            embed_name = st.text_input("Embedding Model", "BAAI/bge-base-en-v1.5")
            try:
                embed_model = model_manager.load_embed_model(embed_name)
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return

            # Reset query engine if models change
            if prev_llm != llm_name or prev_embed != embed_name:
                app_state.reset_query_engine()
                st.session_state.prev_llm = llm_name
                st.session_state.prev_embed = embed_name

        # Database Configuration
        with st.expander("Database Settings", expanded=True):
            collection_type = st.radio("Collection Type", ["Ephemeral", "Permanent"])
            if collection_type == "Ephemeral":
                chroma_manager.setup_ephemeral_connection()
            else:
                chroma_path = st.text_input(
                    "ChromaDB Path", value="", placeholder="./chroma"
                )
                chroma_manager.setup_persistent_connection(chroma_path)

    # Main area
    if app_state.current_collection:
        # Create tabs for different functionalities
        query_tab, data_tab = st.tabs(["Query", "Data"])

        # Initialize index and query engine
        index = doc_manager.get_index(app_state.current_collection, embed_model)

        with query_tab:
            st.header("Query Documents")
            interface.display_query_interface(llm, index)

        with data_tab:
            st.header("Collection Data")
            directory_path = st.text_input(
                "Enter directory path containing documents:", key="doc_path"
            ).strip()

            if directory_path and os.path.isdir(directory_path):
                if st.button("Load Documents"):
                    doc_manager.load_and_insert_documents(
                        directory_path, app_state.current_collection, embed_model
                    )
            else:
                st.button("Load Documents", disabled=True)
                if directory_path:
                    st.error("Invalid directory path")

            st.subheader("Collection Contents")
            doc_manager.display_collection_contents(app_state.current_collection)
    else:
        st.info(
            "Please select or create a collection in the sidebar to start working with documents."
        )


if __name__ == "__main__":
    main()
