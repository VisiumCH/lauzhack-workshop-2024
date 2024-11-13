import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import chromadb
import chromadb.api
import pandas as pd
import streamlit as st
from llama_index.core import (
    Document,
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass
class AppState:
    """Class to manage application state."""

    current_collection: Optional[chromadb.Collection] = None
    index: Optional[VectorStoreIndex] = None
    query_engine: Optional[BaseQueryEngine] = None


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
        # TODO: load Ollama model
        pass

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embed_model(embed_name: str) -> HuggingFaceEmbedding:
        """Load embedding model."""
        # TODO: load embedding model
        pass


class ChromaDBManager:
    """Class to handle ChromaDB operations."""

    def setup_ephemeral_collection(
        self,
        collection_name: str = "default_collection",
    ) -> chromadb.Collection:
        """Set up Ephemeral ChromaDB connection."""
        # TODO: create ephemeral collection
        pass


class IndexManager:
    """Class to handle index-related operations."""

    def _load_documents(self, directory_path: str) -> List[Document]:
        # TODO: load documents from directory
        pass

    def _get_transformations(self) -> None:
        # TODO: get transformations
        pass

    def get_index_from_documents(
        self,
        directory_path: str,
        chroma_collection: chromadb.Collection,
        embed_model: HuggingFaceEmbedding,
    ) -> VectorStoreIndex:
        """Get index from documents in a directory."""
        # TODO: get index from documents
        pass


class QueryEngineManager:
    """Class to handle query engine operations."""

    def _get_prompt_template(self) -> PromptTemplate:
        """Get the prompt template for the query engine."""
        # TODO: get prompt template
        pass

    def get_query_engine(
        self,
        index: VectorStoreIndex,
        llm: Ollama,
        streaming: bool = True,
    ) -> BaseQueryEngine:
        """Setup the query engine."""
        # TODO: setup query engine
        pass


class Interface:
    """Class to handle interface operations."""

    @staticmethod
    def display_collection_contents(chroma_collection: chromadb.Collection) -> None:
        """Display collection contents in a DataFrame."""
        try:
            docs = chroma_collection.get()
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

    def display_query_interface(self, query_engine: BaseQueryEngine) -> None:
        """Display chat-like query interface."""

        prompt = st.chat_input("Ask a question...")

        if prompt:
            try:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(prompt)

                with st.chat_message("assistant", avatar="🦙"):
                    with st.spinner("Thinking..."):
                        response = query_engine.query(prompt)
                        st.write_stream(response.response_gen)

                        sources = response.get_formatted_sources()
                        if sources:
                            st.divider()
                            st.caption("Sources:")
                            st.write(sources)
            except Exception as e:
                st.error(f"Error processing query: {e}")


def init_app_state() -> AppState:
    """Initialize application state."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState(index=None, query_engine=None)
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
    chroma_manager = ChromaDBManager()
    index_manager = IndexManager()
    query_engine_manager = QueryEngineManager()

    interface = Interface()

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
                with st.spinner("Loading embedding model..."):
                    embed_model = model_manager.load_embed_model(embed_name)
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return

        # Document Upload Controls
        with st.expander("Data Settings", expanded=True):
            st.header("Document Indexing")
            directory_path = st.text_input(
                "Enter directory path:", key="doc_path"
            ).strip()

            if directory_path and os.path.isdir(directory_path):
                if st.button("Setup Index"):
                    with st.spinner("Setting up index..."):
                        if app_state.index:
                            app_state.index.vector_store.clear()
                        app_state.current_collection = (
                            chroma_manager.setup_ephemeral_collection()
                        )
                        app_state.index = index_manager.get_index_from_documents(
                            directory_path,
                            app_state.current_collection,
                            embed_model,
                        )
            else:
                st.button("Setup Index", disabled=True)
                if directory_path:
                    st.error("Invalid directory path")

    # Main area
    if app_state.index:
        # Create tabs for Query and Data
        query_tab, data_tab = st.tabs(["Query", "Data"])

        query_engine = query_engine_manager.get_query_engine(
            app_state.index, llm, streaming=True
        )

        with query_tab:
            interface.display_query_interface(query_engine)

        with data_tab:
            st.header("Collection Contents")
            interface.display_collection_contents(app_state.current_collection)
    else:
        st.info(
            "Please select or create an index in the sidebar to start working with documents."
        )


if __name__ == "__main__":
    main()
