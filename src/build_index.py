"""
build_index.py
--------------
Handles document ingestion, text splitting, and vector database (ChromaDB) indexing.
Supports incremental updates by tracking file hashes to avoid redundant processing.

Workflow:
1. Scan `data/transcripts/` for .txt, .pdf, and .json files.
2. Compare current file hashes against `index_metadata.json`.
3. Delete stale chunks for modified/deleted files.
4. Load, split, and embed new/modified documents.
5. Persist updated embeddings to ChromaDB.
"""

import os
import json
import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Resolve paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "transcripts"
DB_DIR = ROOT_DIR / "data" / "chroma_db"
METADATA_FILE = DB_DIR / "index_metadata.json"


def get_file_hash(filepath: Path) -> str:
    """
    Compute the MD5 hash of a file to detect content changes.

    Args:
        filepath (Path): Absolute path to the file.

    Returns:
        str: MD5 hex digest of the file content.
    """
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def load_metadata() -> dict:
    """
    Load the index metadata containing previously indexed file hashes.

    Returns:
        dict: A dictionary mapping relative file paths to their MD5 hashes.
              Returns an empty dict if the metadata file doesn't exist or is corrupted.
    """
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Metadata file corrupted. Rebuilding from scratch.")
    return {}


def save_metadata(metadata: dict):
    """
    Persist the current file hash state to index_metadata.json.

    Args:
        metadata (dict): Mapping of relative file paths to their current hashes.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_changed_files(stored_metadata: dict) -> tuple[list[str], list[str], list[str], dict]:
    """
    Scan the transcripts directory and identify new, modified, or deleted files.

    Args:
        stored_metadata (dict): The previously saved hashes from index_metadata.json.

    Returns:
        tuple: (new_files, modified_files, deleted_files, current_files_mapping)
               where paths are strings relative to ROOT_DIR.
    """
    current_files = {}
    new_files = []
    modified_files = []
    
    # 1. Scan for new and modified files
    if DATA_DIR.exists():
        for ext in ["*.txt", "*.pdf", "*.json"]:
            for filepath in DATA_DIR.rglob(ext):
                # We store relative paths as string keys to avoid absolute path issues
                rel_path = str(filepath.relative_to(ROOT_DIR))
                file_hash = get_file_hash(filepath)
                current_files[rel_path] = file_hash
                
                if rel_path not in stored_metadata:
                    new_files.append(rel_path)
                elif stored_metadata[rel_path] != file_hash:
                    modified_files.append(rel_path)

    # 2. Find deleted files
    deleted_files = [path for path in stored_metadata if path not in current_files]

    return new_files, modified_files, deleted_files, current_files


def load_documents_for_paths(rel_paths: list[str]) -> list:
    """
    Load document content from given relative paths using LangChain loaders.

    Args:
        rel_paths (list[str]): List of paths relative to ROOT_DIR.

    Returns:
        list: List of LangChain Document objects.
    """
    docs = []
    for rel_path in rel_paths:
        abs_path = ROOT_DIR / rel_path
        if not abs_path.exists():
            continue
            
        ext = abs_path.suffix.lower()
        try:
            if ext == '.txt':
                loader = TextLoader(str(abs_path))
            elif ext == '.pdf':
                loader = PyPDFLoader(str(abs_path))
            elif ext == '.json':
                loader = JSONLoader(str(abs_path), jq_schema='.', text_content=False)
            else:
                continue
                
            docs.extend(loader.load())
            logger.debug(f"Loaded: {abs_path.name}")
        except Exception as e:
            logger.error(f"Error loading {abs_path}: {e}")
            
    return docs


def build_index():
    """
    Orchestrate the incremental index build process. 
    Detects changes, removes stale data, and updates ChromaDB with new embeddings.
    """
    logger.info("Checking for file changes...")

    logger.info(f"Detected changes: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted.")
    
    # Initialize embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    
    # 1. Delete chunks for modified and deleted files from ChromaDB
    files_to_remove = modified_files + deleted_files
    if files_to_remove:
        logger.info(f"Removing stale chunks for {len(files_to_remove)} files...")
        for rel_path in files_to_remove:
            abs_path = str(ROOT_DIR / rel_path)
            try:
                # We must use Chroma's underlying collection to delete by metadata
                collection = vectorstore._collection
                collection.delete(where={"source": abs_path})
                logger.debug(f"Deleted chunks for: {rel_path}")
            except Exception as e:
                logger.warning(f"Failed to delete chunks for {rel_path}: {e}")

    # 2. Process and add new/modified files
    files_to_add = new_files + modified_files
    if files_to_add:
        logger.info(f"Loading {len(files_to_add)} new/modified documents...")
        docs = load_documents_for_paths(files_to_add)
        
        if docs:
            logger.info("Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            logger.info(f"Created {len(chunks)} chunks from {len(files_to_add)} files.")
            
            logger.info("Generating embeddings and adding to vector store...")
            vectorstore.add_documents(chunks)
            logger.info("Chunks successfully added.")
        else:
            logger.warning("No content could be loaded from the changed files.")

    # 3. Save new state
    save_metadata(current_files)
    logger.info("✅ Index update complete. Metadata saved.")


if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        logger.error(f"Failed to build index: {e}", exc_info=True)
        exit(1)