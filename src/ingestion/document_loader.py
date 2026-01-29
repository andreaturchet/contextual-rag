"""
Document Loader
---------------
Loads text documents from files.
"""

from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads text documents from files.

    Usage:
        loader = DocumentLoader()
        documents = loader.load_directory("./data/sample_docs")
    """

    def load_file(self, file_path: str) -> Dict:
        """Load a single text file."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "content": content,
            "source": str(path.absolute()),
            "metadata": {
                "filename": path.name,
                "size_bytes": path.stat().st_size
            }
        }

    def load_directory(self, directory_path: str) -> List[Dict]:
        """Load all .txt files from a directory."""
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        
        for file_path in dir_path.glob("*.txt"):
            try:
                doc = self.load_file(str(file_path))
                documents.append(doc)
                print(f"  Loaded: {file_path.name}")
            except Exception as e:
                print(f"  Error: {file_path.name}: {e}")

        print(f"Total: {len(documents)} documents")
        return documents


if __name__ == "__main__":
    print("Testing Document Loader")
    print("-" * 40)
    
    loader = DocumentLoader()
    docs = loader.load_directory("./data/sample_docs")

    for doc in docs[:2]:
        print(f"\n{doc['metadata']['filename']}: {len(doc['content'])} chars")
