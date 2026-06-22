from pathlib import Path
import config
from utils import clear_directory_contents
from rumor_database import build_rumor_database, database_summary

class DocumentManager:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

    def build_rumor_database(self, progress_callback=None):
        if progress_callback:
            progress_callback(0.1, desc="Resetting vector store")
        self.clear_all()

        if progress_callback:
            progress_callback(0.35, desc="Preparing labeled rumor knowledge base")
        result = build_rumor_database()

        md_path = Path(result["markdown_path"])
        if progress_callback:
            progress_callback(0.55, desc="Chunking rumor cases")
        parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(
            md_path,
            source_name=Path(result["csv_path"]).name,
        )

        if not child_chunks:
            raise ValueError("No child chunks were created from the rumor database.")

        if progress_callback:
            progress_callback(0.8, desc="Indexing rumor cases in Qdrant")
        self.rag_system.parent_store.save_many(parent_chunks)
        collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
        collection.add_documents(child_chunks)

        if progress_callback:
            progress_callback(1.0, desc="Rumor RAG database ready")
        return result, len(parent_chunks), len(child_chunks)
    
    def get_markdown_files(self):
        sources = self.rag_system.parent_store.list_sources()
        if sources:
            return sources
        return sorted(p.name for p in self.markdown_dir.glob("*.md"))

    def get_database_summary(self):
        indexed_sources = self.get_markdown_files()
        indexed_text = "\n".join(f"- {source}" for source in indexed_sources) if indexed_sources else "- Not indexed"
        return f"{database_summary()}\nIndexed sources:\n{indexed_text}"
    
    def clear_all(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)

        clear_directory_contents(self.markdown_dir)
        self.rag_system.parent_store.clear_store()

        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)
