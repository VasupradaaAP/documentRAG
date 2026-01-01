"""Document ingestion & retrieval pipeline with FAISS and Sentence Transformers."""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 200
OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "faiss_index.bin"
METADATA_FILE = BASE_DIR / "metadata.json"

print("📦 Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✓ Embedding model loaded")

class DocumentStore:
    """Manages PDF ingestion, embeddings, and retrieval."""
    
    def __init__(self):
        self.index = None
        self.chunks = []
        self.is_loaded = False
        self._load_index()
    
    def pdf_to_vectors(self, pdf_path: str) -> bool:
        """Process PDF: extract text → chunk → embed → index."""
        try:
            print(f"\n📄 Reading PDF: {pdf_path}")
            
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                print(f"   Total pages: {total_pages}")
                
                page_texts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_texts.append({
                        'text': page_text,
                        'page_number': page_num + 1
                    })
                
                print(f"✓ Extracted text from {total_pages} pages")
            
            doc_name = Path(pdf_path).stem
            all_chunks = self._create_chunks(page_texts, doc_name)
            print(f"✓ Created {len(all_chunks)} chunks")
            
            embeddings = self._generate_embeddings(all_chunks)
            print(f"✓ Generated embeddings for {len(embeddings)} chunks")
            
            self._add_to_index(embeddings, all_chunks)
            print(f"✓ Added to FAISS index")
            
            self._save_index()
            print(f"✓ Saved to disk")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing PDF: {e}")
            return False
    
    def _create_chunks(self, page_texts: List[Dict], doc_name: str) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        all_chunks = []
        chunk_id = 0
        
        for page_data in page_texts:
            text = page_data['text']
            page_num = page_data['page_number']
            
            if not text.strip():
                continue
            
            char_chunk_size = CHUNK_SIZE * 4
            char_overlap = OVERLAP * 4
            chunks = self._split_with_overlap(text, char_chunk_size, char_overlap)
            
            for chunk_text in chunks:
                all_chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'document': doc_name,
                    'page': page_num
                })
                chunk_id += 1
        
        return all_chunks
    
    def _split_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text with sliding window."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            start += (chunk_size - overlap)
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Convert text chunks to embeddings."""
        print("   Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
    
    def _add_to_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Add embeddings to FAISS index."""
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            print(f"   Created FAISS index (dimension: {dimension})")
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.is_loaded = True
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, str(INDEX_FILE))
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            
            print(f"   Saved: {INDEX_FILE}")
            print(f"   Saved: {METADATA_FILE}")
        except Exception as e:
            print(f"✗ Error saving index: {e}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            if INDEX_FILE.exists() and METADATA_FILE.exists():
                self.index = faiss.read_index(str(INDEX_FILE))
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                self.is_loaded = True
                print(f"✓ Loaded existing index with {len(self.chunks)} chunks")
        except Exception as e:
            print(f"⚠ Could not load existing index: {e}")
            self.is_loaded = False
    
    def retrieve(self, query: str, top_k: int = 3) -> Dict:
        """Find most relevant chunks for query."""
        if not self.is_loaded or self.index is None:
            return {
                'success': False,
                'chunks': [],
                'message': 'No documents indexed. Please ingest a PDF first.'
            }
        
        try:
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype(np.float32)
            distances, indices = self.index.search(query_embedding, top_k)
            
            retrieved_chunks = []
            for idx in indices[0]:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    retrieved_chunks.append({
                        'text': chunk['text'],
                        'document': chunk['document'],
                        'page': chunk['page'],
                        'chunk_id': chunk['id']
                    })
            
            return {'success': True, 'chunks': retrieved_chunks}
            
        except Exception as e:
            print(f"✗ Retrieval error: {e}")
            return {
                'success': False,
                'chunks': [],
                'message': f'Error during retrieval: {str(e)}'
            }
    
    def format_citations(self, chunks: List[Dict]) -> str:
        """Format citations from chunks."""
        if not chunks:
            return ""
        
        citations = []
        for i, chunk in enumerate(chunks, 1):
            doc = chunk.get('document', 'Unknown')
            page = chunk.get('page', 'N/A')
            citations.append(f"C{i}. {doc} (page {page})")
        
        return "\n".join(citations)
        


def create_document_store() -> DocumentStore:
    """Create and return DocumentStore instance."""
    return DocumentStore()
