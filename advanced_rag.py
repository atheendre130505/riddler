import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import hashlib
from datetime import datetime
import requests
import pickle

# Try to import advanced libraries, fallback to simple ones
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available, using fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using fallback")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence Transformers not available, using fallback")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAG:
    """
    Advanced RAG system with multiple backends for maximum performance
    Supports ChromaDB, FAISS, and Hugging Face embeddings
    """
    
    def __init__(self, persist_directory: str = "./advanced_rag", use_chromadb: bool = True, use_faiss: bool = True):
        """
        Initialize advanced RAG system
        
        Args:
            persist_directory: Directory to persist data
            use_chromadb: Use ChromaDB for vector storage
            use_faiss: Use FAISS for fast similarity search
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # API keys
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
        
        # Initialize vector stores
        self.chroma_client = None
        self.chroma_collection = None
        self.faiss_index = None
        self.faiss_embeddings = None
        
        # Fallback storage
        self.documents = []
        self.document_metadata = []
        self.document_embeddings = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        
        # Initialize backends
        if use_chromadb and CHROMADB_AVAILABLE:
            self._init_chromadb()
        
        if use_faiss and FAISS_AVAILABLE:
            self._init_faiss()
        
        # Load existing data
        self._load_data()
        
        logger.info("Advanced RAG system initialized")
    
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=os.path.join(self.persist_directory, "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="content_embeddings",
                metadata={"description": "Advanced RAG content embeddings"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
            self.chroma_client = None
            self.chroma_collection = None
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            # Load existing FAISS index if available
            faiss_file = os.path.join(self.persist_directory, "faiss_index.bin")
            if os.path.exists(faiss_file):
                self.faiss_index = faiss.read_index(faiss_file)
                logger.info("Loaded existing FAISS index")
            else:
                # Create new index (dimension will be set when first embedding is added)
                self.faiss_index = None
                logger.info("FAISS ready for new index creation")
            
            # Load embeddings
            embeddings_file = os.path.join(self.persist_directory, "faiss_embeddings.pkl")
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    self.faiss_embeddings = pickle.load(f)
                logger.info("Loaded existing FAISS embeddings")
            
        except Exception as e:
            logger.warning(f"FAISS initialization failed: {e}")
            self.faiss_index = None
            self.faiss_embeddings = None
    
    def _load_data(self):
        """Load persisted data"""
        try:
            # Load documents
            docs_file = os.path.join(self.persist_directory, "documents.pkl")
            if os.path.exists(docs_file):
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
            
            # Load metadata
            meta_file = os.path.join(self.persist_directory, "metadata.pkl")
            if os.path.exists(meta_file):
                with open(meta_file, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            
            # Load embeddings
            emb_file = os.path.join(self.persist_directory, "embeddings.pkl")
            if os.path.exists(emb_file):
                with open(emb_file, 'rb') as f:
                    self.document_embeddings = pickle.load(f)
            
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.warning(f"Could not load persisted data: {e}")
    
    def _save_data(self):
        """Save data to disk"""
        try:
            # Save documents
            docs_file = os.path.join(self.persist_directory, "documents.pkl")
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save metadata
            meta_file = os.path.join(self.persist_directory, "metadata.pkl")
            with open(meta_file, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            # Save embeddings
            if self.document_embeddings is not None:
                emb_file = os.path.join(self.persist_directory, "embeddings.pkl")
                with open(emb_file, 'wb') as f:
                    pickle.dump(self.document_embeddings, f)
            
            # Save FAISS index
            if self.faiss_index is not None:
                faiss_file = os.path.join(self.persist_directory, "faiss_index.bin")
                faiss.write_index(self.faiss_index, faiss_file)
            
            # Save FAISS embeddings
            if self.faiss_embeddings is not None:
                embeddings_file = os.path.join(self.persist_directory, "faiss_embeddings.pkl")
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(self.faiss_embeddings, f)
            
        except Exception as e:
            logger.error(f"Could not save data: {e}")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding using the best available method
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        # Try sentence transformer first
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode([text])[0]
            except Exception as e:
                logger.warning(f"Sentence transformer embedding failed: {e}")
        
        # Try Hugging Face API
        if self.hf_api_key:
            hf_emb = self._get_hf_embedding(text)
            if hf_emb is not None:
                return hf_emb
        
        # Fallback to TF-IDF
        try:
            if not hasattr(self, '_tfidf_fitted'):
                self.vectorizer.fit([text])
                self._tfidf_fitted = True
            return self.vectorizer.transform([text]).toarray()[0]
        except Exception as e:
            logger.error(f"All embedding methods failed: {e}")
            return None
    
    def _get_hf_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Hugging Face API"""
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            payload = {
                "inputs": text,
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return np.array(data[0])
            
            return None
            
        except Exception as e:
            logger.warning(f"HF embedding failed: {e}")
            return None
    
    def _chunk_content(self, content: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split content into overlapping chunks"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None, source: str = "") -> str:
        """
        Add document to RAG system using the best available backend
        
        Args:
            content: Document content
            metadata: Document metadata
            source: Document source
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = f"doc_{len(self.documents)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            
            # Chunk content
            chunks = self._chunk_content(content)
            
            # Prepare data
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_id": chunk_ids[i],
                    **(metadata or {})
                }
                chunk_metadata.append(chunk_meta)
            
            # Add to ChromaDB if available
            if self.chroma_collection is not None:
                try:
                    self.chroma_collection.add(
                        documents=chunks,
                        ids=chunk_ids,
                        metadatas=chunk_metadata
                    )
                    logger.info(f"Added {len(chunks)} chunks to ChromaDB")
                except Exception as e:
                    logger.warning(f"ChromaDB add failed: {e}")
            
            # Add to FAISS if available
            if FAISS_AVAILABLE:
                try:
                    embeddings = []
                    for chunk in chunks:
                        emb = self._get_embedding(chunk)
                        if emb is not None:
                            embeddings.append(emb)
                    
                    if embeddings:
                        embeddings_array = np.array(embeddings).astype('float32')
                        
                        if self.faiss_index is None:
                            # Create new index
                            dimension = embeddings_array.shape[1]
                            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                        
                        # Normalize embeddings for cosine similarity
                        faiss.normalize_L2(embeddings_array)
                        self.faiss_index.add(embeddings_array)
                        
                        # Store embeddings metadata
                        if self.faiss_embeddings is None:
                            self.faiss_embeddings = []
                        
                        for i, chunk_id in enumerate(chunk_ids):
                            self.faiss_embeddings.append({
                                "chunk_id": chunk_id,
                                "metadata": chunk_metadata[i],
                                "content": chunks[i]
                            })
                        
                        logger.info(f"Added {len(embeddings)} embeddings to FAISS")
                        
                except Exception as e:
                    logger.warning(f"FAISS add failed: {e}")
            
            # Add to fallback storage
            self.documents.extend(chunks)
            self.document_metadata.extend(chunk_metadata)
            
            # Update fallback embeddings
            self._update_fallback_embeddings()
            
            # Save data
            self._save_data()
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def _update_fallback_embeddings(self):
        """Update fallback embeddings"""
        try:
            if not self.documents:
                return
            
            embeddings = []
            for doc in self.documents:
                emb = self._get_embedding(doc)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    break
            
            if len(embeddings) == len(self.documents):
                self.document_embeddings = np.array(embeddings)
            else:
                # Fallback to TF-IDF
                self.document_embeddings = self.vectorizer.fit_transform(self.documents).toarray()
                
        except Exception as e:
            logger.error(f"Error updating fallback embeddings: {e}")
    
    def search_similar_content(self, query: str, n_results: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content using the best available backend
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Metadata filters
            
        Returns:
            List of similar content with metadata
        """
        try:
            # Try FAISS first (fastest)
            if self.faiss_index is not None and self.faiss_embeddings is not None:
                return self._search_faiss(query, n_results)
            
            # Try ChromaDB
            if self.chroma_collection is not None:
                return self._search_chromadb(query, n_results, filter_metadata)
            
            # Fallback to simple search
            return self._search_fallback(query, n_results)
            
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return []
    
    def _search_faiss(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Search using FAISS"""
        try:
            query_emb = self._get_embedding(query)
            if query_emb is None:
                return []
            
            query_emb = query_emb.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_emb)
            
            # Search
            scores, indices = self.faiss_index.search(query_emb, min(n_results, len(self.faiss_embeddings)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.faiss_embeddings) and score > 0.1:  # Minimum similarity threshold
                    embedding_data = self.faiss_embeddings[idx]
                    results.append({
                        "content": embedding_data["content"],
                        "metadata": embedding_data["metadata"],
                        "similarity": float(score),
                        "id": embedding_data["chunk_id"]
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
            return []
    
    def _search_chromadb(self, query: str, n_results: int, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search using ChromaDB"""
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            similar_content = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_content.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "similarity": 1.0 - results['distances'][0][i] if results['distances'] else 0.5,
                        "id": results['ids'][0][i] if results['ids'] else ""
                    })
            
            return similar_content
            
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
            return []
    
    def _search_fallback(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Fallback search using simple similarity"""
        try:
            if not self.documents or self.document_embeddings is None:
                return []
            
            query_emb = self._get_embedding(query)
            if query_emb is None:
                return []
            
            query_vector = query_emb.reshape(1, -1)
            similarities = cosine_similarity(query_vector, self.document_embeddings)[0]
            
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.document_metadata[idx],
                        "similarity": float(similarities[idx]),
                        "id": self.document_metadata[idx].get("chunk_id", f"chunk_{idx}")
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")
            return []
    
    def get_context_for_analysis(self, content: str, query: str = "", n_context: int = 3) -> str:
        """Get relevant context for content analysis"""
        try:
            search_query = query if query else content[:300]
            similar_content = self.search_similar_content(search_query, n_results=n_context)
            
            context_parts = []
            for item in similar_content:
                context_parts.append(f"Related content (similarity: {item['similarity']:.2f}): {item['content'][:200]}...")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return ""
    
    def enhance_analysis_with_rag(self, content: str, analysis_type: str = "summary", query: str = "") -> Dict[str, Any]:
        """Enhance analysis using advanced RAG system"""
        try:
            context = self.get_context_for_analysis(content, query)
            doc_id = self.add_document(content, {"analysis_type": analysis_type}, "current_analysis")
            
            return {
                "document_id": doc_id,
                "context": context,
                "enhanced": True,
                "analysis_type": analysis_type,
                "context_length": len(context),
                "rag_enhanced": True,
                "total_documents": len(self.documents),
                "backends_used": {
                    "chromadb": self.chroma_collection is not None,
                    "faiss": self.faiss_index is not None,
                    "fallback": len(self.documents) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error enhancing analysis: {str(e)}")
            return {
                "enhanced": False,
                "error": str(e),
                "rag_enhanced": False
            }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            stats = {
                "total_documents": len(self.documents),
                "total_chunks": len(self.documents),
                "embedding_model": "sentence-transformer" if self.embedding_model else "huggingface-api" if self.hf_api_key else "tf-idf",
                "backends": {
                    "chromadb": self.chroma_collection is not None,
                    "faiss": self.faiss_index is not None,
                    "fallback": True
                },
                "status": "active",
                "persist_directory": self.persist_directory
            }
            
            if self.faiss_index is not None:
                stats["faiss_index_size"] = self.faiss_index.ntotal
            
            if self.chroma_collection is not None:
                stats["chromadb_count"] = self.chroma_collection.count()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e), "status": "error"}
    
    def clear_database(self) -> Dict[str, Any]:
        """Clear all databases"""
        try:
            # Clear ChromaDB
            if self.chroma_collection is not None:
                self.chroma_client.delete_collection("content_embeddings")
                self.chroma_collection = self.chroma_client.create_collection("content_embeddings")
            
            # Clear FAISS
            self.faiss_index = None
            self.faiss_embeddings = None
            
            # Clear fallback
            self.documents.clear()
            self.document_metadata.clear()
            self.document_embeddings = None
            
            # Clear persisted files
            for filename in ["documents.pkl", "metadata.pkl", "embeddings.pkl", "faiss_index.bin", "faiss_embeddings.pkl"]:
                filepath = os.path.join(self.persist_directory, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            return {"status": "cleared", "message": "All databases cleared successfully"}
            
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {"status": "error", "message": str(e)}

