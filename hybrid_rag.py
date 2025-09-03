import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import hashlib
from datetime import datetime
import requests
import pickle

# Try to import advanced libraries, fallback gracefully
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using basic similarity")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available, using fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRAG:
    """
    Hybrid RAG system that adapts to available libraries
    Uses ChromaDB + Hugging Face + scikit-learn for maximum performance
    """
    
    def __init__(self, persist_directory: str = "./hybrid_rag", use_chromadb: bool = True):
        """
        Initialize hybrid RAG system
        
        Args:
            persist_directory: Directory to persist data
            use_chromadb: Use ChromaDB for vector storage
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # API keys
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Initialize vector stores
        self.chroma_client = None
        self.chroma_collection = None
        
        # Fallback storage
        self.documents = []
        self.document_metadata = []
        self.document_embeddings = None
        
        # Initialize TF-IDF if available
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = None
        
        # Initialize backends
        if use_chromadb and CHROMADB_AVAILABLE:
            self._init_chromadb()
        
        # Load existing data
        self._load_data()
        
        logger.info("Hybrid RAG system initialized")
    
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=os.path.join(self.persist_directory, "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="content_embeddings",
                metadata={"description": "Hybrid RAG content embeddings"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
            self.chroma_client = None
            self.chroma_collection = None
    
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
            
        except Exception as e:
            logger.error(f"Could not save data: {e}")
    
    def _get_hf_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Hugging Face API"""
        if not self.hf_api_key:
            return None
        
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
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity when scikit-learn is not available"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
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
            
            # Try Hugging Face embeddings first
            embeddings = []
            for doc in self.documents:
                hf_emb = self._get_hf_embedding(doc)
                if hf_emb is not None:
                    embeddings.append(hf_emb)
                else:
                    break
            
            if len(embeddings) == len(self.documents):
                # All HF embeddings successful
                self.document_embeddings = np.array(embeddings)
                logger.info("Updated embeddings using Hugging Face")
            elif self.vectorizer is not None:
                # Use TF-IDF fallback
                self.document_embeddings = self.vectorizer.fit_transform(self.documents).toarray()
                logger.info("Updated embeddings using TF-IDF")
            else:
                # No embeddings available
                self.document_embeddings = None
                logger.warning("No embedding method available")
                
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
            # Try ChromaDB first
            if self.chroma_collection is not None:
                return self._search_chromadb(query, n_results, filter_metadata)
            
            # Fallback to simple search
            return self._search_fallback(query, n_results)
            
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
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
        """Fallback search using available similarity methods"""
        try:
            if not self.documents:
                return []
            
            # Try vector similarity if embeddings available
            if self.document_embeddings is not None and SKLEARN_AVAILABLE:
                return self._search_vector_similarity(query, n_results)
            
            # Fallback to simple word similarity
            return self._search_word_similarity(query, n_results)
            
        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")
            return []
    
    def _search_vector_similarity(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Search using vector similarity"""
        try:
            query_emb = self._get_hf_embedding(query)
            if query_emb is None and self.vectorizer is not None:
                query_vector = self.vectorizer.transform([query]).toarray()
            elif query_emb is not None:
                query_vector = query_emb.reshape(1, -1)
            else:
                return []
            
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
            logger.warning(f"Vector similarity search failed: {e}")
            return []
    
    def _search_word_similarity(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Search using simple word similarity"""
        try:
            similarities = []
            for i, doc in enumerate(self.documents):
                sim = self._simple_similarity(query, doc)
                similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for idx, sim in similarities[:n_results]:
                if sim > 0.1:
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.document_metadata[idx],
                        "similarity": sim,
                        "id": self.document_metadata[idx].get("chunk_id", f"chunk_{idx}")
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"Word similarity search failed: {e}")
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
        """Enhance analysis using hybrid RAG system"""
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
                    "huggingface": self.hf_api_key is not None,
                    "sklearn": SKLEARN_AVAILABLE,
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
                "embedding_method": "huggingface-api" if self.hf_api_key else "tf-idf" if self.vectorizer else "word-similarity",
                "backends": {
                    "chromadb": self.chroma_collection is not None,
                    "huggingface": self.hf_api_key is not None,
                    "sklearn": SKLEARN_AVAILABLE,
                    "fallback": True
                },
                "status": "active",
                "persist_directory": self.persist_directory
            }
            
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
            
            # Clear fallback
            self.documents.clear()
            self.document_metadata.clear()
            self.document_embeddings = None
            
            # Clear persisted files
            for filename in ["documents.pkl", "metadata.pkl", "embeddings.pkl"]:
                filepath = os.path.join(self.persist_directory, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            return {"status": "cleared", "message": "All databases cleared successfully"}
            
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {"status": "error", "message": str(e)}

