import os
import json
import logging
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAG:
    """
    Simplified RAG system using TF-IDF and Hugging Face embeddings
    Works with available APIs: Gemini + Hugging Face
    """
    
    def __init__(self, persist_directory: str = "./rag_data"):
        """
        Initialize simple RAG system
        
        Args:
            persist_directory: Directory to persist data
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Document storage
        self.documents = []
        self.document_metadata = []
        self.document_embeddings = None
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Load existing data
        self._load_data()
        
        logger.info("Simple RAG system initialized")
    
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
        """
        Get embedding from Hugging Face API
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.hf_api_key:
            return None
        
        try:
            # Use Hugging Face inference API
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
        """
        Split content into overlapping chunks
        
        Args:
            content: Content to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of content chunks
        """
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def add_document(self, content: str, metadata: Dict[str, any] = None, source: str = "") -> str:
        """
        Add document to RAG system
        
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
            
            # Add chunks to storage
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.documents.append(chunk)
                
                chunk_metadata = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_id": chunk_id,
                    **(metadata or {})
                }
                self.document_metadata.append(chunk_metadata)
            
            # Update embeddings
            self._update_embeddings()
            
            # Save data
            self._save_data()
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def _update_embeddings(self):
        """Update document embeddings"""
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
                    # Fallback to TF-IDF
                    break
            
            if len(embeddings) == len(self.documents):
                # All HF embeddings successful
                self.document_embeddings = np.array(embeddings)
                logger.info("Updated embeddings using Hugging Face")
            else:
                # Use TF-IDF fallback
                self.document_embeddings = self.vectorizer.fit_transform(self.documents).toarray()
                logger.info("Updated embeddings using TF-IDF")
                
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            # Fallback to TF-IDF
            try:
                self.document_embeddings = self.vectorizer.fit_transform(self.documents).toarray()
            except Exception as e2:
                logger.error(f"TF-IDF fallback also failed: {e2}")
    
    def search_similar_content(self, query: str, n_results: int = 5, filter_metadata: Dict[str, any] = None) -> List[Dict[str, any]]:
        """
        Search for similar content
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Metadata filters
            
        Returns:
            List of similar content with metadata
        """
        try:
            if not self.documents or self.document_embeddings is None:
                return []
            
            # Get query embedding
            query_emb = self._get_hf_embedding(query)
            if query_emb is None:
                # Fallback to TF-IDF
                query_vector = self.vectorizer.transform([query]).toarray()
            else:
                query_vector = query_emb.reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            # Format results
            similar_content = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    similar_content.append({
                        "content": self.documents[idx],
                        "metadata": self.document_metadata[idx],
                        "similarity": float(similarities[idx]),
                        "id": self.document_metadata[idx].get("chunk_id", f"chunk_{idx}")
                    })
            
            return similar_content
            
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return []
    
    def get_context_for_analysis(self, content: str, query: str = "", n_context: int = 3) -> str:
        """
        Get relevant context for content analysis
        
        Args:
            content: Content to analyze
            query: Analysis query
            n_context: Number of context chunks to retrieve
            
        Returns:
            Contextual information
        """
        try:
            # Search for similar content
            search_query = query if query else content[:300]  # Use first 300 chars as query
            similar_content = self.search_similar_content(search_query, n_results=n_context)
            
            # Build context
            context_parts = []
            for item in similar_content:
                context_parts.append(f"Related content (similarity: {item['similarity']:.2f}): {item['content'][:200]}...")
            
            context = "\n\n".join(context_parts)
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return ""
    
    def enhance_analysis_with_rag(self, content: str, analysis_type: str = "summary", query: str = "") -> Dict[str, any]:
        """
        Enhance analysis using RAG system
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis
            query: Analysis query
            
        Returns:
            Enhanced analysis result
        """
        try:
            # Get relevant context
            context = self.get_context_for_analysis(content, query)
            
            # Add current document to system for future reference
            doc_id = self.add_document(content, {"analysis_type": analysis_type}, "current_analysis")
            
            return {
                "document_id": doc_id,
                "context": context,
                "enhanced": True,
                "analysis_type": analysis_type,
                "context_length": len(context),
                "rag_enhanced": True,
                "total_documents": len(self.documents)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing analysis: {str(e)}")
            return {
                "enhanced": False,
                "error": str(e),
                "rag_enhanced": False
            }
    
    def get_document_stats(self) -> Dict[str, any]:
        """
        Get RAG system statistics
        
        Returns:
            System statistics
        """
        try:
            return {
                "total_documents": len(self.documents),
                "total_chunks": len(self.documents),
                "embedding_type": "Hugging Face" if self.hf_api_key else "TF-IDF",
                "status": "active",
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e), "status": "error"}
    
    def clear_database(self) -> Dict[str, any]:
        """
        Clear the database
        
        Returns:
            Clear operation result
        """
        try:
            self.documents.clear()
            self.document_metadata.clear()
            self.document_embeddings = None
            
            # Clear persisted files
            for filename in ["documents.pkl", "metadata.pkl", "embeddings.pkl"]:
                filepath = os.path.join(self.persist_directory, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            return {"status": "cleared", "message": "Database cleared successfully"}
            
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {"status": "error", "message": str(e)}

