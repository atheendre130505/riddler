import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system for enhanced content analysis
    Uses vector embeddings and similarity search for context-aware responses
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize RAG system
        
        Args:
            persist_directory: Directory to persist vector database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="content_embeddings",
            metadata={"description": "Content embeddings for RAG system"}
        )
        
        logger.info("RAG system initialized successfully")
    
    def _generate_document_id(self, content: str, source: str = "") -> str:
        """
        Generate unique document ID
        
        Args:
            content: Document content
            source: Document source
            
        Returns:
            Unique document ID
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{source}_{timestamp}_{content_hash}"
    
    def _chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
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
            doc_id = self._generate_document_id(content, source)
            
            # Chunk content
            chunks = self._chunk_content(content)
            
            # Prepare data for ChromaDB
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }
                chunk_metadata.append(chunk_meta)
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                ids=chunk_ids,
                metadatas=chunk_metadata
            )
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def search_similar_content(self, query: str, n_results: int = 5, filter_metadata: Dict[str, any] = None) -> List[Dict[str, any]]:
        """
        Search for similar content using vector similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Metadata filters
            
        Returns:
            List of similar content with metadata
        """
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            similar_content = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_content.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        "id": results['ids'][0][i] if results['ids'] else ""
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
            search_query = query if query else content[:500]  # Use first 500 chars as query
            similar_content = self.search_similar_content(search_query, n_results=n_context)
            
            # Build context
            context_parts = []
            for item in similar_content:
                context_parts.append(f"Related content: {item['content'][:300]}...")
            
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
                "rag_enhanced": True
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
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "embedding_model": "all-MiniLM-L6-v2",
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e), "status": "error"}
    
    def clear_database(self) -> Dict[str, any]:
        """
        Clear the vector database
        
        Returns:
            Clear operation result
        """
        try:
            # Delete collection
            self.chroma_client.delete_collection("content_embeddings")
            
            # Recreate collection
            self.collection = self.chroma_client.create_collection(
                name="content_embeddings",
                metadata={"description": "Content embeddings for RAG system"}
            )
            
            return {"status": "cleared", "message": "Database cleared successfully"}
            
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {"status": "error", "message": str(e)}

class RAGFallback:
    """
    Fallback RAG implementation when ChromaDB is not available
    Provides basic similarity search using simple text matching
    """
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        logger.info("RAG fallback system initialized")
    
    def add_document(self, content: str, metadata: Dict[str, any] = None, source: str = "") -> str:
        """Add document to fallback system"""
        doc_id = f"fallback_doc_{len(self.documents)}"
        self.documents.append(content)
        self.metadata.append(metadata or {})
        return doc_id
    
    def search_similar_content(self, query: str, n_results: int = 5, filter_metadata: Dict[str, any] = None) -> List[Dict[str, any]]:
        """Simple similarity search using text matching"""
        similar_content = []
        query_lower = query.lower()
        
        for i, doc in enumerate(self.documents):
            if query_lower in doc.lower():
                similar_content.append({
                    "content": doc[:500] + "..." if len(doc) > 500 else doc,
                    "metadata": self.metadata[i],
                    "distance": 0.5,  # Dummy distance
                    "id": f"fallback_{i}"
                })
        
        return similar_content[:n_results]
    
    def get_context_for_analysis(self, content: str, query: str = "", n_context: int = 3) -> str:
        """Get context using fallback method"""
        similar_content = self.search_similar_content(query or content[:200], n_results=n_context)
        context_parts = [item["content"] for item in similar_content]
        return "\n\n".join(context_parts)
    
    def enhance_analysis_with_rag(self, content: str, analysis_type: str = "summary", query: str = "") -> Dict[str, any]:
        """Enhance analysis using fallback RAG"""
        doc_id = self.add_document(content, {"analysis_type": analysis_type}, "fallback_analysis")
        context = self.get_context_for_analysis(content, query)
        
        return {
            "document_id": doc_id,
            "context": context,
            "enhanced": True,
            "analysis_type": analysis_type,
            "context_length": len(context),
            "rag_enhanced": True,
            "fallback_mode": True
        }
    
    def get_document_stats(self) -> Dict[str, any]:
        """Get fallback system stats"""
        return {
            "total_documents": len(self.documents),
            "status": "fallback_active",
            "mode": "fallback"
        }
    
    def clear_database(self) -> Dict[str, any]:
        """Clear fallback database"""
        self.documents.clear()
        self.metadata.clear()
        return {"status": "cleared", "mode": "fallback"}


