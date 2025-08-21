import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGService:
    """Retrieval-Augmented Generation service using Qdrant vector database"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = "user_story_documents"
        self.embedding_model = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self._initialize_collection()
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            # Try OpenAI embeddings first
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            self.embedding_dimension = 1536
        except Exception as e:
            logger.warning(f"OpenAI embeddings not available: {e}, falling back to HuggingFace")
            # Fallback to HuggingFace embeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embedding_dimension = 384
    
    def _initialize_collection(self):
        """Initialize Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension if hasattr(self, 'embedding_dimension') else 1536,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
    
    async def add_document(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a document to the vector database"""
        try:
            # Split document into chunks
            documents = self.text_splitter.create_documents([content], [metadata])
            
            points = []
            for i, doc in enumerate(documents):
                # Generate embedding
                embedding = await self._get_embedding(doc.page_content)
                
                # Create unique point ID
                point_id = hashlib.md5(f"{doc_id}_{i}".encode()).hexdigest()
                
                # Prepare metadata
                point_metadata = {
                    **doc.metadata,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "content": doc.page_content,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=point_metadata
                    )
                )
            
            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added document {doc_id} with {len(points)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False
    
    async def retrieve_documents(
        self, 
        query: str, 
        project_id: Optional[int] = None,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(query)
            
            # Prepare filter
            query_filter = None
            if project_id:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=project_id)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            documents = []
            for result in search_results:
                documents.append({
                    "id": result.payload.get("doc_id"),
                    "content": result.payload.get("content"),
                    "metadata": {k: v for k, v in result.payload.items() if k not in ["content", "doc_id"]},
                    "score": result.score
                })
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if hasattr(self.embedding_model, 'aembed_query'):
                embedding = await self.embedding_model.aembed_query(text)
            else:
                # Synchronous fallback
                embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            # Find all points for this document
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            )
            
            # Delete points
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=search_filter)
            )
            
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    async def get_document_statistics(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics about documents in the collection"""
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            stats = {
                "total_points": collection_info.points_count,
                "total_documents": 0,
                "project_breakdown": {}
            }
            
            # If project_id specified, get project-specific stats
            if project_id:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=project_id)
                        )
                    ]
                )
                
                # This is a simplified approach - in production, you'd want to implement
                # proper aggregation queries
                search_results = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=search_filter,
                    limit=1000  # Limit for demo purposes
                )
                
                stats["project_documents"] = len(set(point.payload.get("doc_id") for point in search_results[0]))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    async def similarity_search_with_metadata(
        self, 
        query: str, 
        metadata_filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search with specific metadata filters"""
        try:
            query_embedding = await self._get_embedding(query)
            
            # Build filter conditions
            filter_conditions = []
            for key, value in metadata_filters.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            
            query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit
            )
            
            return [{
                "id": result.payload.get("doc_id"),
                "content": result.payload.get("content"),
                "metadata": result.payload,
                "score": result.score
            } for result in search_results]
            
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []