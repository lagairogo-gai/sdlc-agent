from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    projects = relationship("Project", back_populates="owner")
    user_stories = relationship("UserStory", back_populates="creator")
    documents = relationship("Document", back_populates="uploader")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default="active")  # active, completed, archived
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Integration configurations
    jira_config = Column(JSON)
    confluence_config = Column(JSON)
    sharepoint_config = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    user_stories = relationship("UserStory", back_populates="project")
    documents = relationship("Document", back_populates="project")

class UserStory(Base):
    __tablename__ = "user_stories"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    acceptance_criteria = Column(Text)
    
    # Agile fields
    priority = Column(String, default="Medium")  # Low, Medium, High, Critical
    story_points = Column(Integer)
    epic = Column(String)
    sprint = Column(String)
    labels = Column(JSON)  # Array of labels
    
    # Status tracking
    status = Column(String, default="Draft")  # Draft, Ready, In Progress, Done
    jira_key = Column(String)  # When exported to Jira
    
    # Metadata
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # AI generation metadata
    generation_context = Column(JSON)  # Context used for generation
    confidence_score = Column(Float)  # AI confidence in the story
    source_documents = Column(JSON)  # List of source document IDs
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="user_stories")
    creator = relationship("User", back_populates="user_stories")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(Integer)
    file_path = Column(String)  # Path to stored file
    
    # Content processing
    processed_content = Column(Text)  # Extracted text content
    processing_status = Column(String, default="pending")  # pending, completed, failed
    processing_error = Column(Text)
    
    # Source information
    source = Column(String, nullable=False)  # upload, jira, confluence, sharepoint
    source_id = Column(String)  # Original ID from source system
    source_metadata = Column(JSON)  # Additional metadata from source
    
    # RAG system integration
    vector_id = Column(String)  # ID in vector database
    embedding_model = Column(String)  # Model used for embeddings
    
    # Relationships
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="documents")
    uploader = relationship("User", back_populates="documents")

class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String, unique=True, nullable=False)  # Unique identifier
    entity_type = Column(String, nullable=False)  # person, feature, system, requirement, etc.
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Graph properties
    properties = Column(JSON)  # Additional properties as JSON
    
    # Source tracking
    source_documents = Column(JSON)  # List of document IDs where this entity was found
    confidence_score = Column(Float)  # Confidence in entity extraction
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class KnowledgeRelation(Base):
    __tablename__ = "knowledge_relations"
    
    id = Column(Integer, primary_key=True, index=True)
    from_entity_id = Column(String, nullable=False)
    to_entity_id = Column(String, nullable=False)
    relation_type = Column(String, nullable=False)  # depends_on, implements, relates_to, etc.
    
    # Relation properties
    properties = Column(JSON)
    strength = Column(Float, default=1.0)  # Strength of the relation
    
    # Source tracking
    source_documents = Column(JSON)
    confidence_score = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class GenerationJob(Base):
    __tablename__ = "generation_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, nullable=False)
    
    # Job configuration
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Input parameters
    requirements = Column(Text)
    context = Column(Text)
    user_prompt = Column(Text)
    llm_config = Column(JSON)
    
    # Job status
    status = Column(String, default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    error_message = Column(Text)
    
    # Results
    generated_stories_count = Column(Integer, default=0)
    generated_stories_ids = Column(JSON)  # List of generated story IDs
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    project = relationship("Project")
    user = relationship("User")

class IntegrationLog(Base):
    __tablename__ = "integration_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    integration_type = Column(String, nullable=False)  # jira, confluence, sharepoint
    action = Column(String, nullable=False)  # sync, export, import
    
    project_id = Column(Integer, ForeignKey("projects.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Operation details
    status = Column(String, nullable=False)  # success, error, warning
    message = Column(Text)
    details = Column(JSON)  # Additional details about the operation
    
    # Metrics
    items_processed = Column(Integer, default=0)
    items_successful = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)
    duration_seconds = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    project = relationship("Project")
    user = relationship("User")