from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging
import asyncio
from contextlib import asynccontextmanager

from .database import get_db, init_db
from .models import UserStory, Document, Project, User
from .schemas import (
    UserStoryCreate, UserStoryResponse, DocumentResponse,
    ProjectCreate, ProjectResponse, UserCreate, UserResponse,
    GenerateUserStoryRequest, LLMConfig
)
from .services.rag_service import RAGService
from .services.knowledge_graph import KnowledgeGraphService
from .services.llm_service import LLMService
from .services.integrations import JiraService, ConfluenceService, SharePointService
from .services.document_processor import DocumentProcessor
from .agents.user_story_agent import UserStoryAgent
from .utils.auth import get_current_user, create_access_token
from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    init_db()
    
    # Initialize services
    app.state.rag_service = RAGService()
    app.state.kg_service = KnowledgeGraphService()
    app.state.llm_service = LLMService()
    app.state.document_processor = DocumentProcessor()
    app.state.user_story_agent = UserStoryAgent(
        rag_service=app.state.rag_service,
        kg_service=app.state.kg_service,
        llm_service=app.state.llm_service
    )
    
    # Initialize integrations
    app.state.jira_service = JiraService()
    app.state.confluence_service = ConfluenceService()
    app.state.sharepoint_service = SharePointService()
    
    logger.info("Application initialized successfully")
    yield
    
    # Cleanup
    logger.info("Application shutting down")

app = FastAPI(
    title="User Story AI Agent",
    description="RAG-based AI agent for generating user stories",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=user.password  # Should be hashed in production
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)

@app.post("/auth/login")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = db.query(User).filter(User.email == email).first()
    if not user or user.hashed_password != password:  # Simple auth for demo
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer", "user": UserResponse.from_orm(user)}

# Project endpoints
@app.post("/projects", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new project"""
    db_project = Project(**project.dict(), owner_id=current_user.id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return ProjectResponse.from_orm(db_project)

@app.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all projects for current user"""
    projects = db.query(Project).filter(Project.owner_id == current_user.id).all()
    return [ProjectResponse.from_orm(project) for project in projects]

# Document upload and processing
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    project_id: int = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    try:
        # Save uploaded file
        file_content = await file.read()
        
        # Process document
        processed_content = await app.state.document_processor.process_document(
            file_content, file.filename, file.content_type
        )
        
        # Store in database
        db_document = Document(
            filename=file.filename,
            content_type=file.content_type,
            processed_content=processed_content,
            project_id=project_id,
            uploaded_by=current_user.id
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Add to RAG system
        await app.state.rag_service.add_document(
            doc_id=str(db_document.id),
            content=processed_content,
            metadata={
                "filename": file.filename,
                "project_id": project_id,
                "source": "upload"
            }
        )
        
        # Update knowledge graph
        await app.state.kg_service.add_document_knowledge(
            db_document.id, processed_content, {"project_id": project_id}
        )
        
        return DocumentResponse.from_orm(db_document)
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Integration endpoints
@app.post("/integrations/jira/sync/{project_id}")
async def sync_jira_data(
    project_id: int,
    jira_config: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Sync data from Jira"""
    try:
        # Configure Jira service
        app.state.jira_service.configure(jira_config)
        
        # Fetch requirements from Jira
        requirements = await app.state.jira_service.fetch_requirements(
            project_key=jira_config.get("project_key")
        )
        
        # Process and store requirements
        for req in requirements:
            await app.state.rag_service.add_document(
                doc_id=f"jira_{req['key']}",
                content=req['description'],
                metadata={
                    "source": "jira",
                    "project_id": project_id,
                    "jira_key": req['key'],
                    "issue_type": req['issue_type']
                }
            )
        
        return {"message": f"Synced {len(requirements)} requirements from Jira"}
    
    except Exception as e:
        logger.error(f"Error syncing Jira data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error syncing Jira: {str(e)}")

@app.post("/integrations/confluence/sync/{project_id}")
async def sync_confluence_data(
    project_id: int,
    confluence_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Sync data from Confluence"""
    try:
        app.state.confluence_service.configure(confluence_config)
        
        pages = await app.state.confluence_service.fetch_pages(
            space_key=confluence_config.get("space_key")
        )
        
        for page in pages:
            await app.state.rag_service.add_document(
                doc_id=f"confluence_{page['id']}",
                content=page['content'],
                metadata={
                    "source": "confluence",
                    "project_id": project_id,
                    "page_title": page['title'],
                    "space_key": page['space_key']
                }
            )
        
        return {"message": f"Synced {len(pages)} pages from Confluence"}
    
    except Exception as e:
        logger.error(f"Error syncing Confluence data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error syncing Confluence: {str(e)}")

# User story generation
@app.post("/user-stories/generate", response_model=List[UserStoryResponse])
async def generate_user_stories(
    request: GenerateUserStoryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate user stories using AI agent"""
    try:
        # Configure LLM
        app.state.llm_service.configure(request.llm_config.dict())
        
        # Generate user stories
        generated_stories = await app.state.user_story_agent.generate_user_stories(
            project_id=request.project_id,
            requirements=request.requirements,
            context=request.context,
            user_prompt=request.user_prompt
        )
        
        # Store generated stories
        db_stories = []
        for story_data in generated_stories:
            db_story = UserStory(
                title=story_data['title'],
                description=story_data['description'],
                acceptance_criteria=story_data['acceptance_criteria'],
                priority=story_data.get('priority', 'Medium'),
                story_points=story_data.get('story_points'),
                epic=story_data.get('epic'),
                project_id=request.project_id,
                created_by=current_user.id
            )
            db.add(db_story)
            db_stories.append(db_story)
        
        db.commit()
        
        # Refresh all stories
        for story in db_stories:
            db.refresh(story)
        
        return [UserStoryResponse.from_orm(story) for story in db_stories]
    
    except Exception as e:
        logger.error(f"Error generating user stories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating user stories: {str(e)}")

@app.get("/user-stories/{project_id}", response_model=List[UserStoryResponse])
async def list_user_stories(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user stories for a project"""
    stories = db.query(UserStory).filter(UserStory.project_id == project_id).all()
    return [UserStoryResponse.from_orm(story) for story in stories]

@app.post("/user-stories/{story_id}/export-jira")
async def export_to_jira(
    story_id: int,
    jira_config: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export user story to Jira"""
    try:
        story = db.query(UserStory).filter(UserStory.id == story_id).first()
        if not story:
            raise HTTPException(status_code=404, detail="User story not found")
        
        app.state.jira_service.configure(jira_config)
        jira_issue = await app.state.jira_service.create_user_story(
            project_key=jira_config.get("project_key"),
            title=story.title,
            description=story.description,
            acceptance_criteria=story.acceptance_criteria,
            priority=story.priority,
            story_points=story.story_points
        )
        
        # Update story with Jira key
        story.jira_key = jira_issue['key']
        db.commit()
        
        return {"message": f"Exported to Jira as {jira_issue['key']}", "jira_key": jira_issue['key']}
    
    except Exception as e:
        logger.error(f"Error exporting to Jira: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting to Jira: {str(e)}")

# Analytics and insights
@app.get("/analytics/project/{project_id}")
async def get_project_analytics(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get analytics for a project"""
    try:
        analytics = await app.state.kg_service.get_project_insights(project_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)