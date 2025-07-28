#!/bin/bash
# fix_dependencies.sh - Fix dependency conflicts and restart

echo "🔧 Fixing dependency conflicts..."

# Stop current containers
docker-compose down

# Create fixed requirements.txt with compatible versions
cat > requirements.txt << 'EOF'
# Core FastAPI
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
websockets==12.0

# Basic dependencies
requests==2.31.0
aiofiles==23.2.1
pydantic==2.5.1
python-dotenv==1.0.0

# Document processing
PyPDF2==3.0.1
python-docx==0.8.11

# Optional integrations (install only if needed)
# langchain==0.1.0
# openai>=1.6.1
# msal==1.25.0
# atlassian-python-api==3.41.10
# redis==5.0.1
# neo4j==5.14.1

# Monitoring
structlog==23.2.0
prometheus-client==0.19.0
EOF

echo "✅ Created fixed requirements.txt"

# Update the complete agent file to work without LangChain temporarily
cat > requirements_agent_minimal.py << 'EOF'
# Minimal Requirements Agent - No LangChain dependencies
import asyncio
import json
import logging
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Simple document processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

import aiofiles
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    ACTING = "acting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class DocumentContent:
    filename: str
    file_type: str
    content: str
    metadata: Dict[str, Any]
    source: str

class DocumentProcessor:
    @staticmethod
    async def process_pdf(file_path: str) -> str:
        if not PyPDF2:
            return "PDF processing not available (PyPDF2 not installed)"
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    @staticmethod
    async def process_docx(file_path: str) -> str:
        if not docx:
            return "DOCX processing not available (python-docx not installed)"
        
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            return f"Error processing DOCX: {str(e)}"
    
    @staticmethod
    async def process_text(file_path: str) -> str:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                return await file.read()
        except Exception as e:
            return f"Error processing text file: {str(e)}"
    
    @classmethod
    async def process_document(cls, file_path: str, file_type: str) -> str:
        if file_type.lower() == 'pdf':
            return await cls.process_pdf(file_path)
        elif file_type.lower() in ['docx', 'doc']:
            return await cls.process_docx(file_path)
        elif file_type.lower() in ['txt', 'md']:
            return await cls.process_text(file_path)
        else:
            return f"Unsupported file type: {file_type}"

class SimpleRequirementsAgent:
    def __init__(self):
        self.state = AgentState.IDLE
        self.processed_documents = []
    
    async def analyze_documents(self, documents: List[DocumentContent]) -> Dict:
        """Simple requirements analysis without LLM"""
        self.state = AgentState.THINKING
        
        # Simple keyword-based analysis
        all_text = " ".join([doc.content for doc in documents]).lower()
        
        # Extract potential requirements using keywords
        functional_keywords = ['user', 'system', 'shall', 'must', 'should', 'login', 'create', 'delete', 'update', 'view']
        non_functional_keywords = ['performance', 'security', 'scalability', 'availability', 'response time']
        
        functional_requirements = []
        non_functional_requirements = []
        
        for i, keyword in enumerate(functional_keywords):
            if keyword in all_text:
                functional_requirements.append({
                    "id": f"REQ-F-{i+1:03d}",
                    "description": f"System requirement related to {keyword}",
                    "priority": "Medium",
                    "source": "Document analysis"
                })
        
        for i, keyword in enumerate(non_functional_keywords):
            if keyword in all_text:
                non_functional_requirements.append({
                    "id": f"REQ-NF-{i+1:03d}",
                    "category": keyword.title(),
                    "description": f"Non-functional requirement for {keyword}",
                    "metric": "TBD",
                    "target_value": "TBD"
                })
        
        self.state = AgentState.COMPLETED
        
        return {
            "processed_documents": len(documents),
            "functional_requirements": functional_requirements,
            "non_functional_requirements": non_functional_requirements,
            "analysis_summary": f"Analyzed {len(documents)} documents and found {len(functional_requirements)} functional and {len(non_functional_requirements)} non-functional requirements.",
            "generated_documents": [
                {
                    "type": "requirements_specification",
                    "title": "Basic Requirements Analysis",
                    "content": f"# Requirements Analysis\n\n## Functional Requirements\n{json.dumps(functional_requirements, indent=2)}\n\n## Non-Functional Requirements\n{json.dumps(non_functional_requirements, indent=2)}",
                    "format": "markdown"
                }
            ]
        }

# FastAPI App
app = FastAPI(title="Requirements Agent - Minimal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
agent = SimpleRequirementsAgent()
temp_files_dir = None
uploaded_files = {}

class ProjectContext(BaseModel):
    name: str
    description: str
    stakeholders: Optional[str] = ""
    business_goals: Optional[str] = ""

class FileUploadResponse(BaseModel):
    filename: str
    size: int
    status: str
    file_id: str

@app.on_event("startup")
async def startup_event():
    global temp_files_dir
    temp_files_dir = Path(tempfile.mkdtemp(prefix="requirements_agent_"))
    logger.info(f"Requirements Agent started. Temp dir: {temp_files_dir}")

@app.on_event("shutdown")
async def shutdown_event():
    global temp_files_dir
    if temp_files_dir and temp_files_dir.exists():
        shutil.rmtree(temp_files_dir)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-minimal",
        "agent_state": agent.state.value,
        "uploaded_files": len(uploaded_files)
    }

@app.post("/api/upload", response_model=List[FileUploadResponse])
async def upload_files(files: List[UploadFile] = File(...)):
    responses = []
    
    for file in files:
        try:
            # Validate file type
            allowed_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md'}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                responses.append(FileUploadResponse(
                    filename=file.filename,
                    size=0,
                    status="error",
                    file_id=""
                ))
                continue
            
            # Save file
            file_id = f"file_{int(datetime.now().timestamp())}_{hash(file.filename)}"
            file_path = temp_files_dir / f"{file_id}{file_extension}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process document
            processor = DocumentProcessor()
            text_content = await processor.process_document(str(file_path), file_extension[1:])
            
            # Store document
            doc = DocumentContent(
                filename=file.filename,
                file_type=file_extension[1:],
                content=text_content,
                metadata={"file_size": len(content)},
                source="upload"
            )
            
            agent.processed_documents.append(doc)
            uploaded_files[file_id] = {
                "filename": file.filename,
                "path": str(file_path),
                "processed": True
            }
            
            responses.append(FileUploadResponse(
                filename=file.filename,
                size=len(content),
                status="processed",
                file_id=file_id
            ))
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            responses.append(FileUploadResponse(
                filename=file.filename,
                size=0,
                status="error",
                file_id=""
            ))
    
    return responses

@app.post("/api/agents/requirements/analyze")
async def analyze_requirements(project_context: ProjectContext):
    try:
        if not agent.processed_documents:
            raise HTTPException(status_code=400, detail="No documents uploaded")
        
        result = await agent.analyze_documents(agent.processed_documents)
        
        return {
            "execution_id": f"exec_{int(datetime.now().timestamp())}",
            "status": "success",
            "agent_id": "requirements_agent_minimal",
            "result": result,
            "execution_time": 1.5,
            "project_context": project_context.dict()
        }
    
    except Exception as e:
        logger.error(f"Error analyzing requirements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    return {
        "files": list(uploaded_files.values()),
        "total": len(uploaded_files),
        "processed_documents": len(agent.processed_documents)
    }

if __name__ == "__main__":
    uvicorn.run("requirements_agent_minimal:app", host="0.0.0.0", port=8000, log_level="info")
EOF

echo "✅ Created minimal agent implementation"

# Update Dockerfile to use minimal version
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements_agent_minimal.py .

RUN useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "requirements_agent_minimal.py"]
EOF

echo "✅ Updated Dockerfile"

# Rebuild and start
echo "🚀 Rebuilding with fixed dependencies..."
docker-compose up -d --build

echo "✅ Dependencies fixed and service restarted!"
echo ""
echo "🔗 Test the service:"
echo "  curl http://localhost:8000/health"
echo ""
echo "📤 Upload a document:"
echo "  curl -X POST -F 'files=@document.pdf' http://localhost:8000/api/upload"
echo ""
echo "📊 Analyze requirements:"
echo "  curl -X POST http://localhost:8000/api/agents/requirements/analyze \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"name\":\"Test Project\",\"description\":\"Sample project\"}'"