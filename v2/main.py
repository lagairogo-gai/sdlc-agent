# main.py - FastAPI server for Requirements Agent with Real Document Processing
import asyncio
import json
import logging
import os
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from requirements_agent import AgentOrchestrator, AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI SDLC - Requirements Agent",
    description="Autonomous Requirements Engineering Agent with Real Document Processing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TaskRequest(BaseModel):
    type: str
    project_context: Dict
    use_uploaded_files: bool = True
    use_integrations: bool = True

class IntegrationRequest(BaseModel):
    integration_type: str
    config: Dict[str, str]

class ProjectContext(BaseModel):
    name: str
    description: str
    stakeholders: Optional[str] = ""
    business_goals: Optional[str] = ""

class AgentResponse(BaseModel):
    execution_id: str
    status: str
    agent_id: str
    result: Dict
    mcp_events: List[Dict]
    a2a_events: List[Dict]
    execution_time: float

class FileUploadResponse(BaseModel):
    filename: str
    size: int
    status: str
    file_id: str

class IntegrationResponse(BaseModel):
    integration_type: str
    status: str
    message: str
    connected_at: Optional[str] = None

# Global variables
orchestrator = None
uploaded_files_storage = {}  # Store uploaded files metadata
temp_files_dir = None

# Initialize orchestrator and temp directory
@app.on_event("startup")
async def startup_event():
    global orchestrator, temp_files_dir
    
    # Create temporary directory for uploaded files
    temp_files_dir = Path(tempfile.mkdtemp(prefix="requirements_agent_"))
    logger.info(f"Created temporary directory: {temp_files_dir}")
    
    llm_config = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-4"
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY", "")
        },
        "azure": {
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_key": os.getenv("AZURE_OPENAI_KEY", "")
        }
    }
    
    orchestrator = AgentOrchestrator(llm_config)
    logger.info("Requirements Agent Orchestrator initialized")

@app.on_event("shutdown")
async def shutdown_event():
    global temp_files_dir
    
    # Clean up temporary files
    if temp_files_dir and temp_files_dir.exists():
        shutil.rmtree(temp_files_dir)
        logger.info("Cleaned up temporary files")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# File upload endpoints
@app.post("/api/upload", response_model=List[FileUploadResponse])
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple requirement documents"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    uploaded_responses = []
    
    for file in files:
        try:
            # Validate file type
            allowed_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md'}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
                )
            
            # Generate unique file ID and save file
            file_id = f"file_{int(datetime.now().timestamp())}_{hash(file.filename)}"
            file_path = temp_files_dir / f"{file_id}{file_extension}"
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Store file metadata
            file_metadata = {
                "id": file_id,
                "filename": file.filename,
                "original_name": file.filename,
                "file_path": str(file_path),
                "size": len(content),
                "content_type": file.content_type,
                "uploaded_at": datetime.now().isoformat(),
                "status": "uploaded"
            }
            
            uploaded_files_storage[file_id] = file_metadata
            
            # Process the document
            requirements_agent = orchestrator.agents["requirements"]
            processed_docs = await requirements_agent.process_uploaded_documents([str(file_path)])
            
            if processed_docs:
                file_metadata["status"] = "processed"
                file_metadata["processed_at"] = datetime.now().isoformat()
            else:
                file_metadata["status"] = "failed"
            
            uploaded_responses.append(FileUploadResponse(
                filename=file.filename,
                size=len(content),
                status=file_metadata["status"],
                file_id=file_id
            ))
            
            # Broadcast to WebSocket clients
            await manager.broadcast(json.dumps({
                "type": "file_uploaded",
                "data": {
                    "filename": file.filename,
                    "status": file_metadata["status"],
                    "file_id": file_id
                }
            }))
            
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            uploaded_responses.append(FileUploadResponse(
                filename=file.filename,
                size=0,
                status="error",
                file_id=""
            ))
    
    return uploaded_responses

@app.get("/api/files")
async def list_uploaded_files():
    """Get list of uploaded files"""
    return {
        "files": list(uploaded_files_storage.values()),
        "total": len(uploaded_files_storage)
    }

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file"""
    if file_id not in uploaded_files_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_metadata = uploaded_files_storage[file_id]
    
    # Delete physical file
    file_path = Path(file_metadata["file_path"])
    if file_path.exists():
        file_path.unlink()
    
    # Remove from storage
    del uploaded_files_storage[file_id]
    
    return {"message": "File deleted successfully", "file_id": file_id}

# Integration endpoints
@app.post("/api/integrations/connect", response_model=IntegrationResponse)
async def connect_integration(integration_request: IntegrationRequest):
    """Connect to external integration"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        requirements_agent = orchestrator.agents["requirements"]
        success = await requirements_agent.connect_integration(
            integration_request.integration_type,
            integration_request.config
        )
        
        if success:
            return IntegrationResponse(
                integration_type=integration_request.integration_type,
                status="connected",
                message="Successfully connected to integration",
                connected_at=datetime.now().isoformat()
            )
        else:
            return IntegrationResponse(
                integration_type=integration_request.integration_type,
                status="failed",
                message="Failed to connect to integration"
            )
    
    except Exception as e:
        logger.error(f"Error connecting integration: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/integrations/{integration_type}/fetch")
async def fetch_from_integration(integration_type: str):
    """Fetch documents from connected integration"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        requirements_agent = orchestrator.agents["requirements"]
        documents = await requirements_agent.fetch_from_integrations()
        
        await manager.broadcast(json.dumps({
            "type": "integration_fetch_complete",
            "data": {
                "integration_type": integration_type,
                "documents_count": len(documents)
            }
        }))
        
        return {
            "integration_type": integration_type,
            "documents_fetched": len(documents),
            "documents": [
                {
                    "filename": doc.filename,
                    "source": doc.source,
                    "file_type": doc.file_type,
                    "size": len(doc.content)
                }
                for doc in documents
            ]
        }
    
    except Exception as e:
        logger.error(f"Error fetching from integration: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/integrations")
async def list_integrations():
    """Get status of all integrations"""
    if not orchestrator:
        return {"integrations": {}}
    
    requirements_agent = orchestrator.agents["requirements"]
    integrations_status = {}
    
    for integration_type, integration in requirements_agent.integration_manager.integrations.items():
        integrations_status[integration_type] = {
            "connected": integration.is_connected,
            "last_sync": integration.last_sync.isoformat() if integration.last_sync else None,
            "config_keys": list(integration.config.keys()) if integration.config else []
        }
    
    return {"integrations": integrations_status}

# Requirements analysis endpoint
@app.post("/api/agents/requirements/analyze", response_model=AgentResponse)
async def analyze_requirements(
    project_context: ProjectContext,
    background_tasks: BackgroundTasks
):
    """Analyze requirements from uploaded documents and integrations"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        start_time = datetime.now()
        
        # Convert project context
        task_dict = {
            "type": "requirements_analysis",
            "project_context": project_context.dict(),
            "uploaded_files": list(uploaded_files_storage.keys()),
            "use_integrations": True
        }
        
        # Create context
        context = AgentContext(
            project_id=f"proj_{int(start_time.timestamp())}",
            session_id=f"sess_{int(start_time.timestamp())}",
            current_task=task_dict,
            conversation_history=[],
            shared_memory={
                "uploaded_files": uploaded_files_storage,
                "project_context": project_context.dict()
            },
            available_tools=list(orchestrator.tool_registry.tools.keys())
        )
        
        # Execute requirements agent
        result = await orchestrator.execute_agent_task("requirements", task_dict, context)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        response = AgentResponse(
            execution_id=result["execution_id"],
            status=result["status"],
            agent_id="requirements_agent",
            result=result["result"],
            mcp_events=result["mcp_events"],
            a2a_events=result["a2a_events"],
            execution_time=execution_time
        )
        
        # Broadcast completion to WebSocket clients
        background_tasks.add_task(
            manager.broadcast,
            json.dumps({
                "type": "requirements_analysis_complete",
                "data": {
                    "execution_id": response.execution_id,
                    "status": response.status,
                    "execution_time": execution_time,
                    "documents_processed": result["result"].get("processed_documents", 0),
                    "requirements_found": len(result["result"].get("functional_requirements", [])),
                    "documents_generated": len(result["result"].get("generated_documents", []))
                }
            })
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing requirements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for backward compatibility
@app.post("/api/agents/requirements/execute", response_model=AgentResponse)
async def execute_requirements_agent(task: TaskRequest, background_tasks: BackgroundTasks):
    """Execute the requirements agent with given task (legacy endpoint)"""
    project_context = ProjectContext(**task.project_context)
    return await analyze_requirements(project_context, background_tasks)

@app.get("/api/agents/requirements/capabilities")
async def get_agent_capabilities():
    """Get requirements agent capabilities"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    agent = orchestrator.agents.get("requirements")
    if not agent:
        raise HTTPException(status_code=404, detail="Requirements agent not found")
    
    return {
        "agent_id": agent.agent_id,
        "capabilities": [
            {
                "name": cap.name,
                "description": cap.description,
                "input_schema": cap.input_schema,
                "output_schema": cap.output_schema,
                "tools": cap.tools
            }
            for cap in agent.capabilities
        ],
        "state": agent.state.value,
        "available_tools": list(orchestrator.tool_registry.tools.keys()),
        "supported_file_types": [".pdf", ".doc", ".docx", ".txt", ".md"],
        "supported_integrations": ["confluence", "jira", "sharepoint", "onedrive", "googledrive"]
    }

@app.get("/api/tools")
async def get_available_tools():
    """Get all available tools for integration"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    return {
        "tools": orchestrator.tool_registry.tools,
        "count": len(orchestrator.tool_registry.tools)
    }

@app.post("/api/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, action: str, parameters: Dict):
    """Execute a specific tool action"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        result = await orchestrator.tool_registry.execute_tool(tool_name, action, parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/agents/requirements/events/{execution_id}")
async def get_execution_events(execution_id: str):
    """Get MCP and A2A events for a specific execution"""
    # In production, this would query a database
    return {
        "execution_id": execution_id,
        "mcp_events": [],
        "a2a_events": [],
        "status": "completed"
    }

@app.get("/api/agents/requirements/documents")
async def get_processed_documents():
    """Get list of all processed documents"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    requirements_agent = orchestrator.agents["requirements"]
    
    documents = []
    for doc in requirements_agent.processed_documents:
        documents.append({
            "filename": doc.filename,
            "file_type": doc.file_type,
            "source": doc.source,
            "content_length": len(doc.content),
            "metadata": doc.metadata
        })
    
    return {
        "documents": documents,
        "total": len(documents),
        "by_source": {
            "upload": len([d for d in requirements_agent.processed_documents if d.source == "upload"]),
            "confluence": len([d for d in requirements_agent.processed_documents if d.source == "confluence"]),
            "jira": len([d for d in requirements_agent.processed_documents if d.source == "jira"]),
            "other": len([d for d in requirements_agent.processed_documents if d.source not in ["upload", "confluence", "jira"]])
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            elif message_data.get("type") == "subscribe_agent":
                agent_id = message_data.get("agent_id")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "agent_id": agent_id,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
            elif message_data.get("type") == "get_status":
                # Send current status
                await manager.send_personal_message(
                    json.dumps({
                        "type": "status_update",
                        "data": {
                            "uploaded_files": len(uploaded_files_storage),
                            "agent_state": "idle",
                            "timestamp": datetime.now().isoformat()
                        }
                    }),
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Health and monitoring endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "agents": list(orchestrator.agents.keys()) if orchestrator else [],
        "uploaded_files": len(uploaded_files_storage),
        "temp_dir_exists": temp_files_dir.exists() if temp_files_dir else False
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    if not orchestrator:
        return {"error": "Orchestrator not initialized"}
    
    requirements_agent = orchestrator.agents.get("requirements")
    integrations_connected = 0
    
    if requirements_agent:
        integrations_connected = sum(
            1 for integration in requirements_agent.integration_manager.integrations.values()
            if integration.is_connected
        )
    
    return {
        "system": {
            "status": "healthy",
            "uptime": "24h 30m",  # Would calculate actual uptime
            "memory_usage": "256MB",
            "cpu_usage": "15%"
        },
        "agents": {
            "requirements_agent": {
                "status": requirements_agent.state.value if requirements_agent else "not_found",
                "documents_processed": len(requirements_agent.processed_documents) if requirements_agent else 0,
                "integrations_connected": integrations_connected,
                "success_rate": 0.95,
                "avg_execution_time": 45.2
            }
        },
        "storage": {
            "uploaded_files": len(uploaded_files_storage),
            "temp_dir_size": sum(f.stat().st_size for f in temp_files_dir.rglob('*') if f.is_file()) if temp_files_dir else 0
        },
        "integrations": {
            "confluence": {"status": "available"},
            "jira": {"status": "available"},
            "sharepoint": {"status": "available"},
            "googledrive": {"status": "available"},
            "onedrive": {"status": "available"}
        }
    }

# Simulation endpoints for testing
@app.post("/api/simulate/agent/execution")
async def simulate_agent_execution(background_tasks: BackgroundTasks):
    """Simulate agent execution for testing purposes"""
    
    async def simulate_execution():
        """Background task to simulate agent execution with real-time updates"""
        phases = [
            {"phase": "reasoning", "duration": 3, "message": "Analyzing uploaded documents and integrations..."},
            {"phase": "planning", "duration": 2, "message": "Creating comprehensive analysis plan..."},
            {"phase": "acting", "duration": 4, "message": "Executing requirements extraction and validation..."}
        ]
        
        for phase in phases:
            await manager.broadcast(json.dumps({
                "type": "agent_phase_update",
                "data": {
                    "phase": phase["phase"],
                    "message": phase["message"],
                    "timestamp": datetime.now().isoformat()
                }
            }))
            
            await asyncio.sleep(phase["duration"])
        
        # Final completion message
        await manager.broadcast(json.dumps({
            "type": "agent_execution_complete",
            "data": {
                "status": "completed",
                "message": "Requirements analysis completed successfully",
                "documents_processed": len(uploaded_files_storage),
                "timestamp": datetime.now().isoformat()
            }
        }))
    
    background_tasks.add_task(simulate_execution)
    return {"status": "simulation_started", "message": "Agent execution simulation initiated"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status_code": 500}

# Static files for frontend (if serving from same container)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )# main.py - FastAPI server for Requirements Agent
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
from requirements_agent import AgentOrchestrator, AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI SDLC - Requirements Agent",
    description="Autonomous Requirements Engineering Agent with Reason-Plan-Act Architecture",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TaskRequest(BaseModel):
    type: str
    project_name: str
    description: str
    stakeholders: List[str]
    business_goals: List[str]
    additional_context: Optional[Dict] = {}

class AgentResponse(BaseModel):
    execution_id: str
    status: str
    agent_id: str
    result: Dict
    mcp_events: List[Dict]
    a2a_events: List[Dict]
    execution_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    agents: List[str]

# Global variables
orchestrator = None
active_connections: List[WebSocket] = []

# Initialize orchestrator
@app.on_event("startup")
async def startup_event():
    global orchestrator
    
    llm_config = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-4"
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY", "")
        },
        "azure": {
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_key": os.getenv("AZURE_OPENAI_KEY", "")
        }
    }
    
    orchestrator = AgentOrchestrator(llm_config)
    logger.info("Requirements Agent Orchestrator initialized")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

manager = ConnectionManager()

# API Routes

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service information"""
    return HealthResponse(
        status="running",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        agents=["requirements_agent"]
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        agents=list(orchestrator.agents.keys()) if orchestrator else []
    )

@app.post("/api/agents/requirements/execute", response_model=AgentResponse)
async def execute_requirements_agent(task: TaskRequest, background_tasks: BackgroundTasks):
    """Execute the requirements agent with given task"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        start_time = datetime.now()
        
        # Convert task to dict
        task_dict = task.dict()
        
        # Create context
        context = AgentContext(
            project_id=f"proj_{int(start_time.timestamp())}",
            session_id=f"sess_{int(start_time.timestamp())}",
            current_task=task_dict,
            conversation_history=[],
            shared_memory={},
            available_tools=list(orchestrator.tool_registry.tools.keys())
        )
        
        # Execute agent
        result = await orchestrator.execute_agent_task("requirements", task_dict, context)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        response = AgentResponse(
            execution_id=result["execution_id"],
            status=result["status"],
            agent_id="requirements_agent",
            result=result["result"],
            mcp_events=result["mcp_events"],
            a2a_events=result["a2a_events"],
            execution_time=execution_time
        )
        
        # Broadcast to connected WebSocket clients
        background_tasks.add_task(
            manager.broadcast, 
            json.dumps({
                "type": "agent_execution_complete",
                "data": response.dict()
            })
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error executing requirements agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/requirements/capabilities")
async def get_agent_capabilities():
    """Get requirements agent capabilities"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    agent = orchestrator.agents.get("requirements")
    if not agent:
        raise HTTPException(status_code=404, detail="Requirements agent not found")
    
    return {
        "agent_id": agent.agent_id,
        "capabilities": [
            {
                "name": cap.name,
                "description": cap.description,
                "input_schema": cap.input_schema,
                "output_schema": cap.output_schema,
                "tools": cap.tools
            }
            for cap in agent.capabilities
        ],
        "state": agent.state.value,
        "available_tools": list(orchestrator.tool_registry.tools.keys())
    }

@app.get("/api/tools")
async def get_available_tools():
    """Get all available tools for integration"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    return {
        "tools": orchestrator.tool_registry.tools,
        "count": len(orchestrator.tool_registry.tools)
    }

@app.post("/api/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, action: str, parameters: Dict):
    """Execute a specific tool action"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        result = await orchestrator.tool_registry.execute_tool(tool_name, action, parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/agents/requirements/events/{execution_id}")
async def get_execution_events(execution_id: str):
    """Get MCP and A2A events for a specific execution"""
    # In production, this would query a database
    # For now, return mock data structure
    return {
        "execution_id": execution_id,
        "mcp_events": [],
        "a2a_events": [],
        "status": "completed"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            elif message_data.get("type") == "subscribe_agent":
                agent_id = message_data.get("agent_id")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "agent_id": agent_id,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Simulation endpoints for testing
@app.post("/api/simulate/agent/execution")
async def simulate_agent_execution(background_tasks: BackgroundTasks):
    """Simulate agent execution for testing purposes"""
    
    async def simulate_execution():
        """Background task to simulate agent execution with real-time updates"""
        phases = [
            {"phase": "reasoning", "duration": 3, "message": "Analyzing requirements..."},
            {"phase": "planning", "duration": 2, "message": "Creating execution plan..."},
            {"phase": "acting", "duration": 4, "message": "Executing requirements gathering..."}
        ]
        
        for phase in phases:
            await manager.broadcast(json.dumps({
                "type": "agent_phase_update",
                "data": {
                    "phase": phase["phase"],
                    "message": phase["message"],
                    "timestamp": datetime.now().isoformat()
                }
            }))
            
            await asyncio.sleep(phase["duration"])
        
        # Final completion message
        await manager.broadcast(json.dumps({
            "type": "agent_execution_complete",
            "data": {
                "status": "completed",
                "message": "Requirements gathering completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        }))
    
    background_tasks.add_task(simulate_execution)
    return {"status": "simulation_started", "message": "Agent execution simulation initiated"}

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    return {
        "system": {
            "status": "healthy",
            "uptime": "24h 30m",
            "memory_usage": "256MB",
            "cpu_usage": "15%"
        },
        "agents": {
            "requirements_agent": {
                "status": "idle",
                "executions_count": 42,
                "success_rate": 0.95,
                "avg_execution_time": 45.2
            }
        },
        "integrations": {
            "openai": {"status": "connected", "requests": 150},
            "redis": {"status": "connected", "memory": "64MB"},
            "neo4j": {"status": "connected", "nodes": 1250}
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

# Static files for frontend (if serving from same container)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )