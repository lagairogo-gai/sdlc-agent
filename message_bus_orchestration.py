# Message Bus & Orchestration Engine
# Handles inter-agent communication and workflow orchestration

import asyncio
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Core agent framework imports
from core_agent_framework import BaseSDLCAgent, AgentContext, AgentState

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    AGENT_STATUS = "agent_status"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

class WorkflowState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed" 
    CANCELLED = "cancelled"

@dataclass
class Message:
    id: str
    type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

@dataclass
class WorkflowStep:
    id: str
    agent_type: str
    agent_id: str
    task: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Dict] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

@dataclass
class WorkflowDefinition:
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    state: WorkflowState = WorkflowState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class MessageBus:
    """Redis-based message bus for agent communication"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        
    async def start(self):
        """Initialize Redis connection and start message processing"""
        self.redis_client = redis.from_url(self.redis_url)
        self.running = True
        
        # Start message processing tasks
        asyncio.create_task(self._process_messages())
        logging.info("Message bus started")
        
    async def stop(self):
        """Stop message bus"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        logging.info("Message bus stopped")
        
    async def publish(self, message: Message):
        """Publish message to the bus"""
        if not self.redis_client:
            raise RuntimeError("Message bus not started")
            
        # Serialize message
        message_data = {
            "id": message.id,
            "type": message.type.value,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "correlation_id": message.correlation_id,
            "reply_to": message.reply_to
        }
        
        # Publish to Redis
        channel = f"agent:{message.recipient_id}" if message.recipient_id else "broadcast"
        await self.redis_client.publish(channel, json.dumps(message_data))
        
        # Store in message history
        await self.redis_client.lpush(
            f"messages:{message.sender_id}",
            json.dumps(message_data)
        )
        await self.redis_client.ltrim(f"messages:{message.sender_id}", 0, 999)  # Keep last 1000
        
    async def subscribe(self, agent_id: str, handler: Callable[[Message], None]):
        """Subscribe agent to messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(handler)
        
        # Subscribe to Redis channel
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(f"agent:{agent_id}", "broadcast")
        
        # Process subscribed messages
        async def process_subscription():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        msg_data = json.loads(message["data"])
                        msg = Message(
                            id=msg_data["id"],
                            type=MessageType(msg_data["type"]),
                            sender_id=msg_data["sender_id"],
                            recipient_id=msg_data.get("recipient_id"),
                            payload=msg_data["payload"],
                            timestamp=msg_data["timestamp"],
                            correlation_id=msg_data.get("correlation_id"),
                            reply_to=msg_data.get("reply_to")
                        )
                        
                        # Call handlers
                        for handler in self.subscribers.get(agent_id, []):
                            await handler(msg)
                            
                    except Exception as e:
                        logging.error(f"Error processing message for {agent_id}: {e}")
        
        asyncio.create_task(process_subscription())
        
    async def _process_messages(self):
        """Background task to process messages"""
        while self.running:
            try:
                # Process any queued messages
                await asyncio.sleep(0.1)
            except Exception as e:
                logging.error(f"Error in message processing: {e}")

class AgentRegistry:
    """Registry for managing available agents"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.agents: Dict[str, Dict] = {}
        
    async def register_agent(self, agent: BaseSDLCAgent):
        """Register an agent"""
        agent_info = {
            "agent_id": agent.agent_id,
            "agent_type": agent.config.agent_type,
            "capabilities": [cap.name for cap in agent.capabilities],
            "status": agent.state.value,
            "registered_at": time.time(),
            "last_heartbeat": time.time()
        }
        
        self.agents[agent.agent_id] = agent_info
        
        # Store in Redis
        await self.redis_client.hset(
            "agents",
            agent.agent_id,
            json.dumps(agent_info)
        )
        
        logging.info(f"Agent registered: {agent.agent_id}")
        
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
        await self.redis_client.hdel("agents", agent_id)
        logging.info(f"Agent unregistered: {agent_id}")
        
    async def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent info"""
        if agent_id in self.agents:
            return self.agents[agent_id]
            
        # Try Redis
        data = await self.redis_client.hget("agents", agent_id)
        if data:
            return json.loads(data)
        return None
        
    async def list_agents(self, agent_type: Optional[str] = None) -> List[Dict]:
        """List all agents or agents of specific type"""
        agents = []
        
        # Get from Redis
        all_agents = await self.redis_client.hgetall("agents")
        for agent_id, agent_data in all_agents.items():
            agent_info = json.loads(agent_data)
            if not agent_type or agent_info["agent_type"] == agent_type:
                agents.append(agent_info)
                
        return agents
        
    async def update_heartbeat(self, agent_id: str):
        """Update agent heartbeat"""
        agent_info = await self.get_agent(agent_id)
        if agent_info:
            agent_info["last_heartbeat"] = time.time()
            await self.redis_client.hset(
                "agents",
                agent_id,
                json.dumps(agent_info)
            )

class WorkflowEngine:
    """Orchestrates multi-agent workflows"""
    
    def __init__(self, message_bus: MessageBus, agent_registry: AgentRegistry):
        self.message_bus = message_bus
        self.agent_registry = agent_registry
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        
    def create_sdlc_workflow(self, project_id: str, requirements: Dict) -> WorkflowDefinition:
        """Create a complete SDLC workflow"""
        workflow_id = f"sdlc_{project_id}_{int(time.time())}"
        
        steps = [
            WorkflowStep(
                id="requirements",
                agent_type="requirements",
                agent_id="requirements_agent",
                task={
                    "type": "gather_requirements",
                    "project_id": project_id,
                    "input": requirements
                }
            ),
            WorkflowStep(
                id="design",
                agent_type="design",
                agent_id="design_agent",
                task={
                    "type": "create_design",
                    "project_id": project_id
                },
                dependencies=["requirements"]
            ),
            WorkflowStep(
                id="code_generation",
                agent_type="code_generation",
                agent_id="code_agent",
                task={
                    "type": "generate_code",
                    "project_id": project_id
                },
                dependencies=["design"]
            ),
            WorkflowStep(
                id="code_review",
                agent_type="code_review",
                agent_id="review_agent",
                task={
                    "type": "review_code",
                    "project_id": project_id
                },
                dependencies=["code_generation"]
            ),
            WorkflowStep(
                id="testing",
                agent_type="testing",
                agent_id="test_agent",
                task={
                    "type": "run_tests",
                    "project_id": project_id
                },
                dependencies=["code_review"]
            ),
            WorkflowStep(
                id="deployment",
                agent_type="deployment",
                agent_id="deploy_agent",
                task={
                    "type": "deploy_application",
                    "project_id": project_id
                },
                dependencies=["testing"]
            ),
            WorkflowStep(
                id="monitoring",
                agent_type="monitoring",
                agent_id="monitor_agent",
                task={
                    "type": "setup_monitoring",
                    "project_id": project_id
                },
                dependencies=["deployment"]
            )
        ]
        
        workflow = WorkflowDefinition(
            id=workflow_id,
            name=f"SDLC Workflow - {project_id}",
            description="Complete software development lifecycle workflow",
            steps=steps
        )
        
        self.workflows[workflow_id] = workflow
        return workflow
        
    async def execute_workflow(self, workflow_id: str) -> Dict:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        workflow.state = WorkflowState.RUNNING
        workflow.started_at = time.time()
        
        # Create execution task
        task = asyncio.create_task(self._execute_workflow_steps(workflow))
        self.running_workflows[workflow_id] = task
        
        try:
            result = await task
            workflow.state = WorkflowState.COMPLETED
            workflow.completed_at = time.time()
            return result
        except Exception as e:
            workflow.state = WorkflowState.FAILED
            logging.error(f"Workflow {workflow_id} failed: {e}")
            raise
        finally:
            if workflow_id in self.running_workflows:
                del self.running_workflows[workflow_id]
                
    async def _execute_workflow_steps(self, workflow: WorkflowDefinition) -> Dict:
        """Execute workflow steps in dependency order"""
        completed_steps = set()
        results = {}
        
        while len(completed_steps) < len(workflow.steps):
            # Find steps ready to execute
            ready_steps = [
                step for step in workflow.steps
                if step.status == "pending" and
                all(dep in completed_steps for dep in step.dependencies)
            ]
            
            if not ready_steps:
                # Check if we're stuck
                pending_steps = [s for s in workflow.steps if s.status == "pending"]
                if pending_steps:
                    raise RuntimeError("Workflow deadlock detected")
                break
                
            # Execute ready steps in parallel
            tasks = []
            for step in ready_steps:
                task = asyncio.create_task(self._execute_step(step, results))
                tasks.append((step, task))
                
            # Wait for steps to complete
            for step, task in tasks:
                try:
                    step_result = await task
                    step.status = "completed"
                    step.result = step_result
                    step.end_time = time.time()
                    results[step.id] = step_result
                    completed_steps.add(step.id)
                    
                    # Notify workflow progress
                    await self._notify_workflow_progress(workflow, step)
                    
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    step.end_time = time.time()
                    logging.error(f"Step {step.id} failed: {e}")
                    raise
        
        return results
        
    async def _execute_step(self, step: WorkflowStep, previous_results: Dict) -> Dict:
        """Execute a single workflow step"""
        step.status = "running"
        step.start_time = time.time()
        
        # Find available agent
        agents = await self.agent_registry.list_agents(step.agent_type)
        if not agents:
            raise RuntimeError(f"No agents available for type: {step.agent_type}")
            
        # Use first available agent (could implement load balancing here)
        selected_agent = agents[0]
        
        # Prepare task with context from previous steps
        enhanced_task = {
            **step.task,
            "previous_results": previous_results,
            "step_id": step.id
        }
        
        # Send task to agent
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_REQUEST,
            sender_id="orchestrator",
            recipient_id=selected_agent["agent_id"],
            payload=enhanced_task,
            timestamp=time.time(),
            correlation_id=step.id
        )
        
        await self.message_bus.publish(message)
        
        # Wait for response (with timeout)
        response = await self._wait_for_response(message.id, timeout=300)  # 5 minutes
        
        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Agent error: {response.payload}")
            
        return response.payload
        
    async def _wait_for_response(self, message_id: str, timeout: int = 300) -> Message:
        """Wait for response to a message"""
        # Implementation would use Redis blocking operations or WebSocket
        # For now, simplified version
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check for response in Redis
            # This is a simplified implementation
            await asyncio.sleep(1)
        
        raise TimeoutError(f"No response received for message {message_id}")
        
    async def _notify_workflow_progress(self, workflow: WorkflowDefinition, completed_step: WorkflowStep):
        """Notify about workflow progress"""
        progress_message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.WORKFLOW_START,
            sender_id="orchestrator",
            recipient_id=None,  # Broadcast
            payload={
                "workflow_id": workflow.id,
                "completed_step": completed_step.id,
                "progress": len([s for s in workflow.steps if s.status == "completed"]) / len(workflow.steps),
                "status": workflow.state.value
            },
            timestamp=time.time()
        )
        
        await self.message_bus.publish(progress_message)
        
    async def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get workflow status"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "state": workflow.state.value,
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "steps": [
                {
                    "id": step.id,
                    "agent_type": step.agent_type,
                    "status": step.status,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "error": step.error
                } for step in workflow.steps
            ]
        }

class OrchestrationAPI:
    """FastAPI app for orchestration management"""
    
    def __init__(self, message_bus: MessageBus, workflow_engine: WorkflowEngine, agent_registry: AgentRegistry):
        self.message_bus = message_bus
        self.workflow_engine = workflow_engine
        self.agent_registry = agent_registry
        self.app = FastAPI(title="Agentic SDLC Orchestrator")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/workflows")
        async def create_workflow(request: Dict):
            """Create new SDLC workflow"""
            try:
                workflow = self.workflow_engine.create_sdlc_workflow(
                    project_id=request["project_id"],
                    requirements=request["requirements"]
                )
                return {"workflow_id": workflow.id, "status": "created"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
                
        @self.app.post("/workflows/{workflow_id}/execute")
        async def execute_workflow(workflow_id: str):
            """Execute workflow"""
            try:
                # Execute workflow in background
                asyncio.create_task(self.workflow_engine.execute_workflow(workflow_id))
                return {"workflow_id": workflow_id, "status": "started"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
                
        @self.app.get("/workflows/{workflow_id}/status")
        async def get_workflow_status(workflow_id: str):
            """Get workflow status"""
            try:
                return await self.workflow_engine.get_workflow_status(workflow_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
                
        @self.app.get("/agents")
        async def list_agents():
            """List all agents"""
            return await self.agent_registry.list_agents()
            
        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str):
            """Get agent details"""
            agent = await self.agent_registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            return agent
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            
            # Subscribe to workflow updates
            async def handle_message(message: Message):
                if message.type in [MessageType.WORKFLOW_START, MessageType.AGENT_STATUS]:
                    await websocket.send_json({
                        "type": message.type.value,
                        "payload": message.payload,
                        "timestamp": message.timestamp
                    })
            
            await self.message_bus.subscribe("orchestrator", handle_message)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except Exception as e:
                logging.info(f"WebSocket connection closed: {e}")

class OrchestrationSystem:
    """Main orchestration system that ties everything together"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.message_bus = MessageBus(redis_url)
        self.agent_registry = None
        self.workflow_engine = None
        self.api = None
        
    async def start(self):
        """Start the orchestration system"""
        # Start message bus
        await self.message_bus.start()
        
        # Initialize registry
        redis_client = redis.from_url(self.redis_url)
        self.agent_registry = AgentRegistry(redis_client)
        
        # Initialize workflow engine
        self.workflow_engine = WorkflowEngine(self.message_bus, self.agent_registry)
        
        # Setup API
        self.api = OrchestrationAPI(self.message_bus, self.workflow_engine, self.agent_registry)
        
        logging.info("Orchestration system started")
        
    async def stop(self):
        """Stop the orchestration system"""
        await self.message_bus.stop()
        logging.info("Orchestration system stopped")
        
    def get_app(self):
        """Get FastAPI app for deployment"""
        return self.api.app

# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Initialize orchestration system
        orchestrator = OrchestrationSystem()
        await orchestrator.start()
        
        # Example: Create and execute a workflow
        workflow = orchestrator.workflow_engine.create_sdlc_workflow(
            project_id="test_project_001",
            requirements={
                "title": "E-commerce Website",
                "description": "Build a modern e-commerce platform",
                "features": ["user auth", "product catalog", "shopping cart", "payments"]
            }
        )
        
        print(f"Created workflow: {workflow.id}")
        print(f"Steps: {[step.id for step in workflow.steps]}")
        
        # Get workflow status
        status = await orchestrator.workflow_engine.get_workflow_status(workflow.id)
        print(f"Workflow status: {status}")
        
        # Start API server
        app = orchestrator.get_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    # Run the example
    asyncio.run(main())