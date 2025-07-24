# Core Agent Framework - Agentic SDLC Foundation
# This is the foundational framework that all SDLC agents will inherit from

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime

# LangChain imports for agent framework
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import AzureOpenAI

# MCP Protocol Support
from mcp import Client, Server, types
from mcp.server.fastapi import FastMCPServer

# A2A Protocol Support
import httpx
from pydantic import BaseModel

class AgentState(Enum):
    IDLE = "idle"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

class LLMProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"

@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Dict
    output_schema: Dict
    tools: List[str]
    mcp_compatible: bool = True
    a2a_compatible: bool = True

@dataclass
class AgentContext:
    project_id: str
    session_id: str
    workflow_id: str
    current_task: Optional[Dict]
    conversation_history: List[BaseMessage] = field(default_factory=list)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    available_tools: List[str] = field(default_factory=list)
    mcp_servers: List[str] = field(default_factory=list)
    a2a_agents: List[str] = field(default_factory=list)

@dataclass
class AgentConfiguration:
    agent_id: str
    agent_type: str
    llm_provider: LLMProvider
    llm_model: str
    temperature: float = 0.1
    max_tokens: int = 4000
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    enable_mcp: bool = True
    enable_a2a: bool = True
    tools_config: Dict = field(default_factory=dict)

class MCPProtocolHandler:
    """Handles MCP protocol communication for agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.mcp_server = FastMCPServer("Agentic-SDLC")
        self.mcp_clients = {}
        
    async def register_tool(self, tool_name: str, tool_func: Callable):
        """Register a tool with MCP server"""
        @self.mcp_server.call_tool()
        async def execute_tool(name: str, arguments: dict) -> Any:
            if name == tool_name:
                return await tool_func(arguments)
            raise ValueError(f"Unknown tool: {name}")
    
    async def connect_to_mcp_server(self, server_url: str):
        """Connect to external MCP server"""
        client = Client()
        await client.connect(server_url)
        self.mcp_clients[server_url] = client
        return client
    
    async def call_mcp_tool(self, server_url: str, tool_name: str, arguments: Dict):
        """Call tool on external MCP server"""
        if server_url not in self.mcp_clients:
            await self.connect_to_mcp_server(server_url)
        
        client = self.mcp_clients[server_url]
        return await client.call_tool(tool_name, arguments)

class A2AProtocolHandler:
    """Handles A2A protocol communication for agents"""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.agent_card = self._create_agent_card()
        
    def _create_agent_card(self) -> Dict:
        """Create A2A Agent Card"""
        return {
            "id": self.agent_id,
            "name": f"SDLC Agent - {self.agent_id}",
            "version": "1.0.0",
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_schema": cap.input_schema,
                    "output_schema": cap.output_schema
                } for cap in self.capabilities
            ],
            "endpoints": {
                "execute": f"/agents/{self.agent_id}/execute",
                "status": f"/agents/{self.agent_id}/status",
                "capabilities": f"/agents/{self.agent_id}/capabilities"
            },
            "auth": {
                "type": "bearer",
                "required": True
            }
        }
    
    async def discover_agents(self, registry_url: str) -> List[Dict]:
        """Discover other A2A agents"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{registry_url}/agents")
            return response.json()
    
    async def delegate_task(self, target_agent_url: str, task: Dict) -> Dict:
        """Delegate task to another A2A agent"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{target_agent_url}/execute",
                json=task,
                headers={"Authorization": f"Bearer {self._get_auth_token()}"}
            )
            return response.json()

class LLMManager:
    """Manages different LLM providers"""
    
    def __init__(self, config: AgentConfiguration):
        self.config = config
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.config.llm_provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key
            )
        elif self.config.llm_provider == LLMProvider.GOOGLE:
            return ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                google_api_key=self.config.api_key
            )
        elif self.config.llm_provider == LLMProvider.AZURE_OPENAI:
            return AzureOpenAI(
                deployment_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                azure_endpoint=self.config.api_base,
                api_key=self.config.api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

class BaseSDLCAgent(ABC):
    """Base class for all SDLC agents with Reason-Plan-Act architecture"""
    
    def __init__(self, config: AgentConfiguration, capabilities: List[AgentCapability]):
        self.config = config
        self.capabilities = capabilities
        self.agent_id = config.agent_id
        self.state = AgentState.IDLE
        self.context: Optional[AgentContext] = None
        
        # Initialize core components
        self.llm_manager = LLMManager(config)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Protocol handlers
        if config.enable_mcp:
            self.mcp_handler = MCPProtocolHandler(self.agent_id)
        if config.enable_a2a:
            self.a2a_handler = A2AProtocolHandler(self.agent_id, capabilities)
            
        # Logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.execution_log = []
        
    def log_execution(self, stage: str, data: Dict):
        """Log execution details for monitoring"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "stage": stage,
            "state": self.state.value,
            "data": data
        }
        self.execution_log.append(log_entry)
        self.logger.info(f"[{stage}] {json.dumps(data)}")
    
    @abstractmethod
    async def reason(self, input_data: Dict) -> Dict:
        """
        Reasoning Phase: Analyze the situation and understand what needs to be done
        This should use the LLM to understand the context and requirements
        """
        pass
        
    @abstractmethod
    async def plan(self, reasoning_output: Dict) -> Dict:
        """
        Planning Phase: Create a step-by-step plan to achieve the goal
        This should break down the work into actionable steps
        """
        pass
        
    @abstractmethod  
    async def act(self, plan: Dict) -> Dict:
        """
        Action Phase: Execute the plan and perform actions
        This should execute the planned steps using available tools
        """
        pass
    
    async def process(self, task: Dict, context: AgentContext) -> Dict:
        """
        Main processing loop: Reason -> Plan -> Act
        This is the core agent execution method
        """
        self.context = context
        start_time = time.time()
        
        try:
            # Reasoning Phase
            self.state = AgentState.REASONING
            self.log_execution("reasoning_start", {"task": task})
            
            reasoning_result = await self.reason(task)
            self.log_execution("reasoning_complete", reasoning_result)
            
            # Planning Phase
            self.state = AgentState.PLANNING
            self.log_execution("planning_start", {"reasoning": reasoning_result})
            
            plan = await self.plan(reasoning_result)
            self.log_execution("planning_complete", plan)
            
            # Action Phase
            self.state = AgentState.ACTING
            self.log_execution("acting_start", {"plan": plan})
            
            result = await self.act(plan)
            self.log_execution("acting_complete", result)
            
            # Complete
            self.state = AgentState.COMPLETED
            execution_time = time.time() - start_time
            
            final_result = {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "reasoning": reasoning_result,
                "plan": plan,
                "logs": self.execution_log[-10:]  # Last 10 log entries
            }
            
            self.log_execution("process_complete", final_result)
            return final_result
            
        except Exception as e:
            self.state = AgentState.ERROR
            error_result = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "logs": self.execution_log[-10:]
            }
            
            self.log_execution("process_error", error_result)
            raise AgentError(f"Agent {self.agent_id} failed: {str(e)}")
    
    async def get_agent_status(self) -> Dict:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "capabilities": [cap.name for cap in self.capabilities],
            "mcp_enabled": self.config.enable_mcp,
            "a2a_enabled": self.config.enable_a2a,
            "last_execution": self.execution_log[-1] if self.execution_log else None
        }
    
    async def get_execution_logs(self, limit: int = 50) -> List[Dict]:
        """Get execution logs for debugging"""
        return self.execution_log[-limit:]
    
    async def use_mcp_tool(self, server_url: str, tool_name: str, arguments: Dict):
        """Use MCP tool if available"""
        if not self.config.enable_mcp:
            raise ValueError("MCP not enabled for this agent")
        return await self.mcp_handler.call_mcp_tool(server_url, tool_name, arguments)
    
    async def delegate_to_agent(self, target_agent_url: str, task: Dict):
        """Delegate task to another agent via A2A"""
        if not self.config.enable_a2a:
            raise ValueError("A2A not enabled for this agent")
        return await self.a2a_handler.delegate_task(target_agent_url, task)

class AgentError(Exception):
    """Custom exception for agent errors"""
    pass

class AgentFactory:
    """Factory for creating and configuring agents"""
    
    @staticmethod
    def create_agent(agent_type: str, config: AgentConfiguration) -> BaseSDLCAgent:
        """Create an agent of specified type"""
        # Import agent classes (will be implemented in separate modules)
        from agents.requirements_agent import RequirementsAgent
        from agents.design_agent import DesignAgent
        from agents.code_agent import CodeGenerationAgent
        from agents.test_agent import TestingAgent
        from agents.deployment_agent import DeploymentAgent
        from agents.monitoring_agent import MonitoringAgent
        
        agent_classes = {
            "requirements": RequirementsAgent,
            "design": DesignAgent,
            "code_generation": CodeGenerationAgent,  
            "testing": TestingAgent,
            "deployment": DeploymentAgent,
            "monitoring": MonitoringAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return agent_classes[agent_type](config)

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = AgentConfiguration(
        agent_id="test_agent_001",
        agent_type="requirements",
        llm_provider=LLMProvider.OPENAI,
        llm_model="gpt-4",
        api_key="your-openai-api-key"
    )
    
    # This would be implemented by specific agent classes
    class TestAgent(BaseSDLCAgent):
        async def reason(self, input_data: Dict) -> Dict:
            return {"understood": True, "complexity": "medium"}
            
        async def plan(self, reasoning_output: Dict) -> Dict:
            return {"steps": ["step1", "step2"], "estimated_time": 300}
            
        async def act(self, plan: Dict) -> Dict:
            return {"completed_steps": plan["steps"], "result": "success"}
    
    # Create and test agent
    capabilities = [
        AgentCapability(
            name="test_capability",
            description="Test capability",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            tools=["test_tool"]
        )
    ]
    
    agent = TestAgent(config, capabilities)
    
    # Test context
    context = AgentContext(
        project_id="test_project",
        session_id="test_session",
        workflow_id="test_workflow"
    )
    
    print("Core Agent Framework initialized successfully!")
    print(f"Agent ID: {agent.agent_id}")
    print(f"State: {agent.state}")
    print(f"MCP Enabled: {agent.config.enable_mcp}")
    print(f"A2A Enabled: {agent.config.enable_a2a}")
