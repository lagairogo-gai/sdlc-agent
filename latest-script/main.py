"""
COMPLETE AI Monitoring System v5 - SINGLE CLEAN VERSION
Model Context Protocol + Agent-to-Agent Communication + Business Intelligence + Detailed Logging
NO DUPLICATES - CLEAN IMPLEMENTATION
"""
import os
import asyncio
import json
import time
import uuid
import logging
import sys
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / "app.log")
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONTEXT PROTOCOL (MCP) IMPLEMENTATION
# =============================================================================

@dataclass
class MCPContext:
    """Model Context Protocol - Shared context between agents"""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str = ""
    context_type: str = "incident_analysis"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Context data
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    agent_insights: Dict[str, Any] = field(default_factory=dict)
    correlation_patterns: List[Dict[str, Any]] = field(default_factory=list)
    learned_behaviors: Dict[str, Any] = field(default_factory=dict)
    
    # Context metadata
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    context_version: int = 1
    
    def update_context(self, agent_id: str, new_data: Dict[str, Any], confidence: float = 0.8):
        """Update context with new agent insights"""
        self.agent_insights[agent_id] = {
            "data": new_data,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence
        }
        self.confidence_scores[agent_id] = confidence
        self.updated_at = datetime.now()
        self.context_version += 1
        logger.info(f"🧠 MCP Context updated by {agent_id} - confidence: {confidence:.2f}")
    
    def get_contextual_insights(self, requesting_agent: str) -> Dict[str, Any]:
        """Get relevant context for requesting agent"""
        relevant_insights = {}
        
        for agent_id, insight in self.agent_insights.items():
            if agent_id != requesting_agent and insight["confidence"] > 0.7:
                relevant_insights[agent_id] = insight
        
        return {
            "shared_knowledge": self.shared_knowledge,
            "peer_insights": relevant_insights,
            "correlation_patterns": self.correlation_patterns,
            "context_confidence": sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0
        }

class MCPRegistry:
    """Registry for managing MCP contexts"""
    
    def __init__(self):
        self.contexts: Dict[str, MCPContext] = {}
        self.context_subscriptions: Dict[str, Set[str]] = {}
        self.update_callbacks: List = []
    
    def create_context(self, incident_id: str, context_type: str = "incident_analysis") -> MCPContext:
        context = MCPContext(incident_id=incident_id, context_type=context_type)
        self.contexts[context.context_id] = context
        logger.info(f"📋 Created MCP context {context.context_id} for incident {incident_id}")
        self._notify_updates()
        return context
    
    def get_context(self, context_id: str) -> Optional[MCPContext]:
        return self.contexts.get(context_id)
    
    def subscribe_agent(self, agent_id: str, context_id: str):
        if agent_id not in self.context_subscriptions:
            self.context_subscriptions[agent_id] = set()
        self.context_subscriptions[agent_id].add(context_id)
    
    def _notify_updates(self):
        """Notify all callbacks about MCP updates"""
        for callback in self.update_callbacks:
            try:
                asyncio.create_task(callback())
            except Exception as e:
                logger.error(f"MCP callback error: {e}")

# =============================================================================
# AGENT-TO-AGENT (A2A) PROTOCOL IMPLEMENTATION
# =============================================================================

@dataclass
class A2AMessage:
    """Agent-to-Agent Protocol Message"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent_id: str = ""
    receiver_agent_id: str = ""
    message_type: str = "info_request"
    content: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"
    created_at: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    correlation_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender_agent_id,
            "receiver": self.receiver_agent_id,
            "type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.created_at.isoformat(),
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id
        }

class A2AProtocol:
    """Agent-to-Agent Communication Protocol"""
    
    def __init__(self):
        self.message_queue: Dict[str, List[A2AMessage]] = {}
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
        self.message_history: List[A2AMessage] = []
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.update_callbacks: List = []
    
    def register_agent_capabilities(self, agent_id: str, capabilities: List[str]):
        self.agent_capabilities[agent_id] = capabilities
        logger.info(f"🤝 Registered A2A capabilities for {agent_id}: {capabilities}")
    
    def send_message(self, message: A2AMessage):
        if message.receiver_agent_id not in self.message_queue:
            self.message_queue[message.receiver_agent_id] = []
        
        self.message_queue[message.receiver_agent_id].append(message)
        self.message_history.append(message)
        logger.info(f"📨 A2A Message: {message.sender_agent_id} → {message.receiver_agent_id} [{message.message_type}]")
        self._notify_updates()
    
    def get_messages(self, agent_id: str) -> List[A2AMessage]:
        messages = self.message_queue.get(agent_id, [])
        self.message_queue[agent_id] = []
        return messages
    
    def initiate_collaboration(self, initiator: str, participants: List[str], task: str, context: Dict[str, Any]) -> str:
        collab_id = str(uuid.uuid4())
        self.active_collaborations[collab_id] = {
            "id": collab_id,
            "initiator": initiator,
            "participants": participants,
            "task": task,
            "context": context,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
        
        for participant in participants:
            if participant != initiator:
                message = A2AMessage(
                    sender_agent_id=initiator,
                    receiver_agent_id=participant,
                    message_type="collaboration_request",
                    content={
                        "collaboration_id": collab_id,
                        "task": task,
                        "context": context
                    },
                    requires_response=True,
                    correlation_id=collab_id
                )
                self.send_message(message)
        
        logger.info(f"🤝 Started A2A collaboration {collab_id}: {task}")
        self._notify_updates()
        return collab_id
    
    def _notify_updates(self):
        """Notify all callbacks about A2A updates"""
        for callback in self.update_callbacks:
            try:
                asyncio.create_task(callback())
            except Exception as e:
                logger.error(f"A2A callback error: {e}")

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    WAITING = "waiting"
    COLLABORATING = "collaborating"

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentExecution:
    agent_id: str
    agent_name: str
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str = ""
    status: AgentStatus = AgentStatus.IDLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    logs: List[Dict[str, Any]] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    duration_seconds: float = 0.0
    
    # MCP + A2A enhancements
    mcp_context_id: str = ""
    a2a_messages_sent: int = 0
    a2a_messages_received: int = 0
    collaboration_sessions: List[str] = field(default_factory=list)
    contextual_insights_used: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Incident:
    id: str = field(default_factory=lambda: f"INC-{int(time.time())}")
    title: str = ""
    description: str = ""
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "open"
    affected_systems: List[str] = field(default_factory=list)
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_status: str = "in_progress"
    current_agent: str = ""
    completed_agents: List[str] = field(default_factory=list)
    failed_agents: List[str] = field(default_factory=list)
    executions: Dict[str, AgentExecution] = field(default_factory=dict)
    root_cause: str = ""
    resolution: str = ""
    pagerduty_incident_id: str = ""
    servicenow_ticket_id: str = ""
    remediation_applied: List[str] = field(default_factory=list)
    incident_type: str = ""
    business_impact: str = ""
    
    # MCP + A2A enhancements
    mcp_context_id: str = ""
    a2a_collaborations: List[str] = field(default_factory=list)
    cross_agent_insights: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# BUSINESS-CENTRIC INCIDENT SCENARIOS
# =============================================================================

BUSINESS_INCIDENT_SCENARIOS = [
    {
        "title": "Order Processing Delays - Critical Payment Integration Issue",
        "description": "Orders are taking 15+ minutes to move from 'Placed' to 'Shipped' state. Payment validation service experiencing high latency causing order pipeline bottleneck.",
        "severity": "critical",
        "affected_systems": ["payment-gateway", "order-management-system", "inventory-service", "shipping-service"],
        "incident_type": "business_critical",
        "business_impact": "Revenue loss: $2,500/minute. Customer satisfaction degradation. Potential order cancellations.",
        "root_cause": "Payment validation service database connection pool exhaustion combined with legacy synchronous processing architecture"
    },
    {
        "title": "Sudden Drop in Order Volume - 78% Below Normal",
        "description": "Customer order volume has dropped by 78% in the last 2 hours compared to historical patterns. Only 156 orders vs expected 712 orders.",
        "severity": "critical",
        "affected_systems": ["e-commerce-frontend", "recommendation-engine", "search-service", "product-catalog"],
        "incident_type": "business_anomaly",
        "business_impact": "Critical revenue impact. Potential customer acquisition funnel breakdown.",
        "root_cause": "Product search indexing failure causing empty search results for 67% of product queries"
    },
    {
        "title": "Payment Failures Spike - Regional Payment Provider Issue",
        "description": "89% increase in failed payment transactions from EU region via Stripe payment provider. Affecting premium customer segment.",
        "severity": "high",
        "affected_systems": ["payment-gateway", "stripe-integration", "fraud-detection", "customer-billing"],
        "incident_type": "payment_critical",
        "business_impact": "Lost sales: $45,000/hour. Premium customer churn risk. Potential compliance issues.",
        "root_cause": "Stripe EU webhook endpoint SSL certificate validation failure combined with retry mechanism bug"
    },
    {
        "title": "Product Search Latency Crisis - 8.5s Average Response Time", 
        "description": "Users experiencing severe latency (8.5s avg) and 34% timeout rate while searching for products. Search conversion dropped 67%.",
        "severity": "critical",
        "affected_systems": ["elasticsearch-cluster", "product-search-api", "autocomplete-service", "cdn"],
        "incident_type": "performance_critical",
        "business_impact": "Search-driven revenue down 67%. User experience severely degraded. Bounce rate increased 340%.",
        "root_cause": "Elasticsearch cluster split-brain condition with corrupted search indices and insufficient memory allocation"
    },
    {
        "title": "Cart Abandonment Rate Spike - 89% vs Normal 23%",
        "description": "Cart abandonment has jumped from normal 23% to critical 89% in last 3 hours. Checkout funnel breaking down at payment step.",
        "severity": "high",
        "affected_systems": ["shopping-cart-service", "checkout-service", "payment-processor", "session-management"],
        "incident_type": "conversion_critical",
        "business_impact": "Potential revenue loss: $78,000. Customer acquisition cost waste. Checkout UX failure.",
        "root_cause": "Session timeout misconfiguration causing cart data loss combined with payment form validation JavaScript errors"
    },
    {
        "title": "Bot Attack Detection - Abnormal Traffic Pattern 1200% Surge",
        "description": "Sudden traffic surge (1200% increase) with abnormal behavior: no cart additions, repetitive product views, bypassing CAPTCHA.",
        "severity": "critical",  
        "affected_systems": ["web-application-firewall", "bot-detection", "cdn", "rate-limiting"],
        "incident_type": "security_business",
        "business_impact": "Infrastructure costs spiking. Legitimate user performance degraded. Potential scraping/fraud attempt.",
        "root_cause": "Coordinated bot attack from 247 IP addresses attempting product data scraping and price manipulation"
    },
    {
        "title": "Trading Platform Latency Spike - Order Execution Delays",
        "description": "Stock trading order execution experiencing 3.4s delays vs normal 0.15s. High-frequency trading clients affected during market hours.",
        "severity": "critical",
        "affected_systems": ["trading-engine", "market-data-feed", "order-management", "risk-engine"],
        "incident_type": "trading_critical",
        "business_impact": "Trading revenue loss: $125,000/minute. Regulatory compliance risk. Client SLA breaches.",
        "root_cause": "Market data feed buffer overflow causing processing backlog in trading engine queue system"
    }
]

# =============================================================================
# ENHANCED WORKFLOW ENGINE WITH DETAILED LOGGING
# =============================================================================

class EnhancedWorkflowEngine:
    """Enhanced Workflow Engine with MCP + A2A + Detailed Logging"""
    
    def __init__(self):
        self.active_incidents: Dict[str, Incident] = {}
        self.incident_history: List[Incident] = []
        self.agent_execution_history: Dict[str, List[AgentExecution]] = {
            "monitoring": [], "rca": [], "pager": [], "ticketing": [], 
            "email": [], "remediation": [], "validation": []
        }
        
        # MCP + A2A components
        self.mcp_registry = MCPRegistry()
        self.a2a_protocol = A2AProtocol()
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        self._register_agent_capabilities()
        self._setup_update_callbacks()
    
    def _register_agent_capabilities(self):
        """Register agent capabilities for A2A collaboration"""
        capabilities = {
            "monitoring": ["business_metrics_analysis", "anomaly_detection", "performance_monitoring", "transaction_analysis"],
            "rca": ["business_impact_analysis", "root_cause_investigation", "dependency_mapping", "failure_correlation"],
            "pager": ["business_stakeholder_escalation", "customer_impact_assessment", "sla_breach_notification"],
            "ticketing": ["business_priority_classification", "customer_impact_tracking", "sla_management"],
            "email": ["customer_communication", "business_stakeholder_updates", "executive_reporting"],
            "remediation": ["business_continuity_actions", "customer_impact_mitigation", "revenue_protection"],
            "validation": ["business_metrics_verification", "customer_experience_testing", "sla_compliance_check"]
        }
        
        for agent_id, agent_capabilities in capabilities.items():
            self.a2a_protocol.register_agent_capabilities(agent_id, agent_capabilities)
    
    def _setup_update_callbacks(self):
        """Setup callbacks for real-time updates"""
        self.mcp_registry.update_callbacks.append(self._broadcast_mcp_update)
        self.a2a_protocol.update_callbacks.append(self._broadcast_a2a_update)
    
    async def _broadcast_mcp_update(self):
        """Broadcast MCP updates to WebSocket clients"""
        if self.websocket_connections:
            update_data = {
                "type": "mcp_update",
                "timestamp": datetime.now().isoformat(),
                "total_contexts": len(self.mcp_registry.contexts)
            }
            
            for ws in self.websocket_connections.copy():
                try:
                    await ws.send_text(json.dumps(update_data))
                except:
                    self.websocket_connections.remove(ws)
    
    async def _broadcast_a2a_update(self):
        """Broadcast A2A updates to WebSocket clients"""
        if self.websocket_connections:
            update_data = {
                "type": "a2a_update",
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(self.a2a_protocol.message_history),
                "active_collaborations": len(self.a2a_protocol.active_collaborations)
            }
            
            for ws in self.websocket_connections.copy():
                try:
                    await ws.send_text(json.dumps(update_data))
                except:
                    self.websocket_connections.remove(ws)
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add WebSocket connection"""
        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.websocket_connections)}")
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.websocket_connections)}")
    
    async def trigger_incident_workflow(self, incident_data: Dict[str, Any]) -> Incident:
        """Trigger incident workflow with business scenarios"""
        scenario = random.choice(BUSINESS_INCIDENT_SCENARIOS)
        incident = Incident(
            title=scenario["title"],
            description=scenario["description"],
            severity=IncidentSeverity(scenario["severity"]),
            affected_systems=scenario["affected_systems"],
            incident_type=scenario["incident_type"],
            business_impact=scenario["business_impact"]
        )
        
        # Create MCP context
        mcp_context = self.mcp_registry.create_context(incident.id, "business_incident_analysis")
        incident.mcp_context_id = mcp_context.context_id
        
        # Set initial shared knowledge
        mcp_context.shared_knowledge.update({
            "incident_metadata": {
                "id": incident.id,
                "type": incident.incident_type,
                "severity": incident.severity.value,
                "business_impact": incident.business_impact,
                "created_at": incident.created_at.isoformat()
            },
            "business_context": scenario
        })
        
        self.active_incidents[incident.id] = incident
        logger.info(f"🚀 Business incident triggered: {incident.title}")
        
        # Start workflow
        asyncio.create_task(self._execute_workflow(incident))
        
        return incident
    
    async def _execute_workflow(self, incident: Incident):
        """Execute workflow with all 7 agents"""
        try:
            incident.workflow_status = "in_progress"
            await self._broadcast_workflow_update(incident, f"Enhanced workflow started: {incident.incident_type}")
            
            # Agent execution sequence
            agent_sequence = [
                ("monitoring", self._execute_monitoring_agent),
                ("rca", self._execute_rca_agent),
                ("pager", self._execute_pager_agent),
                ("ticketing", self._execute_ticketing_agent),
                ("email", self._execute_email_agent),
                ("remediation", self._execute_remediation_agent),
                ("validation", self._execute_validation_agent)
            ]
            
            for agent_id, agent_function in agent_sequence:
                try:
                    incident.current_agent = agent_id
                    await self._broadcast_workflow_update(incident, f"Starting {agent_id} agent")
                    
                    # Process A2A messages
                    await self._process_a2a_messages(agent_id, incident)
                    
                    # Execute agent
                    execution = await agent_function(incident)
                    incident.executions[agent_id] = execution
                    self.agent_execution_history[agent_id].append(execution)
                    
                    if execution.status == AgentStatus.SUCCESS:
                        incident.completed_agents.append(agent_id)
                        await self._broadcast_workflow_update(incident, f"{agent_id} completed with {len(execution.logs)} logs")
                    else:
                        incident.failed_agents.append(agent_id)
                        await self._broadcast_workflow_update(incident, f"{agent_id} failed: {execution.error_message}")
                    
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {str(e)}")
                    incident.failed_agents.append(agent_id)
                    await self._broadcast_workflow_update(incident, f"{agent_id} error: {str(e)}")
            
            # Complete workflow
            await self._complete_workflow(incident)
            
        except Exception as e:
            incident.workflow_status = "failed"
            incident.status = "failed"
            logger.error(f"Workflow failed for incident {incident.id}: {str(e)}")
    
    async def _complete_workflow(self, incident: Incident):
        """Complete the workflow"""
        try:
            incident.workflow_status = "completed"
            incident.current_agent = ""
            incident.status = "resolved" if len(incident.failed_agents) == 0 else "partially_resolved"
            
            await self._broadcast_workflow_update(incident, f"Workflow completed - {len(incident.completed_agents)}/7 agents successful")
            
            self.incident_history.append(incident)
            del self.active_incidents[incident.id]
            
        except Exception as e:
            incident.workflow_status = "failed"
            incident.status = "failed"
            logger.error(f"Workflow completion failed for incident {incident.id}: {str(e)}")
    
    async def _broadcast_workflow_update(self, incident: Incident, message: str):
        """Broadcast workflow updates"""
        if self.websocket_connections:
            update_data = {
                "type": "workflow_update",
                "incident_id": incident.id,
                "current_agent": incident.current_agent,
                "completed_agents": incident.completed_agents,
                "workflow_status": incident.workflow_status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            for ws in self.websocket_connections.copy():
                try:
                    await ws.send_text(json.dumps(update_data))
                except:
                    self.websocket_connections.remove(ws)
    
    async def _process_a2a_messages(self, agent_id: str, incident: Incident):
        """Process A2A messages for agent"""
        messages = self.a2a_protocol.get_messages(agent_id)
        
        for message in messages:
            logger.info(f"📨 Processing A2A message for {agent_id}: {message.message_type}")
            
            if agent_id in incident.executions:
                incident.executions[agent_id].a2a_messages_received += 1
            
            if message.message_type == "collaboration_request":
                collab_id = message.content.get("collaboration_id")
                if agent_id in incident.executions:
                    incident.executions[agent_id].collaboration_sessions.append(collab_id)
            elif message.message_type == "data_share":
                mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
                if mcp_context:
                    shared_data = message.content.get("data", {})
                    confidence = message.content.get("confidence", 0.8)
                    mcp_context.update_context(message.sender_agent_id, shared_data, confidence)

    # AGENT IMPLEMENTATIONS WITH DETAILED LOGGING
    async def _execute_monitoring_agent(self, incident: Incident) -> AgentExecution:
        """Enhanced Monitoring Agent with detailed logging"""
        execution = AgentExecution(
            agent_id="monitoring", agent_name="Business Intelligence Monitoring Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Get MCP context
            mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
            if mcp_context:
                contextual_insights = mcp_context.get_contextual_insights("monitoring")
                execution.contextual_insights_used = contextual_insights
                await self._detailed_log(execution, "🧠 MCP Context loaded - leveraging shared business intelligence", "MCP_ANALYSIS", {
                    "context_confidence": contextual_insights.get("context_confidence", 0.0)
                })
            
            await self._detailed_log(execution, f"🔍 Initiating comprehensive business monitoring for {incident.incident_type}", "BUSINESS_ANALYSIS", {
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact
            })
            execution.progress = 20
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Business-specific monitoring
            if incident.incident_type == "business_critical":
                await self._detailed_log(execution, "📊 Analyzing critical business KPIs: order processing, revenue impact", "FINANCIAL_ANALYSIS")
                execution.progress = 50
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                # A2A collaboration
                collab_id = self.a2a_protocol.initiate_collaboration(
                    "monitoring", ["rca"], 
                    "business_impact_correlation_analysis",
                    {"incident_type": incident.incident_type, "revenue_impact": "$2,500/min"}
                )
                execution.collaboration_sessions.append(collab_id)
                
                await self._detailed_log(execution, f"🤝 A2A Collaboration initiated with RCA agent", "A2A_COLLABORATION", {
                    "collaboration_id": collab_id
                })
                
                execution.output_data = {
                    "business_metrics": {
                        "revenue_loss_per_minute": 2500,
                        "orders_affected": 1847,
                        "customer_satisfaction_risk": "high"
                    },
                    "mcp_enhanced": True,
                    "collaboration_initiated": True
                }
                
            elif incident.incident_type == "payment_critical":
                await self._detailed_log(execution, "💳 Payment system monitoring: transaction flows, regional analysis", "TECHNICAL_ANALYSIS")
                execution.progress = 50
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                # Share payment data via A2A
                payment_data = {
                    "payment_provider": "Stripe",
                    "failure_pattern": "ssl_webhook_validation",
                    "regional_impact": "EU_only"
                }
                
                message = A2AMessage(
                    sender_agent_id="monitoring",
                    receiver_agent_id="remediation",
                    message_type="data_share",
                    content={"data": payment_data, "confidence": 0.94},
                    priority="critical"
                )
                self.a2a_protocol.send_message(message)
                execution.a2a_messages_sent += 1
                
                await self._detailed_log(execution, "📨 Critical payment data shared with remediation agent", "A2A_COMMUNICATION", {
                    "recipient": "remediation",
                    "confidence": 0.94
                })
                
                execution.output_data = {
                    "payment_analysis": {
                        "provider": "Stripe",
                        "failure_rate": "89%",
                        "regional_impact": "EU"
                    },
                    "a2a_intelligence_shared": True
                }
                
            else:
                await self._detailed_log(execution, f"📈 General business monitoring for {incident.incident_type}", "BUSINESS_ANALYSIS")
                execution.progress = 50
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                execution.output_data = {
                    "general_metrics": {
                        "business_impact_score": random.randint(70, 95),
                        "system_health": "degraded"
                    }
                }
            
            # Update MCP context
            if mcp_context:
                mcp_context.update_context("monitoring", execution.output_data, 0.93)
                await self._detailed_log(execution, "🧠 MCP Context updated with monitoring data", "MCP_UPDATE", {
                    "confidence_score": 0.93
                })
            
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, "✅ Business monitoring analysis completed successfully", "SUCCESS", {
                "total_logs": len(execution.logs),
                "mcp_enhanced": True
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Monitoring failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    async def _execute_rca_agent(self, incident: Incident) -> AgentExecution:
        """Enhanced RCA Agent with detailed logging"""
        execution = AgentExecution(
            agent_id="rca", agent_name="Business Impact Root Cause Analysis Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Get MCP context
            mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
            contextual_data = {}
            if mcp_context:
                contextual_data = mcp_context.get_contextual_insights("rca")
                execution.contextual_insights_used = contextual_data
                await self._detailed_log(execution, "🧠 MCP context analysis initiated", "MCP_ANALYSIS", {
                    "peer_insights_count": len(contextual_data.get("peer_insights", {}))
                })
            
            await self._detailed_log(execution, f"🔬 Business-focused root cause analysis for {incident.incident_type}", "ROOT_CAUSE_ANALYSIS", {
                "incident_severity": incident.severity.value
            })
            execution.progress = 25
            await asyncio.sleep(random.uniform(2.0, 2.5))
            
            # Enhanced analysis using context
            confidence_boost = 0.0
            if contextual_data.get("peer_insights"):
                confidence_boost = 0.20
                await self._detailed_log(execution, "💡 Leveraging peer agent insights for enhanced correlation", "PEER_ANALYSIS", {
                    "confidence_boost": confidence_boost
                })
                execution.progress = 50
                await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # Get scenario root cause
            scenario = None
            for s in BUSINESS_INCIDENT_SCENARIOS:
                if s["title"] == incident.title:
                    scenario = s
                    break
            
            if scenario:
                root_cause = scenario["root_cause"]
                await self._detailed_log(execution, f"🎯 Root cause identified: {scenario['incident_type']}", "ROOT_CAUSE_ANALYSIS", {
                    "primary_cause": root_cause
                })
            else:
                root_cause = f"Business-critical {incident.incident_type} issue requiring investigation"
            
            base_confidence = random.uniform(0.88, 0.96)
            enhanced_confidence = min(0.99, base_confidence + confidence_boost)
            
            # Business impact analysis
            financial_impact = self._calculate_financial_impact(incident)
            
            await self._detailed_log(execution, "💰 Financial impact analysis completed", "FINANCIAL_ANALYSIS", financial_impact)
            
            execution.output_data = {
                "root_cause": root_cause,
                "confidence": enhanced_confidence,
                "business_analysis": {
                    "financial_impact": financial_impact,
                    "recovery_priority": self._get_recovery_priority(incident.incident_type)
                },
                "mcp_enhanced": True,
                "used_peer_insights": bool(contextual_data.get("peer_insights"))
            }
            
            # Share RCA findings via A2A
            rca_findings = {
                "root_cause_summary": root_cause,
                "confidence_score": enhanced_confidence,
                "financial_impact": financial_impact
            }
            
            for agent in ["remediation", "validation", "pager"]:
                message = A2AMessage(
                    sender_agent_id="rca",
                    receiver_agent_id=agent,
                    message_type="data_share",
                    content={"data": rca_findings, "confidence": enhanced_confidence},
                    priority="high"
                )
                self.a2a_protocol.send_message(message)
                execution.a2a_messages_sent += 1
            
            await self._detailed_log(execution, "📨 RCA findings shared with multiple agents", "A2A_SHARE", {
                "recipients": ["remediation", "validation", "pager"],
                "confidence": enhanced_confidence
            })
            
            # Update MCP context
            if mcp_context:
                mcp_context.update_context("rca", execution.output_data, enhanced_confidence)
                await self._detailed_log(execution, "🧠 MCP Context updated with RCA analysis", "MCP_UPDATE")
            
            incident.root_cause = execution.output_data["root_cause"]
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ RCA analysis completed - Confidence: {enhanced_confidence:.1%}", "SUCCESS")
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ RCA failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    async def _execute_pager_agent(self, incident: Incident) -> AgentExecution:
        """Enhanced Pager Agent with business stakeholder escalation"""
        execution = AgentExecution(
            agent_id="pager", agent_name="Business Stakeholder Escalation Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            await self._detailed_log(execution, f"📞 Business stakeholder escalation for {incident.incident_type}", "STAKEHOLDER_ANALYSIS", {
                "severity": incident.severity.value
            })
            execution.progress = 30
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Business stakeholder identification
            stakeholders = self._identify_business_stakeholders(incident)
            
            await self._detailed_log(execution, "👥 Business stakeholder identification completed", "STAKEHOLDER_ANALYSIS", {
                "primary_stakeholders": len(stakeholders.get("primary", [])),
                "executive_notification": stakeholders.get("executive_required", False)
            })
            execution.progress = 80
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            execution.output_data = {
                "pagerduty_incident_id": f"BIZ-{incident.incident_type.upper()}-{incident.id[-6:]}",
                "business_escalation": stakeholders,
                "notification_channels": ["PagerDuty", "Business Slack", "Executive Email"],
                "business_sla": self._get_business_sla(incident.incident_type)
            }
            
            incident.pagerduty_incident_id = execution.output_data["pagerduty_incident_id"]
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, "✅ Business stakeholder escalation completed", "SUCCESS")
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Escalation failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    async def _execute_ticketing_agent(self, incident: Incident) -> AgentExecution:
        """Enhanced Ticketing Agent with business classification"""
        execution = AgentExecution(
            agent_id="ticketing", agent_name="Business Impact Ticketing Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            await self._detailed_log(execution, "🎫 Business-focused ticket creation", "CLASSIFICATION")
            execution.progress = 40
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            business_priority = self._get_business_priority(incident)
            
            await self._detailed_log(execution, "📊 Business impact classification completed", "CLASSIFICATION", {
                "business_priority": business_priority
            })
            execution.progress = 85
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            execution.output_data = {
                "ticket_id": f"BIZ-{incident.incident_type.upper()}{datetime.now().strftime('%Y%m%d')}{incident.id[-4:]}",
                "business_priority": business_priority,
                "business_impact_score": self._calculate_business_impact_score(incident)
            }
            
            incident.servicenow_ticket_id = execution.output_data["ticket_id"]
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ Business ticket created: {execution.output_data['ticket_id']}", "SUCCESS")
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Ticketing failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    async def _execute_email_agent(self, incident: Incident) -> AgentExecution:
        """Enhanced Email Agent with stakeholder communication"""
        execution = AgentExecution(
            agent_id="email", agent_name="Business Stakeholder Communication Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            await self._detailed_log(execution, "📧 Business stakeholder communication strategy", "COMMUNICATION_PLANNING")
            execution.progress = 35
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            communication_strategy = self._develop_communication_strategy(incident)
            
            await self._detailed_log(execution, "👥 Stakeholder communication planning completed", "STAKEHOLDER_ANALYSIS", {
                "executive_briefing": communication_strategy.get("executive_briefing", False)
            })
            execution.progress = 80
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            execution.output_data = {
                "communication_strategy": communication_strategy,
                "business_focused": True
            }
            
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, "✅ Business communication completed", "SUCCESS")
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Communication failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    async def _execute_remediation_agent(self, incident: Incident) -> AgentExecution:
        """Enhanced Remediation Agent with business continuity focus"""
        execution = AgentExecution(
            agent_id="remediation", agent_name="Business Continuity Remediation Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            await self._detailed_log(execution, "🔧 Business continuity remediation planning", "BUSINESS_ANALYSIS")
            execution.progress = 25
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            remediation_actions = self._get_remediation_actions(incident.incident_type)
            
            await self._detailed_log(execution, f"⚡ Executing {len(remediation_actions)} remediation procedures", "REMEDIATION_EXECUTION", {
                "actions": len(remediation_actions)
            })
            execution.progress = 70
            await asyncio.sleep(random.uniform(2.0, 2.5))
            
            execution.output_data = {
                "remediation_actions": remediation_actions,
                "business_continuity_focus": True
            }
            
            incident.remediation_applied = remediation_actions
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, "✅ Business remediation completed", "SUCCESS")
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Remediation failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    async def _execute_validation_agent(self, incident: Incident) -> AgentExecution:
        """Enhanced Validation Agent with business verification"""
        execution = AgentExecution(
            agent_id="validation", agent_name="Business Continuity Validation Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Get comprehensive context
            mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
            confidence_factors = []
            
            if mcp_context:
                full_context = mcp_context.get_contextual_insights("validation")
                execution.contextual_insights_used = full_context
                confidence_factors = list(mcp_context.confidence_scores.values())
            
            overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.8
            
            await self._detailed_log(execution, "🔍 Comprehensive business validation initiated", "BUSINESS_VALIDATION", {
                "agent_insights_count": len(confidence_factors),
                "overall_confidence": f"{overall_confidence:.2%}"
            })
            execution.progress = 30
            await asyncio.sleep(random.uniform(2.0, 2.5))
            
            # Business metrics validation
            business_validation = self._validate_business_metrics(incident)
            
            await self._detailed_log(execution, "💼 Business metrics validation completed", "BUSINESS_VALIDATION", business_validation)
            execution.progress = 70
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # Determine resolution success
            base_success_rate = 0.80
            business_boost = 0.15 if overall_confidence > 0.85 else 0.10
            final_success_rate = base_success_rate + business_boost
            resolution_successful = random.random() < final_success_rate
            
            execution.output_data = {
                "business_validation": business_validation,
                "resolution_successful": resolution_successful,
                "validation_score": random.uniform(0.94, 0.99) if resolution_successful else random.uniform(0.75, 0.89),
                "overall_confidence": overall_confidence
            }
            
            # Final MCP update
            if mcp_context:
                mcp_context.update_context("validation", execution.output_data, 0.97)
                mcp_context.shared_knowledge["final_resolution"] = {
                    "status": "resolved" if resolution_successful else "partially_resolved",
                    "validation_score": execution.output_data["validation_score"],
                    "validated_at": datetime.now().isoformat()
                }
                
                await self._detailed_log(execution, "🧠 Final MCP context update", "MCP_FINAL_UPDATE", {
                    "resolution_status": "resolved" if resolution_successful else "partially_resolved"
                })
            
            # Set incident resolution
            if resolution_successful:
                incident.resolution = f"Business incident {incident.incident_type} fully resolved using MCP+A2A enhanced analysis. Business continuity restored with {overall_confidence:.1%} confidence."
                incident.status = "resolved"
            else:
                incident.resolution = f"Business incident {incident.incident_type} partially resolved - continued monitoring required. Validation score: {execution.output_data['validation_score']:.1%}."
                incident.status = "partially_resolved"
            
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            status_msg = "fully resolved" if resolution_successful else "partially resolved"
            await self._detailed_log(execution, f"✅ Business validation completed - Issue {status_msg}", "SUCCESS", {
                "validation_score": f"{execution.output_data['validation_score']:.1%}"
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Validation failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution

    # HELPER METHODS
    def _calculate_financial_impact(self, incident: Incident) -> Dict[str, Any]:
        """Calculate financial impact"""
        return {
            "immediate_loss": f"${random.randint(10000, 150000)}",
            "hourly_impact": f"${random.randint(5000, 50000)}",
            "affected_transactions": random.randint(100, 5000),
            "impact_category": incident.severity.value
        }
    
    def _get_recovery_priority(self, incident_type: str) -> str:
        """Get recovery priority"""
        priorities = {
            "business_critical": "revenue_protection_first",
            "payment_critical": "payment_flow_restoration",
            "performance_critical": "user_experience_recovery"
        }
        return priorities.get(incident_type, "service_restoration")
    
    def _identify_business_stakeholders(self, incident: Incident) -> Dict[str, Any]:
        """Identify business stakeholders"""
        stakeholders = {
            "business_critical": {
                "primary": ["VP Operations", "Customer Success Manager"],
                "executive_required": True
            },
            "payment_critical": {
                "primary": ["CFO", "Payment Operations Manager"],
                "executive_required": True
            }
        }
        
        return stakeholders.get(incident.incident_type, {
            "primary": ["Operations Manager"],
            "executive_required": incident.severity == IncidentSeverity.CRITICAL
        })
    
    def _get_business_sla(self, incident_type: str) -> str:
        """Get business SLA"""
        slas = {
            "business_critical": "30 minutes",
            "payment_critical": "15 minutes",
            "trading_critical": "5 minutes"
        }
        return slas.get(incident_type, "2 hours")
    
    def _get_business_priority(self, incident: Incident) -> str:
        """Get business priority"""
        if incident.severity == IncidentSeverity.CRITICAL:
            return "P0 - Business Critical"
        elif incident.severity == IncidentSeverity.HIGH:
            return "P1 - High Business Impact"
        else:
            return "P2 - Medium Business Impact"
    
    def _calculate_business_impact_score(self, incident: Incident) -> int:
        """Calculate business impact score"""
        base_score = {
            "critical": 90,
            "high": 70,
            "medium": 50
        }.get(incident.severity.value, 50)
        return min(100, base_score)
    
    def _develop_communication_strategy(self, incident: Incident) -> Dict[str, Any]:
        """Develop communication strategy"""
        if incident.severity == IncidentSeverity.CRITICAL:
            return {
                "channels": ["email", "slack", "sms"],
                "executive_briefing": True,
                "customer_communication": True
            }
        else:
            return {
                "channels": ["email", "slack"],
                "executive_briefing": False,
                "customer_communication": False
            }
    
    def _get_remediation_actions(self, incident_type: str) -> List[str]:
        """Get remediation actions"""
        actions = {
            "business_critical": [
                "activate_backup_payment_processing",
                "implement_order_queue_priority_system",
                "enable_customer_communication_automation"
            ],
            "payment_critical": [
                "failover_to_secondary_payment_provider",
                "activate_transaction_retry_mechanisms",
                "implement_payment_method_diversification"
            ]
        }
        
        return actions.get(incident_type, [
            "restore_core_functionality",
            "minimize_customer_impact"
        ])
    
    def _validate_business_metrics(self, incident: Incident) -> Dict[str, Any]:
        """Validate business metrics"""
        if incident.incident_type == "business_critical":
            return {
                "revenue_flow": "restored" if random.random() < 0.85 else "partially_restored",
                "customer_experience": "baseline_achieved" if random.random() < 0.80 else "improving",
                "operational_capacity": "90%" if random.random() < 0.85 else "75%"
            }
        else:
            return {
                "revenue_flow": "stable" if random.random() < 0.80 else "recovering",
                "customer_experience": "acceptable" if random.random() < 0.75 else "degraded",
                "operational_capacity": "85%" if random.random() < 0.80 else "70%"
            }
    
    # ENHANCED LOGGING UTILITY
    async def _detailed_log(self, execution: AgentExecution, message: str, log_type: str = "INFO", additional_data: Dict[str, Any] = None):
        """Enhanced detailed logging with business context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": log_type,
            "message": message,
            "execution_id": execution.execution_id,
            "agent_id": execution.agent_id,
            "mcp_context_id": execution.mcp_context_id,
            "log_type": log_type,
            "additional_data": additional_data or {},
            "business_context": True
        }
        execution.logs.append(log_entry)
        
        # Console logging with emoji
        log_prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "BUSINESS_ANALYSIS": "💼",
            "FINANCIAL_ANALYSIS": "💰",
            "TECHNICAL_ANALYSIS": "🔧",
            "A2A_COLLABORATION": "🤝",
            "A2A_COMMUNICATION": "📨",
            "A2A_SHARE": "📤",
            "MCP_ANALYSIS": "🧠",
            "MCP_UPDATE": "🧠📝",
            "MCP_FINAL_UPDATE": "🧠✨",
            "STAKEHOLDER_ANALYSIS": "👥",
            "CLASSIFICATION": "📊",
            "COMMUNICATION_PLANNING": "📋",
            "BUSINESS_VALIDATION": "💼✅",
            "ROOT_CAUSE_ANALYSIS": "🎯",
            "PEER_ANALYSIS": "🤝🔍",
            "REMEDIATION_EXECUTION": "⚡"
        }.get(log_type, "📝")
        
        formatted_message = f"{log_prefix} [{execution.agent_id.upper()}] {message}"
        if additional_data:
            formatted_message += f" | {json.dumps(additional_data, default=str)}"
        
        logger.info(f"[{execution.incident_id}] {formatted_message}")

# Global workflow engine
workflow_engine = EnhancedWorkflowEngine()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

class EnhancedMonitoringApp:
    def __init__(self):
        self.app = FastAPI(
            title="Complete AI Monitoring System v5 - Clean Version",
            description="MCP + A2A + Business Intelligence + Detailed Logging - NO DUPLICATES",
            version="5.0.0-clean",
            docs_url="/api/docs"
        )
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        # Trigger business incidents
        @self.app.post("/api/trigger-incident")
        async def trigger_business_incident(incident_data: dict):
            incident = await workflow_engine.trigger_incident_workflow(incident_data)
            return {
                "incident_id": incident.id,
                "workflow_id": incident.workflow_id,
                "mcp_context_id": incident.mcp_context_id,
                "status": "business_incident_workflow_started",
                "title": incident.title,
                "severity": incident.severity.value,
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact,
                "message": f"Complete v5 business incident {incident.id} workflow initiated",
                "enhanced_features": [
                    "Business-Centric Incident Scenarios", 
                    "Model Context Protocol", 
                    "Agent-to-Agent Communication", 
                    "Detailed Agent Console Logs"
                ]
            }
        
        # GET DETAILED AGENT LOGS - KEY FEATURE
        @self.app.get("/api/incidents/{incident_id}/agent/{agent_id}/logs")
        async def get_detailed_agent_logs(incident_id: str, agent_id: str):
            """Get comprehensive detailed logs for a specific agent execution"""
            incident = None
            if incident_id in workflow_engine.active_incidents:
                incident = workflow_engine.active_incidents[incident_id]
            else:
                incident = next((i for i in workflow_engine.incident_history if i.id == incident_id), None)
            
            if not incident or agent_id not in incident.executions:
                return {"error": "Incident or agent execution not found"}
            
            execution = incident.executions[agent_id]
            
            return {
                "incident_id": incident_id,
                "agent_id": agent_id,
                "agent_name": execution.agent_name,
                "execution_id": execution.execution_id,
                "status": execution.status.value,
                "progress": execution.progress,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": execution.duration_seconds,
                "business_context": {
                    "incident_type": incident.incident_type,
                    "business_impact": incident.business_impact,
                    "severity": incident.severity.value
                },
                "mcp_enhancements": {
                    "context_id": execution.mcp_context_id,
                    "contextual_insights_used": execution.contextual_insights_used,
                    "mcp_enhanced": bool(execution.contextual_insights_used)
                },
                "a2a_communications": {
                    "messages_sent": execution.a2a_messages_sent,
                    "messages_received": execution.a2a_messages_received,
                    "collaboration_sessions": execution.collaboration_sessions
                },
                "execution_data": {
                    "input_data": execution.input_data,
                    "output_data": execution.output_data,
                    "error_message": execution.error_message
                },
                "detailed_logs": execution.logs,
                "log_summary": {
                    "total_log_entries": len(execution.logs),
                    "log_types": list(set(log.get("log_type", "INFO") for log in execution.logs)),
                    "business_focused_logs": sum(1 for log in execution.logs if log.get("business_context"))
                }
            }
        
        # Get incident status
        @self.app.get("/api/incidents/{incident_id}/status")
        async def get_incident_status(incident_id: str):
            incident = None
            if incident_id in workflow_engine.active_incidents:
                incident = workflow_engine.active_incidents[incident_id]
            else:
                incident = next((i for i in workflow_engine.incident_history if i.id == incident_id), None)
            
            if not incident:
                return {"error": "Incident not found"}
            
            # MCP context data
            mcp_data = {}
            if incident.mcp_context_id:
                context = workflow_engine.mcp_registry.get_context(incident.mcp_context_id)
                if context:
                    mcp_data = {
                        "context_id": context.context_id,
                        "context_version": context.context_version,
                        "agent_insights_count": len(context.agent_insights),
                        "avg_confidence": sum(context.confidence_scores.values()) / len(context.confidence_scores) if context.confidence_scores else 0.0,
                        "business_context": True
                    }
            
            # A2A data
            a2a_data = {
                "total_messages_sent": sum(exec.a2a_messages_sent for exec in incident.executions.values()),
                "total_messages_received": sum(exec.a2a_messages_received for exec in incident.executions.values()),
                "business_collaboration_focus": True
            }
            
            return {
                "incident_id": incident.id,
                "title": incident.title,
                "description": incident.description,
                "severity": incident.severity.value,
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact,
                "status": incident.status,
                "workflow_status": incident.workflow_status,
                "current_agent": incident.current_agent,
                "completed_agents": incident.completed_agents,
                "failed_agents": incident.failed_agents,
                "created_at": incident.created_at.isoformat(),
                "updated_at": incident.updated_at.isoformat(),
                "affected_systems": incident.affected_systems,
                "root_cause": incident.root_cause,
                "resolution": incident.resolution,
                "pagerduty_incident_id": incident.pagerduty_incident_id,
                "servicenow_ticket_id": incident.servicenow_ticket_id,
                "remediation_applied": incident.remediation_applied,
                "enhanced_features": {
                    "mcp_context": mcp_data,
                    "a2a_protocol": a2a_data
                },
                "executions": {
                    agent_id: {
                        "agent_name": execution.agent_name,
                        "status": execution.status.value,
                        "progress": execution.progress,
                        "started_at": execution.started_at.isoformat() if execution.started_at else None,
                        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                        "duration": execution.duration_seconds,
                        "error": execution.error_message,
                        "detailed_logging": {
                            "total_log_entries": len(execution.logs),
                            "log_types": list(set(log.get("log_type", "INFO") for log in execution.logs)),
                            "logs_available": True
                        },
                        "mcp_enhanced": bool(execution.contextual_insights_used),
                        "a2a_messages": {
                            "sent": execution.a2a_messages_sent,
                            "received": execution.a2a_messages_received
                        },
                        "collaborations": len(execution.collaboration_sessions),
                        "output_data": execution.output_data
                    }
                    for agent_id, execution in incident.executions.items()
                }
            }
        
        # Dashboard stats
        @self.app.get("/api/dashboard/stats")
        async def get_dashboard_stats():
            all_incidents = list(workflow_engine.active_incidents.values()) + workflow_engine.incident_history
            today_incidents = [i for i in all_incidents if i.created_at.date() == datetime.now().date()]
            business_incidents = [i for i in all_incidents if hasattr(i, 'business_impact') and i.business_impact]
            
            # Agent statistics
            agent_stats = {}
            for agent_id in workflow_engine.agent_execution_history:
                executions = workflow_engine.agent_execution_history[agent_id]
                successful = len([e for e in executions if e.status == AgentStatus.SUCCESS])
                total = len(executions)
                avg_duration = sum([e.duration_seconds for e in executions if e.duration_seconds > 0]) / max(total, 1)
                
                # Enhanced metrics
                mcp_enhanced = len([e for e in executions if e.contextual_insights_used])
                a2a_messages = sum([e.a2a_messages_sent + e.a2a_messages_received for e in executions])
                total_logs = sum([len(e.logs) for e in executions])
                business_logs = sum([
                    sum(1 for log in e.logs if log.get("business_context"))
                    for e in executions
                ])
                
                agent_stats[agent_id] = {
                    "agent_name": f"Business Enhanced {agent_id.title()} Agent",
                    "total_executions": total,
                    "successful_executions": successful,
                    "success_rate": (successful / max(total, 1)) * 100,
                    "average_duration": round(avg_duration, 2),
                    "enhanced_features": {
                        "mcp_enhanced_executions": mcp_enhanced,
                        "a2a_messages_total": a2a_messages,
                        "detailed_logging": {
                            "total_logs": total_logs,
                            "business_context_logs": business_logs,
                            "avg_logs_per_execution": total_logs / max(total, 1)
                        }
                    }
                }
            
            return {
                "incidents": {
                    "total_all_time": len(all_incidents),
                    "active": len(workflow_engine.active_incidents),
                    "today": len(today_incidents),
                    "resolved_today": len([i for i in today_incidents if i.status == "resolved"]),
                    "business_incidents": len(business_incidents)
                },
                "agents": agent_stats,
                "enhanced_features": {
                    "mcp": {
                        "total_contexts": len(workflow_engine.mcp_registry.contexts),
                        "business_intelligence_active": True
                    },
                    "a2a": {
                        "total_messages": len(workflow_engine.a2a_protocol.message_history),
                        "active_collaborations": len(workflow_engine.a2a_protocol.active_collaborations),
                        "business_focused_protocol": True
                    }
                },
                "system": {
                    "version": "5.0.0-clean",
                    "architecture": [
                        "All 7 Business-Enhanced Agents",
                        "Model Context Protocol", 
                        "Agent-to-Agent Communication",
                        "Detailed Agent Console Logging",
                        "Business-Centric Incident Scenarios"
                    ],
                    "available_business_scenarios": len(BUSINESS_INCIDENT_SCENARIOS),
                    "detailed_logging_active": True
                }
            }
        
        # Get agents
        @self.app.get("/api/agents")
        async def get_agents():
            agent_configs = {
                "monitoring": "Business Intelligence Monitoring Agent with comprehensive detailed logging",
                "rca": "Business Impact Root Cause Analysis Agent with detailed analysis logging", 
                "pager": "Business Stakeholder Escalation Agent with comprehensive logging",
                "ticketing": "Business Impact Ticketing Agent with comprehensive business context logging",
                "email": "Business Stakeholder Communication Agent with detailed communication logging",
                "remediation": "Business Continuity Remediation Agent with comprehensive action logging",
                "validation": "Business Continuity Validation Agent with detailed validation logging"
            }
            
            agents_data = {}
            for agent_id, description in agent_configs.items():
                executions = workflow_engine.agent_execution_history[agent_id]
                recent = executions[-1] if executions else None
                
                successful_count = len([e for e in executions if e.status == AgentStatus.SUCCESS])
                total_count = len(executions)
                avg_duration = sum([e.duration_seconds for e in executions if e.duration_seconds > 0]) / max(total_count, 1)
                
                total_logs = sum([len(e.logs) for e in executions])
                business_logs = sum([
                    sum(1 for log in e.logs if log.get("business_context"))
                    for e in executions
                ])
                
                agents_data[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": f"Business Enhanced {agent_id.title()} Agent",
                    "status": "ready",
                    "description": description,
                    "total_executions": total_count,
                    "successful_executions": successful_count,
                    "success_rate": (successful_count / max(total_count, 1)) * 100,
                    "average_duration": round(avg_duration, 2),
                    "enhanced_features": {
                        "detailed_logging": {
                            "total_logs": total_logs,
                            "business_context_logs": business_logs,
                            "detailed_logging_active": True
                        },
                        "business_enhanced": True
                    },
                    "capabilities": workflow_engine.a2a_protocol.agent_capabilities.get(agent_id, []),
                    "recent_performance": {
                        "last_execution_status": recent.status.value if recent else "idle",
                        "last_log_count": len(recent.logs) if recent else 0
                    }
                }
            
            return {
                "agents": agents_data, 
                "total_agents": 7,
                "system_capabilities": {
                    "business_intelligence_monitoring": True,
                    "detailed_agent_logging": True,
                    "mcp_context_sharing": True,
                    "a2a_communication": True
                }
            }
        
        # Agent execution history
        @self.app.get("/api/agents/{agent_id}/history")
        async def get_agent_history(agent_id: str, limit: int = 20):
            if agent_id not in workflow_engine.agent_execution_history:
                return {"error": "Agent not found"}
            
            executions = workflow_engine.agent_execution_history[agent_id][-limit:]
            
            return {
                "agent_id": agent_id,
                "total_executions": len(workflow_engine.agent_execution_history[agent_id]),
                "recent_executions": [
                    {
                        "execution_id": exec.execution_id,
                        "incident_id": exec.incident_id,
                        "status": exec.status.value,
                        "started_at": exec.started_at.isoformat() if exec.started_at else None,
                        "duration": exec.duration_seconds,
                        "progress": exec.progress,
                        "detailed_logs_count": len(exec.logs),
                        "business_enhanced": True
                    }
                    for exec in executions
                ]
            }
        
        # MCP contexts
        @self.app.get("/api/mcp/contexts")
        async def get_mcp_contexts():
            contexts = []
            for context_id, context in workflow_engine.mcp_registry.contexts.items():
                contexts.append({
                    "context_id": context.context_id,
                    "incident_id": context.incident_id,
                    "context_type": context.context_type,
                    "created_at": context.created_at.isoformat(),
                    "agent_count": len(context.agent_insights),
                    "confidence_avg": sum(context.confidence_scores.values()) / len(context.confidence_scores) if context.confidence_scores else 0.0,
                    "business_enhanced": "business" in context.context_type
                })
            
            return {
                "total_contexts": len(contexts), 
                "contexts": contexts
            }
        
        # A2A messages
        @self.app.get("/api/a2a/messages/history")
        async def get_a2a_messages(limit: int = 50):
            recent_messages = workflow_engine.a2a_protocol.message_history[-limit:]
            
            return {
                "total_messages": len(workflow_engine.a2a_protocol.message_history),
                "recent_messages": [msg.to_dict() for msg in recent_messages]
            }
        
        # A2A collaborations
        @self.app.get("/api/a2a/collaborations")
        async def get_a2a_collaborations():
            collaborations = []
            for collab_id, collab in workflow_engine.a2a_protocol.active_collaborations.items():
                collaborations.append({
                    "collaboration_id": collab_id,
                    "initiator": collab["initiator"],
                    "participants": collab["participants"],
                    "task": collab["task"],
                    "status": collab["status"],
                    "created_at": collab["created_at"]
                })
            
            return {
                "total_collaborations": len(collaborations),
                "collaborations": collaborations
            }
        
        # Recent incidents
        @self.app.get("/api/incidents")
        async def get_recent_incidents(limit: int = 10):
            all_incidents = list(workflow_engine.active_incidents.values()) + workflow_engine.incident_history
            all_incidents.sort(key=lambda x: x.created_at, reverse=True)
            recent_incidents = all_incidents[:limit]
            
            return {
                "incidents": [
                    {
                        "id": incident.id,
                        "title": incident.title,
                        "description": incident.description,
                        "severity": incident.severity.value,
                        "incident_type": incident.incident_type,
                        "business_impact": getattr(incident, 'business_impact', 'Assessment pending'),
                        "status": incident.status,
                        "workflow_status": incident.workflow_status,
                        "current_agent": incident.current_agent,
                        "completed_agents": incident.completed_agents,
                        "created_at": incident.created_at.isoformat(),
                        "business_enhanced": True,
                        "detailed_logs_available": sum(len(exec.logs) for exec in incident.executions.values())
                    }
                    for incident in recent_incidents
                ],
                "total_incidents": len(all_incidents)
            }
        
        # WebSocket for real-time updates
        @self.app.websocket("/ws/realtime")
        async def websocket_realtime_updates(websocket: WebSocket):
            await websocket.accept()
            await workflow_engine.add_websocket_connection(websocket)
            
            try:
                initial_data = {
                    "type": "connection_established",
                    "message": "Real-time business intelligence updates connected",
                    "timestamp": datetime.now().isoformat(),
                    "features": ["MCP Context Updates", "A2A Collaboration", "Detailed Logging"]
                }
                await websocket.send_text(json.dumps(initial_data))
                
                while True:
                    try:
                        data = await websocket.receive_text()
                        response = {
                            "type": "echo",
                            "received": data,
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket.send_text(json.dumps(response))
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        break
                        
            except WebSocketDisconnect:
                pass
            finally:
                await workflow_engine.remove_websocket_connection(websocket)
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "service": "Complete AI Monitoring System v5 - Clean Version",
                "version": "5.0.0-clean",
                "features": [
                    "All 7 Business-Enhanced Agents",
                    "Model Context Protocol (MCP)",
                    "Agent-to-Agent (A2A) Communication",
                    "Detailed Agent Console Logging",
                    "Business-Centric Incident Scenarios",
                    "Real-time WebSocket Updates"
                ],
                "business_scenarios": {
                    "total_available": len(BUSINESS_INCIDENT_SCENARIOS),
                    "categories": ["E-commerce", "Financial Services", "Trading"]
                },
                "workflow_engine": {
                    "active_incidents": len(workflow_engine.active_incidents),
                    "total_incidents": len(workflow_engine.incident_history) + len(workflow_engine.active_incidents),
                    "mcp_contexts": len(workflow_engine.mcp_registry.contexts),
                    "a2a_collaborations": len(workflow_engine.a2a_protocol.active_collaborations),
                    "detailed_logging_active": True
                }
            }
        
        # Serve frontend
        frontend_path = Path("frontend/build")
        if frontend_path.exists():
            self.app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
        else:
            @self.app.get("/")
            async def root():
                return {
                    "message": "🚀 Complete AI Monitoring System v5 - Clean Version",
                    "version": "5.0.0-clean",
                    "status": "CLEAN - NO DUPLICATES",
                    "key_features": [
                        "✅ All 7 Agents Dashboard - Business Enhanced",
                        "✅ Real-time Progress Tracking",
                        "✅ WebSocket Live Updates",
                        "✅ Agent Execution History",
                        "✅ Detailed Console Logs - CLEAN IMPLEMENTATION",
                        "✅ Business-Centric Incident Scenarios",
                        "✅ MCP + A2A Integration"
                    ],
                    "business_scenarios": [
                        "Order Processing Delays",
                        "Payment Failures Spike", 
                        "Search Latency Crisis",
                        "Cart Abandonment Spike",
                        "Bot Attack Detection",
                        "Trading Platform Latency",
                        "And more..."
                    ],
                    "detailed_logging": {
                        "feature": "ACTIVE - CLEAN IMPLEMENTATION",
                        "access": "Click any agent to view detailed console logs",
                        "content": "Business context, technical analysis, MCP insights, A2A communications"
                    },
                    "api_endpoints": {
                        "trigger_incident": "POST /api/trigger-incident",
                        "agent_logs": "GET /api/incidents/{id}/agent/{id}/logs",
                        "incident_status": "GET /api/incidents/{id}/status",
                        "dashboard_stats": "GET /api/dashboard/stats",
                        "agents": "GET /api/agents",
                        "health": "GET /health"
                    }
                }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        logger.info("🚀 Starting Complete AI Monitoring System v5 - CLEAN VERSION")
        logger.info("✅ ALL 7 BUSINESS-ENHANCED AGENTS: ACTIVE")
        logger.info("🧠 Model Context Protocol: ACTIVE")
        logger.info("🤝 Agent-to-Agent Protocol: ACTIVE") 
        logger.info("📝 DETAILED AGENT CONSOLE LOGS: CLEAN IMPLEMENTATION")
        logger.info("💼 Business-Centric Incident Scenarios: LOADED")
        logger.info("🔧 NO DUPLICATES - CLEAN CODE")
        logger.info(f"🌐 Dashboard: http://localhost:{port}")
        logger.info("🎯 Click any agent to view detailed console logs!")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    app = EnhancedMonitoringApp()
    app.run()
