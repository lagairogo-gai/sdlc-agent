"""
COMPLETE MCP + A2A Enhanced AI Monitoring System
Model Context Protocol + Agent-to-Agent Communication
ENHANCED with Business-Centric Incidents and Detailed Console Logs
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

# ENHANCED BUSINESS-CENTRIC INCIDENT SCENARIOS
BUSINESS_INCIDENT_SCENARIOS = [
    # E-commerce Business Incidents
    {
        "title": "Order Processing Delays - Critical Payment Integration Issue",
        "description": "Orders are taking 15+ minutes to move from 'Placed' to 'Shipped' state. Payment validation service experiencing high latency causing order pipeline bottleneck.",
        "severity": "critical",
        "affected_systems": ["payment-gateway", "order-management-system", "inventory-service", "shipping-service"],
        "incident_type": "business_critical",
        "business_impact": "Revenue loss: $2,500/minute. Customer satisfaction degradation. Potential order cancellations.",
        "root_cause": "Payment validation service database connection pool exhaustion combined with legacy synchronous processing architecture",
        "metrics": {
            "avg_processing_time": "15.7 minutes",
            "normal_processing_time": "2.3 minutes",
            "orders_affected": 1847,
            "revenue_at_risk": "$156,000"
        }
    },
    {
        "title": "Sudden Drop in Order Volume - 78% Below Normal",
        "description": "Customer order volume has dropped by 78% in the last 2 hours compared to historical patterns. Only 156 orders vs expected 712 orders.",
        "severity": "critical",
        "affected_systems": ["e-commerce-frontend", "recommendation-engine", "search-service", "product-catalog"],
        "incident_type": "business_anomaly",
        "business_impact": "Critical revenue impact. Potential customer acquisition funnel breakdown.",
        "root_cause": "Product search indexing failure causing empty search results for 67% of product queries",
        "metrics": {
            "current_order_rate": "156 orders/2hr",
            "expected_order_rate": "712 orders/2hr",
            "conversion_rate": "0.24%",
            "search_success_rate": "33%"
        }
    },
    {
        "title": "Payment Failures Spike - Regional Payment Provider Issue",
        "description": "89% increase in failed payment transactions from EU region via Stripe payment provider. Affecting premium customer segment.",
        "severity": "high",
        "affected_systems": ["payment-gateway", "stripe-integration", "fraud-detection", "customer-billing"],
        "incident_type": "payment_critical",
        "business_impact": "Lost sales: $45,000/hour. Premium customer churn risk. Potential compliance issues.",
        "root_cause": "Stripe EU webhook endpoint SSL certificate validation failure combined with retry mechanism bug",
        "metrics": {
            "failure_rate": "67%",
            "normal_failure_rate": "2.1%",
            "failed_transactions": 2341,
            "affected_regions": ["DE", "FR", "IT", "ES"]
        }
    },
    {
        "title": "Product Search Latency Crisis - 8.5s Average Response Time", 
        "description": "Users experiencing severe latency (8.5s avg) and 34% timeout rate while searching for products. Search conversion dropped 67%.",
        "severity": "critical",
        "affected_systems": ["elasticsearch-cluster", "product-search-api", "autocomplete-service", "cdn"],
        "incident_type": "performance_critical",
        "business_impact": "Search-driven revenue down 67%. User experience severely degraded. Bounce rate increased 340%.",
        "root_cause": "Elasticsearch cluster split-brain condition with corrupted search indices and insufficient memory allocation",
        "metrics": {
            "search_latency": "8.5s",
            "normal_latency": "0.35s",
            "timeout_rate": "34%",
            "bounce_rate_increase": "340%"
        }
    },
    {
        "title": "Cart Abandonment Rate Spike - 89% vs Normal 23%",
        "description": "Cart abandonment has jumped from normal 23% to critical 89% in last 3 hours. Checkout funnel breaking down at payment step.",
        "severity": "high",
        "affected_systems": ["shopping-cart-service", "checkout-service", "payment-processor", "session-management"],
        "incident_type": "conversion_critical",
        "business_impact": "Potential revenue loss: $78,000. Customer acquisition cost waste. Checkout UX failure.",
        "root_cause": "Session timeout misconfiguration causing cart data loss combined with payment form validation JavaScript errors",
        "metrics": {
            "abandonment_rate": "89%",
            "normal_rate": "23%",
            "checkout_completion": "11%",
            "session_timeout_errors": 3421
        }
    },
    {
        "title": "Refund Request Surge - Electronics Category 340% Increase",
        "description": "Massive spike in refund requests (340% increase) in Electronics category within 4-hour window. Pattern suggests product quality issue.",
        "severity": "high",
        "affected_systems": ["refund-processing", "inventory-management", "customer-service", "product-catalog"],
        "incident_type": "quality_incident",
        "business_impact": "Refund liability: $234,000. Brand reputation risk. Inventory writeoff potential.",
        "root_cause": "Defective batch of smartphones with battery swelling issues shipped to customers from Supplier XYZ",
        "metrics": {
            "refund_requests": 1847,
            "normal_daily_refunds": 134,
            "affected_products": "iPhone cases, Samsung accessories",
            "supplier_batch": "XYZ-2024-Q1-B4"
        }
    },
    {
        "title": "Bot Attack Detection - Abnormal Traffic Pattern 1200% Surge",
        "description": "Sudden traffic surge (1200% increase) with abnormal behavior: no cart additions, repetitive product views, bypassing CAPTCHA.",
        "severity": "critical",  
        "affected_systems": ["web-application-firewall", "bot-detection", "cdn", "rate-limiting"],
        "incident_type": "security_business",
        "business_impact": "Infrastructure costs spiking. Legitimate user performance degraded. Potential scraping/fraud attempt.",
        "root_cause": "Coordinated bot attack from 247 IP addresses attempting product data scraping and price manipulation",
        "metrics": {
            "traffic_increase": "1200%",
            "bot_traffic_ratio": "78%",
            "blocked_requests": 45672,
            "source_ips": 247
        }
    },

    # Financial Services Business Incidents
    {
        "title": "Trading Platform Latency Spike - Order Execution Delays",
        "description": "Stock trading order execution experiencing 3.4s delays vs normal 0.15s. High-frequency trading clients affected during market hours.",
        "severity": "critical",
        "affected_systems": ["trading-engine", "market-data-feed", "order-management", "risk-engine"],
        "incident_type": "trading_critical",
        "business_impact": "Trading revenue loss: $125,000/minute. Regulatory compliance risk. Client SLA breaches.",
        "root_cause": "Market data feed buffer overflow causing processing backlog in trading engine queue system",
        "metrics": {
            "execution_latency": "3.4s",
            "sla_target": "0.15s", 
            "orders_delayed": 8934,
            "hft_clients_affected": 23
        }
    },
    {
        "title": "Credit Card Transaction Failures - Payment Processor Down",
        "description": "78% of credit card transactions failing across all merchant accounts. Payment processor reporting database connectivity issues.",
        "severity": "critical",
        "affected_systems": ["payment-processor", "merchant-gateway", "fraud-detection", "settlement-system"],
        "incident_type": "payment_infrastructure",
        "business_impact": "Transaction revenue halted. Merchant complaints escalating. Potential regulatory notification required.",
        "root_cause": "Primary payment processor database failover mechanism failure during routine maintenance window",
        "metrics": {
            "transaction_failure_rate": "78%",
            "merchants_affected": 1200,
            "transaction_volume_loss": "$2.3M/hour",
            "sla_breach_severity": "Critical"
        }
    },

    # Healthcare Business Incidents  
    {
        "title": "Patient Portal Login Failures - Authentication System Down",
        "description": "Patient portal experiencing 89% login failure rate. Patients unable to access medical records, appointment scheduling broken.",
        "severity": "critical",
        "affected_systems": ["patient-portal", "authentication-service", "medical-records", "appointment-system"],
        "incident_type": "patient_critical",
        "business_impact": "Patient care disruption. Appointment booking halted. Potential HIPAA compliance issues.",
        "root_cause": "LDAP directory service corruption following security patch deployment affecting patient authentication",
        "metrics": {
            "login_failure_rate": "89%",
            "patients_affected": 12400,
            "appointments_missed": 342,
            "portal_availability": "11%"
        }
    },

    # SaaS Business Incidents
    {
        "title": "Multi-Tenant Database Performance Degradation",
        "description": "SaaS application database queries timing out for premium tier customers. Query execution time increased 900% affecting business workflows.",
        "severity": "critical",
        "affected_systems": ["tenant-database", "application-server", "cache-layer", "load-balancer"],
        "incident_type": "saas_performance",
        "business_impact": "Premium customer churn risk. SLA violations. Revenue recognition delays.",
        "root_cause": "Database index corruption on tenant partition table causing full table scans for premium customer queries",
        "metrics": {
            "query_execution_time": "45s",
            "normal_execution": "0.5s",
            "premium_customers_affected": 134,
            "sla_violations": 89
        }
    }
]

# =============================================================================
# COMPLETE ENHANCED WORKFLOW ENGINE WITH DETAILED LOGGING
# =============================================================================

class CompleteEnhancedWorkflowEngine:
    """Complete Enhanced Workflow Engine with ALL features + MCP + A2A + Detailed Logging"""
    
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
        
        # WebSocket connections for real-time updates
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
        """Broadcast MCP updates to all connected WebSocket clients"""
        if self.websocket_connections:
            update_data = {
                "type": "mcp_update",
                "timestamp": datetime.now().isoformat(),
                "total_contexts": len(self.mcp_registry.contexts),
                "latest_context": list(self.mcp_registry.contexts.keys())[-1] if self.mcp_registry.contexts else None
            }
            
            for ws in self.websocket_connections.copy():
                try:
                    await ws.send_text(json.dumps(update_data))
                except:
                    self.websocket_connections.remove(ws)
    
    async def _broadcast_a2a_update(self):
        """Broadcast A2A updates to all connected WebSocket clients"""
        if self.websocket_connections:
            update_data = {
                "type": "a2a_update",
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(self.a2a_protocol.message_history),
                "active_collaborations": len(self.a2a_protocol.active_collaborations),
                "latest_message": self.a2a_protocol.message_history[-1].to_dict() if self.a2a_protocol.message_history else None
            }
            
            for ws in self.websocket_connections.copy():
                try:
                    await ws.send_text(json.dumps(update_data))
                except:
                    self.websocket_connections.remove(ws)
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.websocket_connections)}")
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.websocket_connections)}")
    
    async def trigger_incident_workflow(self, incident_data: Dict[str, Any]) -> Incident:
        """Enhanced incident workflow with business scenarios + MCP + A2A support"""
        scenario = random.choice(BUSINESS_INCIDENT_SCENARIOS)
        incident = Incident(
            title=scenario["title"],
            description=scenario["description"],
            severity=IncidentSeverity(scenario["severity"]),
            affected_systems=scenario["affected_systems"],
            incident_type=scenario["incident_type"],
            business_impact=scenario["business_impact"]
        )
        
        # Create MCP context for enhanced intelligence sharing
        mcp_context = self.mcp_registry.create_context(incident.id, "business_incident_analysis")
        incident.mcp_context_id = mcp_context.context_id
        
        # Set initial shared knowledge with business context
        mcp_context.shared_knowledge.update({
            "incident_metadata": {
                "id": incident.id,
                "type": incident.incident_type,
                "severity": incident.severity.value,
                "affected_systems": incident.affected_systems,
                "business_impact": incident.business_impact,
                "created_at": incident.created_at.isoformat()
            },
            "business_context": scenario,
            "metrics": scenario.get("metrics", {}),
            "expected_resolution_time": self._get_business_resolution_sla(incident.incident_type)
        })
        
        self.active_incidents[incident.id] = incident
        logger.info(f"🚀 Business incident triggered: {incident.title} [{incident.severity.value}]")
        logger.info(f"💼 Business Impact: {incident.business_impact}")
        logger.info(f"🧠 MCP Context: {incident.mcp_context_id}")
        logger.info(f"🤝 A2A Protocol: Ready for business-focused collaboration")
        
        # Start the complete enhanced workflow
        asyncio.create_task(self._execute_complete_enhanced_workflow(incident))
        
        return incident
    
    def _get_business_resolution_sla(self, incident_type: str) -> str:
        """Get business SLA for incident resolution"""
        slas = {
            "business_critical": "30 minutes",
            "payment_critical": "15 minutes", 
            "trading_critical": "5 minutes",
            "patient_critical": "20 minutes",
            "security_business": "45 minutes",
            "performance_critical": "60 minutes",
            "conversion_critical": "45 minutes"
        }
        return slas.get(incident_type, "2 hours")
    
    async def _execute_complete_enhanced_workflow(self, incident: Incident):
        """Execute complete enhanced workflow with ALL 7 agents + MCP + A2A + Detailed Logging"""
        try:
            incident.workflow_status = "in_progress"
            await self._broadcast_workflow_update(incident, f"🚀 Enhanced business incident workflow started: {incident.incident_type}")
            
            # Agent execution order with enhanced business capabilities
            agent_sequence = [
                ("monitoring", self._execute_business_monitoring_agent),
                ("rca", self._execute_business_rca_agent),
                ("pager", self._execute_business_pager_agent),
                ("ticketing", self._execute_business_ticketing_agent),
                ("email", self._execute_business_email_agent),
                ("remediation", self._execute_business_remediation_agent),
                ("validation", self._execute_business_validation_agent)
            ]
            
            for agent_id, agent_function in agent_sequence:
                try:
                    incident.current_agent = agent_id
                    await self._broadcast_workflow_update(incident, f"🔄 Starting business-focused {agent_id} agent analysis")
                    
                    # Process any pending A2A messages for this agent
                    await self._process_a2a_messages(agent_id, incident)
                    
                    # Execute the enhanced business agent
                    execution = await agent_function(incident)
                    incident.executions[agent_id] = execution
                    self.agent_execution_history[agent_id].append(execution)
                    
                    if execution.status == AgentStatus.SUCCESS:
                        incident.completed_agents.append(agent_id)
                        await self._broadcast_workflow_update(incident, f"✅ Business {agent_id} agent completed with {len(execution.logs)} detailed log entries")
                    else:
                        incident.failed_agents.append(agent_id)
                        await self._broadcast_workflow_update(incident, f"❌ Business {agent_id} agent failed: {execution.error_message}")
                    
                    # Realistic delay for complex business analysis
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    
                except Exception as e:
                    logger.error(f"Enhanced business agent {agent_id} failed: {str(e)}")
                    incident.failed_agents.append(agent_id)
                    await self._broadcast_workflow_update(incident, f"💥 Business {agent_id} agent error: {str(e)}")
            
            # Complete the enhanced workflow
            await self._complete_enhanced_workflow(incident)
            
        except Exception as e:
            incident.workflow_status = "failed"
            incident.status = "failed"
            logger.error(f"Enhanced business workflow failed for incident {incident.id}: {str(e)}")
            await self._broadcast_workflow_update(incident, f"💥 Enhanced business workflow failed: {str(e)}")
    
    async def _complete_enhanced_workflow(self, incident: Incident):
        """Complete the enhanced workflow with final updates"""
        try:
            incident.workflow_status = "completed"
            incident.current_agent = ""
            incident.status = "resolved" if len(incident.failed_agents) == 0 else "partially_resolved"
            
            # Final broadcast with business metrics
            business_summary = f"Business incident resolved - {len(incident.completed_agents)}/7 agents successful"
            await self._broadcast_workflow_update(incident, f"🎉 {business_summary} with full MCP+A2A integration")
            
            self.incident_history.append(incident)
            del self.active_incidents[incident.id]
            
        except Exception as e:
            incident.workflow_status = "failed"
            incident.status = "failed"
            logger.error(f"Complete enhanced business workflow failed for incident {incident.id}: {str(e)}")
            await self._broadcast_workflow_update(incident, f"💥 Enhanced business workflow failed: {str(e)}")
    
    async def _broadcast_workflow_update(self, incident: Incident, message: str):
        """Broadcast workflow updates to WebSocket clients"""
        if self.websocket_connections:
            update_data = {
                "type": "workflow_update",
                "incident_id": incident.id,
                "current_agent": incident.current_agent,
                "completed_agents": incident.completed_agents,
                "workflow_status": incident.workflow_status,
                "message": message,
                "business_impact": incident.business_impact,
                "timestamp": datetime.now().isoformat()
            }
            
            for ws in self.websocket_connections.copy():
                try:
                    await ws.send_text(json.dumps(update_data))
                except:
                    self.websocket_connections.remove(ws)
    
    async def _process_a2a_messages(self, agent_id: str, incident: Incident):
        """Process pending A2A messages for an agent"""
        messages = self.a2a_protocol.get_messages(agent_id)
        
        for message in messages:
            logger.info(f"📨 Processing A2A message for {agent_id}: {message.message_type}")
            
            if agent_id in incident.executions:
                incident.executions[agent_id].a2a_messages_received += 1
            
            # Handle different message types
            if message.message_type == "collaboration_request":
                collab_id = message.content.get("collaboration_id")
                if agent_id in incident.executions:
                    incident.executions[agent_id].collaboration_sessions.append(collab_id)
                    incident.executions[agent_id].status = AgentStatus.COLLABORATING
            elif message.message_type == "data_share":
                # Update MCP context with shared data
                mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
                if mcp_context:
                    shared_data = message.content.get("data", {})
                    confidence = message.content.get("confidence", 0.8)
                    mcp_context.update_context(message.sender_agent_id, shared_data, confidence)

    # ENHANCED BUSINESS-FOCUSED AGENT IMPLEMENTATIONS WITH DETAILED LOGGING
    async def _execute_business_monitoring_agent(self, incident: Incident) -> AgentExecution:
        """Business-focused Enhanced Monitoring Agent with comprehensive logging"""
        execution = AgentExecution(
            agent_id="monitoring", agent_name="Business Intelligence Monitoring Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Get MCP context for enhanced business analysis
            mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
            if mcp_context:
                contextual_insights = mcp_context.get_contextual_insights("monitoring")
                execution.contextual_insights_used = contextual_insights
                await self._detailed_log(execution, "🧠 MCP Context loaded - leveraging shared business intelligence", "INFO", {
                    "context_confidence": contextual_insights.get("context_confidence", 0.0),
                    "pagerduty_incident_id": incident.pagerduty_incident_id,
                "servicenow_ticket_id": incident.servicenow_ticket_id,
                "remediation_applied": incident.remediation_applied,
                "business_enhanced_features": {
                    "mcp_context": mcp_data,
                    "a2a_protocol": a2a_data,
                    "business_focus": True,
                    "detailed_agent_logging": True
                },
                "executions": execution_details,
                "log_analytics": {
                    "total_logs_all_agents": sum(len(exec.logs) for exec in incident.executions.values()),
                    "business_context_logs": sum(
                        sum(1 for log in exec.logs if log.get("business_context"))
                        for exec in incident.executions.values()
                    ),
                    "agent_log_distribution": {
                        agent_id: len(exec.logs)
                        for agent_id, exec in incident.executions.items()
                    }
                }
            }
        
        # Enhanced dashboard stats with business intelligence
        @self.app.get("/api/dashboard/stats")
        async def get_complete_business_dashboard_stats():
            all_incidents = list(workflow_engine.active_incidents.values()) + workflow_engine.incident_history
            today_incidents = [i for i in all_incidents if i.created_at.date() == datetime.now().date()]
            business_incidents = [i for i in all_incidents if hasattr(i, 'business_impact') and i.business_impact]
            
            # Agent performance statistics with business focus
            agent_stats = {}
            for agent_id in workflow_engine.agent_execution_history:
                executions = workflow_engine.agent_execution_history[agent_id]
                successful = len([e for e in executions if e.status == AgentStatus.SUCCESS])
                total = len(executions)
                avg_duration = sum([e.duration_seconds for e in executions if e.duration_seconds > 0]) / max(total, 1)
                
                # MCP + A2A + Business specific metrics
                mcp_enhanced = len([e for e in executions if e.contextual_insights_used])
                a2a_messages = sum([e.a2a_messages_sent + e.a2a_messages_received for e in executions])
                collaborations = sum([len(e.collaboration_sessions) for e in executions])
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
                    "business_enhanced_features": {
                        "mcp_enhanced_executions": mcp_enhanced,
                        "mcp_enhancement_rate": (mcp_enhanced / max(total, 1)) * 100,
                        "a2a_messages_total": a2a_messages,
                        "collaborations_total": collaborations,
                        "avg_messages_per_execution": a2a_messages / max(total, 1),
                        "detailed_logging": {
                            "total_logs": total_logs,
                            "business_context_logs": business_logs,
                            "avg_logs_per_execution": total_logs / max(total, 1),
                            "business_log_ratio": (business_logs / max(total_logs, 1)) * 100
                        }
                    },
                    "business_capabilities": workflow_engine.a2a_protocol.agent_capabilities.get(agent_id, [])
                }
            
            # MCP statistics with business focus
            mcp_stats = {
                "total_contexts": len(workflow_engine.mcp_registry.contexts),
                "business_contexts": len([c for c in workflow_engine.mcp_registry.contexts.values() if "business" in c.context_type]),
                "avg_context_confidence": 0.0,
                "total_agent_insights": 0,
                "context_versions_total": 0,
                "business_intelligence_active": True
            }
            
            if workflow_engine.mcp_registry.contexts:
                confidences = []
                insight_counts = []
                version_counts = []
                
                for context in workflow_engine.mcp_registry.contexts.values():
                    if context.confidence_scores:
                        confidences.extend(context.confidence_scores.values())
                    insight_counts.append(len(context.agent_insights))
                    version_counts.append(context.context_version)
                
                mcp_stats.update({
                    "avg_context_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                    "total_agent_insights": sum(insight_counts),
                    "context_versions_total": sum(version_counts),
                    "avg_insights_per_context": sum(insight_counts) / len(insight_counts) if insight_counts else 0,
                    "business_intelligence_enhancement": True
                })
            
            # A2A statistics with business collaboration focus
            a2a_stats = {
                "total_messages": len(workflow_engine.a2a_protocol.message_history),
                "business_collaboration_messages": len([
                    m for m in workflow_engine.a2a_protocol.message_history 
                    if "business" in m.content.get("task", "").lower()
                ]),
                "active_collaborations": len(workflow_engine.a2a_protocol.active_collaborations),
                "registered_agents": len(workflow_engine.a2a_protocol.agent_capabilities),
                "total_capabilities": sum(len(caps) for caps in workflow_engine.a2a_protocol.agent_capabilities.values()),
                "avg_messages_per_incident": 0.0,
                "business_focused_protocol": True
            }
            
            if all_incidents:
                total_messages = sum(
                    sum(exec.a2a_messages_sent + exec.a2a_messages_received for exec in incident.executions.values())
                    for incident in all_incidents
                )
                a2a_stats["avg_messages_per_incident"] = total_messages / len(all_incidents)
            
            # Business intelligence metrics
            business_intelligence = {
                "business_incidents_total": len(business_incidents),
                "business_incident_types": list(set(i.incident_type for i in business_incidents if hasattr(i, 'incident_type'))),
                "avg_business_impact_resolution": sum(
                    1 for i in business_incidents if i.status == "resolved"
                ) / max(len(business_incidents), 1) * 100,
                "critical_business_incidents": len([
                    i for i in business_incidents if i.severity == IncidentSeverity.CRITICAL
                ]),
                "business_sla_compliance": 95.5,  # Simulated metric
                "revenue_protection_incidents": len([
                    i for i in business_incidents if "revenue" in i.business_impact.lower()
                ]) if business_incidents else 0
            }
            
            return {
                "incidents": {
                    "total_all_time": len(all_incidents),
                    "active": len(workflow_engine.active_incidents),
                    "today": len(today_incidents),
                    "resolved_today": len([i for i in today_incidents if i.status == "resolved"]),
                    "business_incidents": len(business_incidents),
                    "business_enhanced_resolution_rate": len([i for i in all_incidents if "MCP+A2A" in str(i.resolution)]) / max(len(all_incidents), 1) * 100,
                    "average_resolution_time_minutes": 12.5,
                    "incident_types_distribution": {
                        incident_type: len([i for i in all_incidents if hasattr(i, 'incident_type') and i.incident_type == incident_type])
                        for incident_type in set(i.incident_type for i in all_incidents if hasattr(i, 'incident_type') and i.incident_type)
                    }
                },
                "agents": agent_stats,
                "business_enhanced_features": {
                    "mcp": mcp_stats,
                    "a2a": a2a_stats,
                    "business_intelligence": business_intelligence
                },
                "system": {
                    "version": "4.0.0 - Complete Business Intelligence MCP+A2A Enhanced",
                    "architecture": [
                        "All 7 Business-Enhanced Specialized Agents",
                        "Model Context Protocol with Business Intelligence", 
                        "Agent-to-Agent Communication with Business Collaboration",
                        "Real-time WebSocket Updates",
                        "Comprehensive Business Analysis",
                        "Detailed Agent Console Logging",
                        "Business-Centric Incident Scenarios"
                    ],
                    "uptime_hours": 24,
                    "total_workflows": len(all_incidents),
                    "successful_workflows": len([i for i in all_incidents if i.status == "resolved"]),
                    "overall_success_rate": (len([i for i in all_incidents if i.status == "resolved"]) / max(len(all_incidents), 1)) * 100,
                    "available_business_scenarios": len(BUSINESS_INCIDENT_SCENARIOS),
                    "websocket_connections": len(workflow_engine.websocket_connections),
                    "business_intelligence_active": True,
                    "detailed_logging_active": True
                },
                "logging_analytics": {
                    "total_logs_system_wide": sum(
                        sum(len(exec.logs) for exec in incident.executions.values())
                        for incident in all_incidents
                    ),
                    "business_context_logs": sum(
                        sum(
                            sum(1 for log in exec.logs if log.get("business_context"))
                            for exec in incident.executions.values()
                        )
                        for incident in all_incidents
                    ),
                    "avg_logs_per_agent_per_incident": sum(
                        sum(len(exec.logs) for exec in incident.executions.values())
                        for incident in all_incidents
                    ) / max(len(all_incidents) * 7, 1)
                }
            }
        
        # Enhanced agents endpoint with business capabilities
        @self.app.get("/api/agents")
        async def get_complete_business_agents():
            agent_configs = {
                "monitoring": "Business Intelligence Monitoring Agent - Real-time business metrics analysis, revenue impact assessment, customer experience monitoring, and performance degradation detection with comprehensive logging",
                "rca": "Business Impact Root Cause Analysis Agent - AI-powered business-focused root cause analysis with financial impact correlation, customer impact assessment, and cross-business dependency mapping with detailed analysis logging", 
                "pager": "Business Stakeholder Escalation Agent - Intelligent escalation to business owners and technical teams with context-aware notification routing, executive briefing, and business impact communication with comprehensive logging",
                "ticketing": "Business Impact Ticketing Agent - Smart business-focused ticket classification with SLA tracking, business priority assessment, stakeholder assignment, and comprehensive business context logging",
                "email": "Business Stakeholder Communication Agent - Context-aware business stakeholder notifications with personalized messaging, executive reporting, customer communication, and detailed communication logging",
                "remediation": "Business Continuity Remediation Agent - Business continuity focused automated remediation with revenue protection, customer impact mitigation, and operational continuity with comprehensive action logging",
                "validation": "Business Continuity Validation Agent - Comprehensive business continuity verification with revenue flow validation, customer experience testing, and business metrics verification with detailed validation logging"
            }
            
            agents_data = {}
            for agent_id, description in agent_configs.items():
                executions = workflow_engine.agent_execution_history[agent_id]
                recent = executions[-1] if executions else None
                
                # Calculate enhanced statistics
                successful_count = len([e for e in executions if e.status == AgentStatus.SUCCESS])
                total_count = len(executions)
                avg_duration = sum([e.duration_seconds for e in executions if e.duration_seconds > 0]) / max(total_count, 1)
                
                # MCP + A2A + Logging specific stats
                mcp_enhanced_count = len([e for e in executions if e.contextual_insights_used])
                a2a_messages_total = sum([e.a2a_messages_sent + e.a2a_messages_received for e in executions])
                collaborations_total = sum([len(e.collaboration_sessions) for e in executions])
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
                    "last_activity": recent.started_at.isoformat() if recent and recent.started_at else "Never",
                    "business_enhanced_features": {
                        "business_intelligence_focus": True,
                        "mcp_enhanced_executions": mcp_enhanced_count,
                        "mcp_enhancement_rate": (mcp_enhanced_count / max(total_count, 1)) * 100,
                        "a2a_messages_total": a2a_messages_total,
                        "collaborations_total": collaborations_total,
                        "avg_messages_per_execution": a2a_messages_total / max(total_count, 1),
                        "detailed_logging": {
                            "total_logs": total_logs,
                            "business_context_logs": business_logs,
                            "avg_logs_per_execution": total_logs / max(total_count, 1),
                            "business_log_percentage": (business_logs / max(total_logs, 1)) * 100,
                            "detailed_logging_active": True
                        },
                        "context_aware": True,
                        "real_time_updates": True
                    },
                    "business_capabilities": workflow_engine.a2a_protocol.agent_capabilities.get(agent_id, []),
                    "recent_performance": {
                        "last_execution_status": recent.status.value if recent else "idle",
                        "last_duration": recent.duration_seconds if recent else 0,
                        "last_progress": recent.progress if recent else 0,
                        "last_log_count": len(recent.logs) if recent else 0
                    }
                }
            
            return {
                "agents": agents_data, 
                "total_agents": 7,
                "system_capabilities": {
                    "business_intelligence_monitoring": True,
                    "mcp_context_sharing": True,
                    "a2a_communication": True,
                    "real_time_collaboration": True,
                    "comprehensive_business_analysis": True,
                    "detailed_agent_logging": True,
                    "business_continuity_focus": True
                }
            }
        
        # Get recent business incidents
        @self.app.get("/api/incidents")
        async def get_recent_business_incidents(limit: int = 10):
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
                        "business_impact": getattr(incident, 'business_impact', 'Business impact assessment pending'),
                        "status": incident.status,
                        "workflow_status": incident.workflow_status,
                        "current_agent": incident.current_agent,
                        "completed_agents": incident.completed_agents,
                        "failed_agents": incident.failed_agents,
                        "created_at": incident.created_at.isoformat(),
                        "affected_systems": incident.affected_systems,
                        "mcp_context_id": incident.mcp_context_id,
                        "a2a_collaborations": len(incident.a2a_collaborations),
                        "business_enhanced": True,
                        "detailed_logs_available": sum(len(exec.logs) for exec in incident.executions.values()),
                        "agent_log_summary": {
                            agent_id: {
                                "log_count": len(exec.logs),
                                "business_logs": sum(1 for log in exec.logs if log.get("business_context")),
                                "status": exec.status.value
                            }
                            for agent_id, exec in incident.executions.items()
                        }
                    }
                    for incident in recent_incidents
                ],
                "total_incidents": len(all_incidents),
                "business_incidents": len([i for i in all_incidents if hasattr(i, 'business_impact') and i.business_impact])
            }
        
        # WebSocket endpoint for real-time updates
        @self.app.websocket("/ws/realtime")
        async def websocket_realtime_updates(websocket: WebSocket):
            await websocket.accept()
            await workflow_engine.add_websocket_connection(websocket)
            
            try:
                # Send initial status with business intelligence
                initial_data = {
                    "type": "connection_established",
                    "message": "Real-time business intelligence updates connected",
                    "timestamp": datetime.now().isoformat(),
                    "features": [
                        "Business-Centric MCP Context Updates", 
                        "A2A Business Collaboration Tracking", 
                        "Detailed Agent Logging Updates",
                        "Business Workflow Progress",
                        "Revenue Impact Monitoring"
                    ]
                }
                await websocket.send_text(json.dumps(initial_data))
                
                # Keep connection alive and handle client messages
                while True:
                    try:
                        data = await websocket.receive_text()
                        # Echo back for connection verification
                        response = {
                            "type": "echo",
                            "received": data,
                            "timestamp": datetime.now().isoformat(),
                            "business_intelligence_active": True
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
        
        # Enhanced health check with business intelligence status
        @self.app.get("/health")
        async def complete_business_health_check():
            return {
                "status": "healthy",
                "service": "Complete Business Intelligence MCP + A2A Enhanced AI Monitoring System",
                "version": "4.0.0",
                "architecture": {
                    "all_seven_business_agents": "Active",
                    "model_context_protocol": "Active with Business Intelligence",
                    "agent_to_agent_protocol": "Active with Business Collaboration",
                    "real_time_updates": "Active",
                    "cross_agent_intelligence": "Enabled",
                    "detailed_agent_logging": "Active",
                    "business_intelligence": "Enabled"
                },
                "enhanced_features": [
                    "All 7 Business-Enhanced Specialized Agents",
                    "Model Context Protocol (MCP) with Business Intelligence",
                    "Agent-to-Agent (A2A) Communication with Business Collaboration",
                    "Real-time WebSocket Updates",
                    "Cross-agent business intelligence sharing",
                    "Contextual business decision making",
                    "Collaborative business problem solving",
                    "Business-centric incident scenarios",
                    "Comprehensive business analysis and validation",
                    "Detailed agent console logging with business context",
                    "Revenue impact monitoring and protection",
                    "Customer experience tracking and optimization"
                ],
                "business_scenarios": {
                    "total_available": len(BUSINESS_INCIDENT_SCENARIOS),
                    "categories": list(set(s["incident_type"] for s in BUSINESS_INCIDENT_SCENARIOS)),
                    "business_focus": [
                        "E-commerce Operations",
                        "Payment Processing",
                        "Customer Experience",
                        "Revenue Protection",
                        "Trading Operations",
                        "Healthcare Patient Care",
                        "SaaS Platform Management"
                    ]
                },
                "workflow_engine": {
                    "active_incidents": len(workflow_engine.active_incidents),
                    "total_incidents": len(workflow_engine.incident_history) + len(workflow_engine.active_incidents),
                    "mcp_contexts": len(workflow_engine.mcp_registry.contexts),
                    "a2a_collaborations": len(workflow_engine.a2a_protocol.active_collaborations),
                    "total_agent_messages": len(workflow_engine.a2a_protocol.message_history),
                    "websocket_connections": len(workflow_engine.websocket_connections),
                    "business_intelligence_active": True
                },
                "agents_status": {
                    agent_id: {
                        "total_executions": len(executions),
                        "capabilities": len(workflow_engine.a2a_protocol.agent_capabilities.get(agent_id, [])),
                        "status": "ready",
                        "business_enhanced": True,
                        "detailed_logging": True,
                        "total_logs": sum(len(exec.logs) for exec in executions),
                        "business_context_logs": sum(
                            sum(1 for log in exec.logs if log.get("business_context"))
                            for exec in executions
                        )
                    }
                    for agent_id, executions in workflow_engine.agent_execution_history.items()
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
                    "message": "🚀 Complete Business Intelligence MCP + A2A Enhanced AI Monitoring System v4.0",
                    "version": "4.0.0",
                    "architecture": "ALL Previous Features + Model Context Protocol + Agent-to-Agent Communication + Business Intelligence + Detailed Agent Logging",
                    "key_enhancements": [
                        "✅ All 7 Agents Dashboard - Fully Enhanced with Business Intelligence",
                        "✅ Real-time Progress Tracking - Business-Focused",
                        "✅ WebSocket Live Updates - Business Intelligence Stream",
                        "✅ Agent Execution History - Complete Business Context",
                        "✅ Detailed Console Logs - RESTORED with Business Context",
                        "🆕 Business-Centric Incident Scenarios - 11 Real Business Cases",
                        "🆕 Revenue Impact Monitoring and Protection",
                        "🆕 Customer Experience Tracking and Optimization"
                    ],
                    "advanced_features": [
                        "🧠 Model Context Protocol - Business intelligence sharing across agents",
                        "🤝 Agent-to-Agent Protocol - Business-focused collaboration",
                        "🔗 Cross-agent business collaboration and coordination",
                        "📊 Contextual business decision making with historical insights",
                        "🎯 Enhanced business accuracy through collective intelligence",
                        "📈 Real-time business knowledge sharing and learning",
                        "💼 Business continuity focus with revenue protection",
                        "👥 Customer impact assessment and mitigation"
                    ],
                    "business_scenarios": [
                        "Order Processing Delays with Revenue Impact",
                        "Payment Failures with Regional Analysis",
                        "Customer Experience Degradation",
                        "Cart Abandonment Rate Spikes",
                        "Performance Issues with Conversion Impact",
                        "Refund Request Surges",
                        "Bot Attacks with Business Impact",
                        "Trading Platform Latency Issues",
                        "Patient Portal Access Problems",
                        "SaaS Performance Degradation",
                        "Credit Card Processing Failures"
                    ],
                    "detailed_logging": {
                        "feature": "RESTORED and ENHANCED",
                        "access": "Click any agent to view detailed console logs",
                        "content": "Business context, technical analysis, MCP insights, A2A communications",
                        "format": "Structured logs with emoji indicators and categorization"
                    },
                    "capabilities": [
                        "All 7 specialized agents with business intelligence capabilities",
                        "Real-time MCP business context updates via WebSocket",
                        "Live A2A business collaboration tracking and monitoring",
                        "Comprehensive business incident analysis with financial impact assessment",
                        "Enhanced business decision making through collective agent intelligence",
                        "Detailed agent console logging with complete business context and analysis depth"
                    ]
                }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        logger.info("🚀 Starting Complete Business Intelligence MCP + A2A Enhanced AI Monitoring System v4.0...")
        logger.info("✅ ALL 7 BUSINESS-ENHANCED AGENTS: ACTIVE")
        logger.info("🧠 Model Context Protocol with Business Intelligence: ACTIVE")
        logger.info("🤝 Agent-to-Agent Protocol with Business Collaboration: ACTIVE")
        logger.info("🔗 Real-time Updates with Business Intelligence: ENABLED")
        logger.info("📊 Cross-agent Business Intelligence: OPERATIONAL")
        logger.info("📝 DETAILED AGENT CONSOLE LOGS: RESTORED AND ENHANCED")
        logger.info("💼 Business-Centric Incident Scenarios: LOADED")
        logger.info("💰 Revenue Impact Monitoring: ACTIVE")
        logger.info("👥 Customer Experience Tracking: ENABLED")
        logger.info(f"🌐 Complete Business Intelligence Dashboard: http://localhost:{port}")
        logger.info("🎯 Click any agent to view detailed console logs with business context!")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    app = CompleteEnhancedMonitoringApp()
    app.run()
eer_insights_available": len(contextual_insights.get("peer_insights", {}))
                })
            
            await self._detailed_log(execution, f"🔍 Initiating comprehensive business monitoring for {incident.incident_type}", "INFO", {
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact,
                "affected_systems": incident.affected_systems
            })
            execution.progress = 10
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Business-specific monitoring based on incident type
            if incident.incident_type == "business_critical":
                await self._detailed_log(execution, "📊 Analyzing critical business KPIs: order processing times, revenue impact, customer satisfaction metrics", "INFO")
                execution.progress = 25
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                await self._detailed_log(execution, "💰 Revenue impact analysis: calculating per-minute loss, affected customer segments", "ANALYSIS", {
                    "revenue_loss_per_minute": "$2,500",
                    "orders_in_queue": 1847,
                    "customer_segments_affected": ["premium", "enterprise"]
                })
                execution.progress = 45
                await asyncio.sleep(random.uniform(1.0, 1.5))
                
                # A2A collaboration with RCA agent for business analysis
                collab_id = self.a2a_protocol.initiate_collaboration(
                    "monitoring", ["rca"], 
                    "business_impact_correlation_analysis",
                    {
                        "incident_type": incident.incident_type, 
                        "revenue_impact": "$2,500/min",
                        "customer_impact": "critical"
                    }
                )
                execution.collaboration_sessions.append(collab_id)
                
                await self._detailed_log(execution, f"🤝 A2A Collaboration initiated with RCA agent for business correlation", "COLLABORATION", {
                    "collaboration_id": collab_id,
                    "task": "business_impact_correlation_analysis"
                })
                
                execution.output_data = {
                    "business_metrics": {
                        "revenue_loss_per_minute": 2500,
                        "orders_affected": 1847,
                        "processing_time_degradation": "580%",
                        "customer_satisfaction_risk": "high"
                    },
                    "technical_metrics": {
                        "payment_service_latency": "15.7s",
                        "database_connection_pool": "98% utilization",
                        "queue_depth": 15420
                    },
                    "mcp_enhanced": True,
                    "collaboration_initiated": True
                }
                
            elif incident.incident_type == "payment_critical":
                await self._detailed_log(execution, "💳 Payment system comprehensive monitoring: transaction flows, failure patterns, regional analysis", "INFO")
                execution.progress = 20
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                await self._detailed_log(execution, "🌍 Regional payment analysis: EU region showing 89% failure rate via Stripe integration", "ANALYSIS", {
                    "failure_rate_eu": "89%",
                    "normal_failure_rate": "2.1%",
                    "payment_provider": "Stripe",
                    "affected_countries": ["DE", "FR", "IT", "ES"]
                })
                execution.progress = 50
                await asyncio.sleep(random.uniform(1.0, 1.5))
                
                await self._detailed_log(execution, "🔍 Payment provider webhook analysis: SSL certificate validation failures detected", "TECHNICAL", {
                    "webhook_failures": 2341,
                    "ssl_cert_status": "validation_failed",
                    "retry_mechanism": "malfunctioning"
                })
                
                # Share payment intelligence with remediation agent
                payment_data = {
                    "payment_provider": "Stripe",
                    "failure_pattern": "ssl_webhook_validation",
                    "regional_impact": "EU_only",
                    "business_priority": "critical"
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
                
                await self._detailed_log(execution, "📨 Critical payment data shared with remediation agent via A2A protocol", "A2A_COMMUNICATION", {
                    "recipient": "remediation",
                    "data_confidence": 0.94,
                    "priority": "critical"
                })
                
                execution.output_data = {
                    "payment_analysis": {
                        "provider": "Stripe",
                        "failure_rate": "89%",
                        "regional_impact": "EU",
                        "ssl_issue": True,
                        "webhook_failures": 2341
                    },
                    "business_impact": {
                        "lost_revenue_per_hour": 45000,
                        "customer_segments": ["premium", "international"],
                        "compliance_risk": "moderate"
                    },
                    "a2a_intelligence_shared": True
                }
                
            elif incident.incident_type == "performance_critical":
                await self._detailed_log(execution, "⚡ Performance degradation analysis: search latency, conversion impact, user experience metrics", "INFO")
                execution.progress = 30
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                await self._detailed_log(execution, "🔍 Search performance deep dive: 8.5s average latency, 34% timeout rate, 67% conversion drop", "ANALYSIS", {
                    "search_latency": "8.5s",
                    "normal_latency": "0.35s",
                    "timeout_rate": "34%",
                    "conversion_impact": "-67%"
                })
                execution.progress = 55
                await asyncio.sleep(random.uniform(1.0, 1.5))
                
                await self._detailed_log(execution, "📊 Elasticsearch cluster analysis: split-brain condition detected, index corruption identified", "TECHNICAL", {
                    "cluster_status": "split_brain",
                    "corrupted_indices": 3,
                    "memory_utilization": "97%",
                    "heap_usage": "critical"
                })
                
                execution.output_data = {
                    "performance_metrics": {
                        "search_latency": 8.5,
                        "timeout_rate": 34,
                        "bounce_rate_increase": 340,
                        "conversion_drop": 67
                    },
                    "technical_diagnosis": {
                        "elasticsearch_status": "split_brain",
                        "index_corruption": True,
                        "memory_pressure": "critical"
                    },
                    "business_context": "search_revenue_critical"
                }
                
            else:
                await self._detailed_log(execution, f"📈 General business monitoring for {incident.incident_type}: KPI analysis, system health, customer impact", "INFO")
                execution.progress = 40
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                execution.output_data = {
                    "general_metrics": {
                        "business_impact_score": random.randint(70, 95),
                        "system_health": "degraded",
                        "customer_impact": "moderate_to_high"
                    }
                }
            
            # Update MCP context with comprehensive business findings
            if mcp_context:
                mcp_context.update_context("monitoring", execution.output_data, 0.93)
                execution.contextual_insights_used["updated_context"] = True
                await self._detailed_log(execution, "🧠 MCP Context updated with comprehensive business monitoring data", "MCP_UPDATE", {
                    "confidence_score": 0.93,
                    "data_points": len(execution.output_data)
                })
            
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ Business monitoring analysis completed successfully", "SUCCESS", {
                "total_log_entries": len(execution.logs),
                "mcp_enhanced": True,
                "a2a_messages_sent": execution.a2a_messages_sent,
                "collaborations": len(execution.collaboration_sessions)
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Business monitoring analysis failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    async def _execute_business_rca_agent(self, incident: Incident) -> AgentExecution:
        """Business-focused Enhanced RCA Agent with comprehensive analysis logging"""
        execution = AgentExecution(
            agent_id="rca", agent_name="Business Impact Root Cause Analysis Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Get comprehensive context from MCP
            mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
            contextual_data = {}
            if mcp_context:
                contextual_data = mcp_context.get_contextual_insights("rca")
                execution.contextual_insights_used = contextual_data
                await self._detailed_log(execution, "🧠 Comprehensive MCP context analysis initiated", "MCP_ANALYSIS", {
                    "peer_insights_count": len(contextual_data.get("peer_insights", {})),
                    "context_confidence": contextual_data.get("context_confidence", 0.0),
                    "shared_knowledge_keys": len(contextual_data.get("shared_knowledge", {}))
                })
            
            await self._detailed_log(execution, f"🔬 Advanced business-focused root cause analysis for {incident.incident_type}", "INFO", {
                "analysis_type": "business_impact_focused",
                "incident_severity": incident.severity.value,
                "business_impact": incident.business_impact
            })
            execution.progress = 15
            await asyncio.sleep(random.uniform(2.0, 2.5))
            
            # Enhanced analysis using contextual insights
            confidence_boost = 0.0
            if contextual_data.get("peer_insights"):
                confidence_boost = 0.20
                await self._detailed_log(execution, "💡 Leveraging peer agent insights for enhanced root cause correlation", "PEER_ANALYSIS", {
                    "monitoring_insights": "payment_latency_patterns" if "payment" in incident.incident_type else "performance_degradation",
                    "confidence_boost": confidence_boost,
                    "correlation_patterns": len(contextual_data.get("correlation_patterns", []))
                })
                execution.progress = 35
                await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # Get scenario-specific root cause with business context
            scenario = None
            for s in BUSINESS_INCIDENT_SCENARIOS:
                if s["title"] == incident.title:
                    scenario = s
                    break
            
            if scenario:
                await self._detailed_log(execution, f"📋 Business scenario analysis: {scenario['incident_type']}", "BUSINESS_ANALYSIS", {
                    "business_impact": scenario["business_impact"],
                    "affected_systems": len(scenario["affected_systems"]),
                    "metrics_available": bool(scenario.get("metrics"))
                })
                execution.progress = 55
                await asyncio.sleep(random.uniform(1.5, 2.0))
                
                root_cause = scenario["root_cause"]
                business_correlation = self._analyze_business_correlation(scenario)
                
                await self._detailed_log(execution, f"🎯 Root cause identified with business correlation analysis", "ROOT_CAUSE", {
                    "primary_cause": root_cause,
                    "business_correlation": business_correlation,
                    "technical_component": scenario["affected_systems"][0] if scenario["affected_systems"] else "unknown"
                })
            else:
                root_cause = f"Business-critical {incident.incident_type} issue requiring comprehensive investigation"
                business_correlation = "general_business_impact"
            
            base_confidence = random.uniform(0.88, 0.96)
            enhanced_confidence = min(0.99, base_confidence + confidence_boost)
            
            # Business impact analysis
            financial_impact = self._calculate_financial_impact(incident, scenario)
            customer_impact = self._assess_customer_impact(incident, scenario)
            
            await self._detailed_log(execution, "💰 Financial impact analysis completed", "FINANCIAL_ANALYSIS", financial_impact)
            await self._detailed_log(execution, "👥 Customer impact assessment completed", "CUSTOMER_ANALYSIS", customer_impact)
            
            execution.output_data = {
                "root_cause": root_cause,
                "confidence": enhanced_confidence,
                "business_analysis": {
                    "financial_impact": financial_impact,
                    "customer_impact": customer_impact,
                    "business_correlation": business_correlation,
                    "recovery_priority": self._get_recovery_priority(incident.incident_type)
                },
                "technical_analysis": {
                    "primary_component": scenario["affected_systems"][0] if scenario and scenario["affected_systems"] else "unknown",
                    "failure_pattern": self._identify_failure_pattern(incident.incident_type),
                    "dependency_impact": "high" if len(incident.affected_systems) > 2 else "medium"
                },
                "mcp_enhanced": True,
                "used_peer_insights": bool(contextual_data.get("peer_insights")),
                "context_confidence": contextual_data.get("context_confidence", 0.0)
            }
            
            # Share comprehensive RCA findings with multiple agents via A2A
            rca_findings = {
                "root_cause_summary": root_cause,
                "confidence_score": enhanced_confidence,
                "business_priority_actions": self._get_business_priority_actions(incident.incident_type),
                "financial_impact": financial_impact,
                "recovery_timeline": self._get_recovery_timeline(incident.incident_type)
            }
            
            # Share with multiple relevant agents
            for agent in ["remediation", "validation", "pager", "email"]:
                message = A2AMessage(
                    sender_agent_id="rca",
                    receiver_agent_id=agent,
                    message_type="data_share",
                    content={"data": rca_findings, "confidence": enhanced_confidence},
                    priority="high"
                )
                self.a2a_protocol.send_message(message)
                execution.a2a_messages_sent += 1
                
                await self._detailed_log(execution, f"📨 RCA findings shared with {agent} agent", "A2A_SHARE", {
                    "recipient": agent,
                    "confidence": enhanced_confidence,
                    "data_type": "comprehensive_rca_findings"
                })
            
            # Update MCP context with comprehensive RCA findings
            if mcp_context:
                mcp_context.update_context("rca", execution.output_data, enhanced_confidence)
                await self._detailed_log(execution, "🧠 MCP Context updated with comprehensive RCA analysis", "MCP_UPDATE", {
                    "confidence_score": enhanced_confidence,
                    "business_insights": True,
                    "technical_insights": True
                })
            
            incident.root_cause = execution.output_data["root_cause"]
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ Business-focused RCA analysis completed successfully", "SUCCESS", {
                "confidence": f"{enhanced_confidence:.1%}",
                "business_analysis": "completed",
                "technical_analysis": "completed",
                "a2a_messages_sent": execution.a2a_messages_sent,
                "total_log_entries": len(execution.logs)
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Business RCA analysis failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    def _analyze_business_correlation(self, scenario: Dict[str, Any]) -> str:
        """Analyze business correlation patterns"""
        if "payment" in scenario["incident_type"]:
            return "revenue_direct_impact"
        elif "performance" in scenario["incident_type"]:
            return "customer_experience_degradation"
        elif "security" in scenario["incident_type"]:
            return "trust_and_compliance_risk"
        else:
            return "operational_efficiency_impact"
    
    def _calculate_financial_impact(self, incident: Incident, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed financial impact"""
        if scenario and scenario.get("metrics"):
            metrics = scenario["metrics"]
            return {
                "immediate_loss": metrics.get("revenue_at_risk", "$0"),
                "hourly_impact": metrics.get("revenue_loss_per_hour", "$0"),
                "affected_transactions": metrics.get("orders_affected", 0),
                "impact_category": "critical" if incident.severity == IncidentSeverity.CRITICAL else "moderate"
            }
        else:
            return {
                "immediate_loss": f"${random.randint(10000, 150000)}",
                "hourly_impact": f"${random.randint(5000, 50000)}",
                "affected_transactions": random.randint(100, 5000),
                "impact_category": incident.severity.value
            }
    
    def _assess_customer_impact(self, incident: Incident, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Assess detailed customer impact"""
        return {
            "affected_customers": random.randint(1000, 25000),
            "experience_degradation": "severe" if incident.severity == IncidentSeverity.CRITICAL else "moderate",
            "churn_risk": "high" if "critical" in incident.incident_type else "medium",
            "support_ticket_increase": f"{random.randint(200, 800)}%"
        }
    
    def _identify_failure_pattern(self, incident_type: str) -> str:
        """Identify technical failure pattern"""
        patterns = {
            "business_critical": "synchronous_processing_bottleneck",
            "payment_critical": "integration_failure_cascade", 
            "performance_critical": "resource_exhaustion_spiral",
            "security_business": "attack_vector_exploitation"
        }
        return patterns.get(incident_type, "system_degradation_cascade")
    
    def _get_recovery_priority(self, incident_type: str) -> str:
        """Get business recovery priority"""
        priorities = {
            "business_critical": "revenue_protection_first",
            "payment_critical": "payment_flow_restoration",
            "performance_critical": "user_experience_recovery",
            "trading_critical": "market_position_protection"
        }
        return priorities.get(incident_type, "service_restoration")
    
    def _get_business_priority_actions(self, incident_type: str) -> List[str]:
        """Get business-focused priority actions"""
        actions = {
            "business_critical": [
                "restore_order_processing_flow",
                "activate_backup_payment_systems", 
                "notify_high_value_customers",
                "enable_priority_queue_processing"
            ],
            "payment_critical": [
                "switch_to_backup_payment_provider",
                "notify_affected_merchants",
                "implement_transaction_retry_logic",
                "activate_fraud_detection_bypass"
            ],
            "performance_critical": [
                "implement_search_result_caching",
                "activate_cdn_optimization",
                "enable_search_result_preloading",
                "implement_graceful_degradation"
            ]
        }
        return actions.get(incident_type, [
            "restore_core_functionality",
            "minimize_customer_impact",
            "implement_workaround_solutions"
        ])
    
    def _get_recovery_timeline(self, incident_type: str) -> str:
        """Get expected recovery timeline"""
        timelines = {
            "business_critical": "15-30 minutes",
            "payment_critical": "10-20 minutes",
            "performance_critical": "30-60 minutes",
            "trading_critical": "5-15 minutes"
        }
        return timelines.get(incident_type, "30-90 minutes")
    
    async def _execute_business_pager_agent(self, incident: Incident) -> AgentExecution:
        """Business-focused Enhanced Pager Agent with stakeholder-aware escalation"""
        execution = AgentExecution(
            agent_id="pager", agent_name="Business Stakeholder Escalation Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            await self._detailed_log(execution, f"📞 Business stakeholder escalation analysis for {incident.incident_type}", "INFO", {
                "business_impact": incident.business_impact,
                "severity": incident.severity.value,
                "incident_type": incident.incident_type
            })
            execution.progress = 20
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Business stakeholder identification
            stakeholders = self._identify_business_stakeholders(incident)
            escalation_strategy = self._determine_escalation_strategy(incident)
            
            await self._detailed_log(execution, f"👥 Business stakeholder identification completed", "STAKEHOLDER_ANALYSIS", {
                "primary_stakeholders": stakeholders["primary"],
                "secondary_stakeholders": stakeholders["secondary"],
                "executive_notification": stakeholders["executive_required"],
                "escalation_strategy": escalation_strategy
            })
            execution.progress = 50
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Enhanced team selection based on business context
            technical_team = self._get_business_technical_team(incident.incident_type, incident.severity.value)
            business_owner = self._get_business_owner(incident.incident_type)
            
            await self._detailed_log(execution, f"🎯 Team assignment and business owner notification", "TEAM_ASSIGNMENT", {
                "technical_team": technical_team,
                "business_owner": business_owner,
                "notification_channels": ["PagerDuty", "Slack", "Email", "SMS"],
                "escalation_timeline": escalation_strategy["timeline"]
            })
            execution.progress = 75
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # A2A coordination with email agent for unified communication
            coord_message = A2AMessage(
                sender_agent_id="pager",
                receiver_agent_id="email",
                message_type="collaboration_request",
                content={
                    "task": "coordinated_business_stakeholder_notification",
                    "technical_team": technical_team,
                    "business_owner": business_owner,
                    "stakeholders": stakeholders,
                    "escalation_strategy": escalation_strategy,
                    "incident_details": {
                        "type": incident.incident_type,
                        "severity": incident.severity.value,
                        "business_impact": incident.business_impact,
                        "financial_impact": "critical"
                    }
                },
                priority="high"
            )
            self.a2a_protocol.send_message(coord_message)
            execution.a2a_messages_sent += 1
            
            await self._detailed_log(execution, f"🤝 A2A coordination with email agent for unified stakeholder communication", "A2A_COORDINATION", {
                "collaboration_type": "stakeholder_notification",
                "coordination_agent": "email",
                "stakeholder_groups": len(stakeholders),
                "notification_priority": "high"
            })
            
            execution.output_data = {
                "pagerduty_incident_id": f"BIZ-{incident.incident_type.upper()}-{incident.id[-6:]}",
                "business_escalation": {
                    "technical_team": technical_team,
                    "business_owner": business_owner,
                    "stakeholders": stakeholders,
                    "escalation_strategy": escalation_strategy
                },
                "notification_channels": ["PagerDuty", "Business Slack", "Executive Email", "SMS"],
                "business_sla": self._get_business_resolution_sla(incident.incident_type),
                "coordinated_notification": True,
                "mcp_context_used": True
            }
            
            incident.pagerduty_incident_id = execution.output_data["pagerduty_incident_id"]
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ Business stakeholder escalation completed successfully", "SUCCESS", {
                "pagerduty_id": execution.output_data["pagerduty_incident_id"],
                "stakeholders_notified": len(stakeholders["primary"]) + len(stakeholders["secondary"]),
                "business_owner": business_owner,
                "technical_team": technical_team,
                "total_log_entries": len(execution.logs)
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Business stakeholder escalation failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    def _identify_business_stakeholders(self, incident: Incident) -> Dict[str, List[str]]:
        """Identify business stakeholders based on incident type and impact"""
        business_stakeholders = {
            "business_critical": {
                "primary": ["VP Operations", "Customer Success Manager", "Revenue Operations"],
                "secondary": ["Product Manager", "Marketing Director"],
                "executive_required": True
            },
            "payment_critical": {
                "primary": ["CFO", "Payment Operations Manager", "Compliance Officer"],
                "secondary": ["Customer Support Director", "Risk Management"],
                "executive_required": True
            },
            "performance_critical": {
                "primary": ["VP Engineering", "Customer Experience Manager", "Product Owner"],
                "secondary": ["Marketing Manager", "Sales Operations"],
                "executive_required": False
            },
            "trading_critical": {
                "primary": ["Head of Trading", "Risk Manager", "Compliance Officer"],
                "secondary": ["Client Relations", "Operations Manager"],
                "executive_required": True
            }
        }
        
        return business_stakeholders.get(incident.incident_type, {
            "primary": ["Operations Manager", "Technical Lead"],
            "secondary": ["Product Manager"],
            "executive_required": incident.severity == IncidentSeverity.CRITICAL
        })
    
    def _determine_escalation_strategy(self, incident: Incident) -> Dict[str, Any]:
        """Determine escalation strategy based on business context"""
        if incident.severity == IncidentSeverity.CRITICAL:
            return {
                "timeline": "immediate",
                "frequency": "every_5_minutes",
                "channels": ["phone", "sms", "slack", "email"],
                "executive_briefing": True
            }
        elif incident.severity == IncidentSeverity.HIGH:
            return {
                "timeline": "within_10_minutes",
                "frequency": "every_15_minutes", 
                "channels": ["slack", "email", "sms"],
                "executive_briefing": False
            }
        else:
            return {
                "timeline": "within_30_minutes",
                "frequency": "every_30_minutes",
                "channels": ["slack", "email"],
                "executive_briefing": False
            }
    
    def _get_business_technical_team(self, incident_type: str, severity: str) -> str:
        """Get business-focused technical team"""
        teams = {
            "business_critical": "E-commerce Platform Team",
            "payment_critical": "Payment Infrastructure Team", 
            "performance_critical": "Site Reliability Engineering",
            "trading_critical": "Trading Systems Team",
            "security_business": "Security Operations Center"
        }
        base_team = teams.get(incident_type, "Platform Engineering")
        
        if severity == "critical":
            return f"Senior {base_team} + On-Call Architecture"
        return base_team
    
    def _get_business_owner(self, incident_type: str) -> str:
        """Get business owner for incident type"""
        owners = {
            "business_critical": "Director of E-commerce Operations",
            "payment_critical": "VP of Financial Operations",
            "performance_critical": "VP of Engineering",
            "trading_critical": "Head of Trading Operations",
            "security_business": "Chief Security Officer"
        }
        return owners.get(incident_type, "VP of Operations")
    
    async def _execute_business_ticketing_agent(self, incident: Incident) -> AgentExecution:
        """Business-focused Enhanced Ticketing Agent with business impact classification"""
        execution = AgentExecution(
            agent_id="ticketing", agent_name="Business Impact Ticketing Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            await self._detailed_log(execution, f"🎫 Business-focused ticket creation and classification", "INFO", {
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact,
                "affected_systems": len(incident.affected_systems)
            })
            execution.progress = 25
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Enhanced business classification
            business_priority = self._get_business_priority(incident)
            sla_requirements = self._get_business_sla_requirements(incident)
            stakeholder_groups = self._get_stakeholder_groups(incident.incident_type)
            
            await self._detailed_log(execution, f"📊 Business impact classification completed", "CLASSIFICATION", {
                "business_priority": business_priority,
                "sla_target": sla_requirements["target"],
                "escalation_required": sla_requirements["escalation_required"],
                "stakeholder_groups": len(stakeholder_groups)
            })
            execution.progress = 60
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Business context ticket creation
            ticket_details = self._create_business_ticket_details(incident)
            
            await self._detailed_log(execution, f"📝 Business context ticket creation", "TICKET_CREATION", {
                "ticket_category": ticket_details["category"],
                "business_subcategory": ticket_details["subcategory"],
                "financial_impact": ticket_details["financial_impact"],
                "customer_impact": ticket_details["customer_impact"]
            })
            execution.progress = 85
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            execution.output_data = {
                "ticket_id": f"BIZ-{incident.incident_type.upper()}{datetime.now().strftime('%Y%m%d')}{incident.id[-4:]}",
                "business_classification": {
                    "priority": business_priority,
                    "category": ticket_details["category"],
                    "subcategory": ticket_details["subcategory"],
                    "sla_requirements": sla_requirements
                },
                "business_context": {
                    "financial_impact": ticket_details["financial_impact"],
                    "customer_impact": ticket_details["customer_impact"],
                    "stakeholder_groups": stakeholder_groups,
                    "business_owner": self._get_business_owner(incident.incident_type)
                },
                "assigned_team": self._get_business_technical_team(incident.incident_type, incident.severity.value),
                "mcp_enhanced_classification": True,
                "business_impact_score": self._calculate_business_impact_score(incident)
            }
            
            incident.servicenow_ticket_id = execution.output_data["ticket_id"]
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ Business-focused ticket creation completed", "SUCCESS", {
                "ticket_id": execution.output_data["ticket_id"],
                "business_priority": business_priority,
                "sla_target": sla_requirements["target"],
                "business_impact_score": execution.output_data["business_impact_score"],
                "total_log_entries": len(execution.logs)
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Business ticketing failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    def _get_business_priority(self, incident: Incident) -> str:
        """Get business-focused priority classification"""
        if incident.severity == IncidentSeverity.CRITICAL:
            if "revenue" in incident.business_impact.lower() or "payment" in incident.incident_type:
                return "P0 - Business Critical"
            else:
                return "P1 - Critical"
        elif incident.severity == IncidentSeverity.HIGH:
            return "P2 - High Business Impact"
        else:
            return "P3 - Medium Business Impact"
    
    def _get_business_sla_requirements(self, incident: Incident) -> Dict[str, Any]:
        """Get business SLA requirements"""
        sla_map = {
            "business_critical": {"target": "15 minutes", "escalation_required": True, "executive_notification": True},
            "payment_critical": {"target": "10 minutes", "escalation_required": True, "executive_notification": True},
            "performance_critical": {"target": "30 minutes", "escalation_required": True, "executive_notification": False},
            "trading_critical": {"target": "5 minutes", "escalation_required": True, "executive_notification": True}
        }
        
        return sla_map.get(incident.incident_type, {
            "target": "60 minutes",
            "escalation_required": False,
            "executive_notification": False
        })
    
    def _get_stakeholder_groups(self, incident_type: str) -> List[str]:
        """Get stakeholder groups for incident type"""
        groups = {
            "business_critical": ["Operations", "Customer Success", "Finance", "Executive"],
            "payment_critical": ["Finance", "Compliance", "Customer Support", "Executive"],
            "performance_critical": ["Engineering", "Product", "Customer Experience"],
            "trading_critical": ["Trading", "Risk Management", "Compliance", "Executive"]
        }
        return groups.get(incident_type, ["Operations", "Engineering"])
    
    def _create_business_ticket_details(self, incident: Incident) -> Dict[str, str]:
        """Create business-focused ticket details"""
        return {
            "category": f"Business Impact - {incident.incident_type.replace('_', ' ').title()}",
            "subcategory": self._get_business_subcategory(incident.incident_type),
            "financial_impact": self._extract_financial_impact(incident.business_impact),
            "customer_impact": self._extract_customer_impact(incident.business_impact)
        }
    
    def _get_business_subcategory(self, incident_type: str) -> str:
        """Get business subcategory"""
        subcategories = {
            "business_critical": "Revenue Operations",
            "payment_critical": "Payment Processing",
            "performance_critical": "Customer Experience",
            "trading_critical": "Market Operations",
            "security_business": "Security & Compliance"
        }
        return subcategories.get(incident_type, "General Operations")
    
    def _extract_financial_impact(self, business_impact: str) -> str:
        """Extract financial impact from business impact description"""
        if "$" in business_impact:
            # Extract dollar amounts from the description
            import re
            amounts = re.findall(r'\$[\d,]+', business_impact)
            if amounts:
                return f"Financial loss: {amounts[0]} per unit time"
        return "Moderate financial impact"
    
    def _extract_customer_impact(self, business_impact: str) -> str:
        """Extract customer impact from business impact description"""
        if "customer" in business_impact.lower():
            if "satisfaction" in business_impact.lower():
                return "Customer satisfaction degradation"
            elif "churn" in business_impact.lower():
                return "Customer churn risk"
            else:
                return "Customer experience impact"
        return "Operational impact"
    
    def _calculate_business_impact_score(self, incident: Incident) -> int:
        """Calculate business impact score (0-100)"""
        base_score = {
            "critical": 90,
            "high": 70,
            "medium": 50,
            "low": 30
        }.get(incident.severity.value, 50)
        
        # Adjust based on incident type
        type_multiplier = {
            "business_critical": 1.0,
            "payment_critical": 0.95,
            "trading_critical": 0.98,
            "performance_critical": 0.85
        }.get(incident.incident_type, 0.8)
        
        return min(100, int(base_score * type_multiplier))
    
    async def _execute_business_email_agent(self, incident: Incident) -> AgentExecution:
        """Business-focused Enhanced Email Agent with stakeholder-specific communication"""
        execution = AgentExecution(
            agent_id="email", agent_name="Business Stakeholder Communication Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            await self._detailed_log(execution, f"📧 Business stakeholder communication strategy development", "INFO", {
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact,
                "severity": incident.severity.value
            })
            execution.progress = 20
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Business stakeholder communication planning
            communication_strategy = self._develop_communication_strategy(incident)
            stakeholder_segments = self._segment_stakeholders(incident)
            
            await self._detailed_log(execution, f"👥 Stakeholder segmentation and communication planning", "COMMUNICATION_PLANNING", {
                "stakeholder_segments": len(stakeholder_segments),
                "communication_channels": communication_strategy["channels"],
                "message_personalization": communication_strategy["personalized"],
                "executive_briefing": communication_strategy["executive_briefing"]
            })
            execution.progress = 50
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # Personalized message creation for each stakeholder group
            personalized_messages = self._create_personalized_messages(incident, stakeholder_segments)
            
            await self._detailed_log(execution, f"✍️ Personalized message creation for stakeholder groups", "MESSAGE_CREATION", {
                "message_variants": len(personalized_messages),
                "executive_summary": "executive" in personalized_messages,
                "technical_details": "technical" in personalized_messages,
                "customer_impact_focus": "customer_facing" in personalized_messages
            })
            execution.progress = 75
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Business impact communication
            business_metrics = self._prepare_business_metrics_communication(incident)
            
            await self._detailed_log(execution, f"📊 Business impact metrics communication preparation", "BUSINESS_METRICS", {
                "financial_impact_included": business_metrics["financial_impact_included"],
                "customer_metrics": business_metrics["customer_metrics"],
                "operational_metrics": business_metrics["operational_metrics"],
                "recovery_timeline": business_metrics["recovery_timeline"]
            })
            
            execution.output_data = {
                "communication_execution": {
                    "stakeholder_segments": stakeholder_segments,
                    "personalized_messages": personalized_messages,
                    "business_metrics": business_metrics
                },
                "communication_strategy": communication_strategy,
                "notification_types": {
                    "executive_summary": communication_strategy["executive_briefing"],
                    "technical_details": True,
                    "business_impact_analysis": True,
                    "customer_communication": self._requires_customer_communication(incident),
                    "regulatory_notification": self._requires_regulatory_notification(incident)
                },
                "a2a_coordinated": True,
                "mcp_context_used": True,
                "business_focused": True
            }
            
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ Business stakeholder communication completed successfully", "SUCCESS", {
                "stakeholder_groups_notified": len(stakeholder_segments),
                "message_variants": len(personalized_messages),
                "business_metrics_shared": True,
                "executive_briefing": communication_strategy["executive_briefing"],
                "total_log_entries": len(execution.logs)
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Business stakeholder communication failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    def _develop_communication_strategy(self, incident: Incident) -> Dict[str, Any]:
        """Develop business communication strategy"""
        if incident.severity == IncidentSeverity.CRITICAL:
            return {
                "channels": ["email", "slack", "sms", "phone"],
                "frequency": "every_15_minutes",
                "personalized": True,
                "executive_briefing": True,
                "customer_communication": True
            }
        else:
            return {
                "channels": ["email", "slack"],
                "frequency": "every_30_minutes",
                "personalized": True,
                "executive_briefing": False,
                "customer_communication": False
            }
    
    def _segment_stakeholders(self, incident: Incident) -> Dict[str, List[str]]:
        """Segment stakeholders based on business context"""
        segments = {
            "executive": ["CEO", "COO", "CTO"],
            "business_owners": [self._get_business_owner(incident.incident_type)],
            "technical": [self._get_business_technical_team(incident.incident_type, incident.severity.value)],
            "customer_facing": ["Customer Success", "Support", "Account Management"],
            "compliance": ["Legal", "Compliance Officer", "Risk Management"]
        }
        
        # Filter segments based on incident type
        if incident.incident_type == "payment_critical":
            segments["financial"] = ["CFO", "Finance Director", "Payment Operations"]
        elif incident.incident_type == "trading_critical":
            segments["trading"] = ["Head of Trading", "Risk Manager", "Trading Operations"]
        
        return segments
    
    def _create_personalized_messages(self, incident: Incident, stakeholder_segments: Dict[str, List[str]]) -> Dict[str, str]:
        """Create personalized messages for each stakeholder segment"""
        messages = {}
        
        if "executive" in stakeholder_segments:
            messages["executive"] = f"Executive Brief: {incident.incident_type} incident with {incident.business_impact}. Technical teams engaged, resolution in progress."
        
        if "business_owners" in stakeholder_segments:
            messages["business_owners"] = f"Business Impact Alert: {incident.title}. Direct impact on operations requires immediate attention and coordination."
        
        if "technical" in stakeholder_segments:
            messages["technical"] = f"Technical Incident: {incident.description}. Root cause analysis in progress, remediation actions being implemented."
        
        if "customer_facing" in stakeholder_segments:
            messages["customer_facing"] = f"Customer Impact: Service degradation affecting customer experience. Prepare for increased support volume and customer communications."
        
        return messages
    
    def _prepare_business_metrics_communication(self, incident: Incident) -> Dict[str, Any]:
        """Prepare business metrics for communication"""
        return {
            "financial_impact_included": "$" in incident.business_impact,
            "customer_metrics": self._extract_customer_metrics(incident),
            "operational_metrics": self._extract_operational_metrics(incident),
            "recovery_timeline": self._get_recovery_timeline(incident.incident_type),
            "sla_status": self._get_sla_status(incident)
        }
    
    def _extract_customer_metrics(self, incident: Incident) -> Dict[str, Any]:
        """Extract customer-related metrics"""
        return {
            "affected_customers": "estimated_high" if incident.severity == IncidentSeverity.CRITICAL else "estimated_medium",
            "experience_impact": "severe" if "critical" in incident.incident_type else "moderate",
            "support_volume_increase": "expected"
        }
    
    def _extract_operational_metrics(self, incident: Incident) -> Dict[str, Any]:
        """Extract operational metrics"""
        return {
            "systems_affected": len(incident.affected_systems),
            "business_process_impact": "high" if incident.severity == IncidentSeverity.CRITICAL else "medium",
            "operational_capacity": "reduced" if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH] else "stable"
        }
    
    def _get_sla_status(self, incident: Incident) -> str:
        """Get SLA status for the incident"""
        elapsed_time = (datetime.now() - incident.created_at).total_seconds() / 60  # minutes
        sla_target = 15 if incident.severity == IncidentSeverity.CRITICAL else 30
        
        if elapsed_time < sla_target * 0.5:
            return "within_sla"
        elif elapsed_time < sla_target:
            return "approaching_sla_breach"
        else:
            return "sla_breached"
    
    def _requires_customer_communication(self, incident: Incident) -> bool:
        """Check if customer communication is required"""
        return incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH] and \
               "customer" in incident.business_impact.lower()
    
    def _requires_regulatory_notification(self, incident: Incident) -> bool:
        """Check if regulatory notification is required"""
        return incident.incident_type in ["payment_critical", "trading_critical", "security_business"] and \
               incident.severity == IncidentSeverity.CRITICAL
    
    async def _execute_business_remediation_agent(self, incident: Incident) -> AgentExecution:
        """Business-focused Enhanced Remediation Agent with business continuity focus"""
        execution = AgentExecution(
            agent_id="remediation", agent_name="Business Continuity Remediation Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Get comprehensive RCA insights from MCP context and A2A messages
            mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
            rca_insights = {}
            if mcp_context and "rca" in mcp_context.agent_insights:
                rca_insights = mcp_context.agent_insights["rca"]["data"]
            
            await self._detailed_log(execution, f"🔧 Business continuity focused remediation planning", "INFO", {
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact,
                "rca_insights_available": bool(rca_insights),
                "priority": "business_continuity_first"
            })
            execution.progress = 15
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # Business continuity assessment
            continuity_assessment = self._assess_business_continuity_impact(incident)
            
            await self._detailed_log(execution, f"🏢 Business continuity impact assessment", "CONTINUITY_ASSESSMENT", {
                "business_functions_affected": continuity_assessment["functions_affected"],
                "revenue_impact_per_minute": continuity_assessment["revenue_impact"],
                "customer_experience_degradation": continuity_assessment["customer_impact"],
                "operational_capacity": continuity_assessment["operational_capacity"]
            })
            execution.progress = 35
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Enhanced remediation actions with business priority
            business_priority_actions = self._get_business_priority_remediation_actions(incident, rca_insights)
            technical_actions = self._get_technical_remediation_actions(incident.incident_type)
            
            await self._detailed_log(execution, f"⚡ Business-priority remediation strategy development", "REMEDIATION_PLANNING", {
                "business_priority_actions": len(business_priority_actions),
                "technical_actions": len(technical_actions),
                "rca_enhanced": bool(rca_insights),
                "business_continuity_focus": True
            })
            execution.progress = 55
            await asyncio.sleep(random.uniform(2.0, 2.5))
            
            # Execute business continuity procedures
            continuity_procedures = self._execute_business_continuity_procedures(incident)
            
            await self._detailed_log(execution, f"🚀 Business continuity procedures execution", "CONTINUITY_EXECUTION", {
                "immediate_actions": continuity_procedures["immediate"],
                "workaround_solutions": continuity_procedures["workarounds"],
                "customer_impact_mitigation": continuity_procedures["customer_mitigation"],
                "revenue_protection": continuity_procedures["revenue_protection"]
            })
            execution.progress = 80
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # A2A coordination with validation agent for business verification
            validation_request = A2AMessage(
                sender_agent_id="remediation",
                receiver_agent_id="validation",
                message_type="collaboration_request",
                content={
                    "task": "business_continuity_validation",
                    "business_actions_applied": business_priority_actions,
                    "continuity_procedures": continuity_procedures,
                    "business_metrics_to_verify": [
                        "revenue_flow_restoration",
                        "customer_experience_recovery",
                        "operational_capacity_verification"
                    ],
                    "incident_context": {
                        "type": incident.incident_type,
                        "business_impact": incident.business_impact,
                        "continuity_assessment": continuity_assessment
                    }
                },
                priority="high"
            )
            self.a2a_protocol.send_message(validation_request)
            execution.a2a_messages_sent += 1
            
            await self._detailed_log(execution, f"📨 Business continuity validation request sent to validation agent", "A2A_COLLABORATION", {
                "validation_focus": "business_continuity",
                "metrics_to_verify": 3,
                "collaboration_priority": "high"
            })
            
            execution.output_data = {
                "business_remediation": {
                    "continuity_assessment": continuity_assessment,
                    "business_priority_actions": business_priority_actions,
                    "continuity_procedures": continuity_procedures,
                    "revenue_protection_measures": self._get_revenue_protection_measures(incident)
                },
                "technical_remediation": {
                    "technical_actions": technical_actions,
                    "system_recovery_steps": self._get_system_recovery_steps(incident.incident_type),
                    "monitoring_enhancements": self._get_monitoring_enhancements(incident.incident_type)
                },
                "remediation_strategy": f"business_continuity_focused_{incident.incident_type}",
                "automation_level": self._get_business_automation_level(incident.incident_type),
                "rca_enhanced": bool(rca_insights),
                "validation_requested": True,
                "business_intelligence_confidence": rca_insights.get("confidence", 0.8) if rca_insights else 0.8
            }
            
            # Update MCP context with remediation results
            if mcp_context:
                mcp_context.update_context("remediation", execution.output_data, 0.91)
                await self._detailed_log(execution, f"🧠 MCP Context updated with business remediation results", "MCP_UPDATE", {
                    "confidence_score": 0.91,
                    "business_focus": True,
                    "continuity_data": True
                })
            
            incident.remediation_applied = business_priority_actions + technical_actions
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            await self._detailed_log(execution, f"✅ Business continuity remediation completed successfully", "SUCCESS", {
                "total_actions_applied": len(business_priority_actions) + len(technical_actions),
                "business_continuity_procedures": len(continuity_procedures),
                "a2a_collaboration_initiated": True,
                "revenue_protection_active": True,
                "total_log_entries": len(execution.logs)
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Business continuity remediation failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    def _assess_business_continuity_impact(self, incident: Incident) -> Dict[str, Any]:
        """Assess business continuity impact"""
        impact_mapping = {
            "business_critical": {
                "functions_affected": ["order_processing", "payment_flow", "customer_service"],
                "revenue_impact": "$2,500/minute",
                "customer_impact": "severe_degradation",
                "operational_capacity": "25%_reduction"
            },
            "payment_critical": {
                "functions_affected": ["payment_processing", "transaction_settlement", "merchant_services"],
                "revenue_impact": "$45,000/hour",
                "customer_impact": "payment_frustration",
                "operational_capacity": "payment_halt"
            },
            "performance_critical": {
                "functions_affected": ["customer_experience", "search_functionality", "conversion_flow"],
                "revenue_impact": "$1,200/minute",
                "customer_impact": "experience_degradation",
                "operational_capacity": "50%_efficiency"
            }
        }
        
        return impact_mapping.get(incident.incident_type, {
            "functions_affected": ["core_operations"],
            "revenue_impact": "$500/minute",
            "customer_impact": "moderate",
            "operational_capacity": "reduced"
        })
    
    def _get_business_priority_remediation_actions(self, incident: Incident, rca_insights: Dict[str, Any]) -> List[str]:
        """Get business-priority remediation actions"""
        base_actions = {
            "business_critical": [
                "activate_backup_payment_processing",
                "implement_order_queue_priority_system", 
                "enable_customer_communication_automation",
                "activate_revenue_protection_protocols"
            ],
            "payment_critical": [
                "failover_to_secondary_payment_provider",
                "activate_transaction_retry_mechanisms",
                "implement_payment_method_diversification",
                "enable_merchant_notification_system"
            ],
            "performance_critical": [
                "activate_search_result_caching",
                "implement_graceful_performance_degradation",
                "enable_cdn_optimization_boost",
                "activate_customer_experience_preservation"
            ]
        }
        
        actions = base_actions.get(incident.incident_type, [
            "restore_core_business_functions",
            "minimize_customer_impact",
            "protect_revenue_streams"
        ])
        
        # Enhance actions based on RCA insights
        if rca_insights and rca_insights.get("business_priority_actions"):
            actions.extend(rca_insights["business_priority_actions"][:2])
        
        return actions
    
    def _get_technical_remediation_actions(self, incident_type: str) -> List[str]:
        """Get technical remediation actions"""
        technical_actions = {
            "business_critical": [
                "database_connection_pool_optimization",
                "payment_service_scaling",
                "order_processing_queue_tuning"
            ],
            "payment_critical": [
                "ssl_certificate_renewal_automation",
                "webhook_endpoint_failover",
                "payment_provider_connection_restore"
            ],
            "performance_critical": [
                "elasticsearch_cluster_recovery",
                "index_corruption_repair",
                "search_service_memory_optimization"
            ]
        }
        
        return technical_actions.get(incident_type, [
            "service_restart_procedures",
            "configuration_optimization",
            "monitoring_enhancement"
        ])
    
    def _execute_business_continuity_procedures(self, incident: Incident) -> Dict[str, List[str]]:
        """Execute business continuity procedures"""
        return {
            "immediate": [
                f"activate_{incident.incident_type}_business_continuity_plan",
                "notify_key_business_stakeholders",
                "implement_customer_impact_mitigation"
            ],
            "workarounds": [
                f"enable_{incident.incident_type}_workaround_solutions",
                "activate_backup_business_processes",
                "implement_manual_override_procedures"
            ],
            "customer_mitigation": [
                "customer_communication_automation",
                "support_team_capacity_scaling",
                "customer_experience_preservation"
            ],
            "revenue_protection": [
                "revenue_stream_diversification",
                "payment_method_redundancy",
                "transaction_prioritization"
            ]
        }
    
    def _get_revenue_protection_measures(self, incident: Incident) -> List[str]:
        """Get revenue protection measures"""
        return [
            f"{incident.incident_type}_revenue_safeguarding",
            "payment_flow_continuity",
            "customer_retention_protocols",
            "business_impact_minimization"
        ]
    
    def _get_system_recovery_steps(self, incident_type: str) -> List[str]:
        """Get system recovery steps"""
        recovery_steps = {
            "business_critical": [
                "payment_gateway_restoration",
                "order_management_recovery",
                "inventory_system_synchronization"
            ],
            "payment_critical": [
                "payment_provider_reconnection",
                "transaction_processing_restoration",
                "fraud_detection_reactivation"
            ],
            "performance_critical": [
                "search_index_rebuilding",
                "cache_warming_procedures",
                "performance_optimization_application"
            ]
        }
        
        return recovery_steps.get(incident_type, [
            "service_health_restoration",
            "data_consistency_verification",
            "performance_baseline_recovery"
        ])
    
    def _get_monitoring_enhancements(self, incident_type: str) -> List[str]:
        """Get monitoring enhancements"""
        return [
            f"{incident_type}_specific_monitoring",
            "business_metrics_alerting",
            "customer_impact_tracking",
            "revenue_flow_monitoring"
        ]
    
    def _get_business_automation_level(self, incident_type: str) -> str:
        """Get business automation level"""
        levels = {
            "business_critical": "high_with_business_validation",
            "payment_critical": "medium_with_financial_oversight",
            "performance_critical": "high_automated",
            "trading_critical": "low_manual_oversight_required"
        }
        return levels.get(incident_type, "medium_automated")
    
    async def _execute_business_validation_agent(self, incident: Incident) -> AgentExecution:
        """Business-focused Enhanced Validation Agent with comprehensive business verification"""
        execution = AgentExecution(
            agent_id="validation", agent_name="Business Continuity Validation Agent",
            incident_id=incident.id, mcp_context_id=incident.mcp_context_id
        )
        
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Get comprehensive context from all agents via MCP
            mcp_context = self.mcp_registry.get_context(incident.mcp_context_id)
            full_context = {}
            confidence_factors = []
            
            if mcp_context:
                full_context = mcp_context.get_contextual_insights("validation")
                execution.contextual_insights_used = full_context
                confidence_factors = list(mcp_context.confidence_scores.values())
            
            overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.8
            
            await self._detailed_log(execution, f"🔍 Comprehensive business continuity validation initiated", "INFO", {
                "incident_type": incident.incident_type,
                "business_impact": incident.business_impact,
                "agent_insights_count": len(confidence_factors),
                "overall_system_confidence": f"{overall_confidence:.2%}"
            })
            execution.progress = 20
            await asyncio.sleep(random.uniform(2.0, 2.5))
            
            # Business metrics validation
            business_validation = self._validate_business_metrics(incident)
            
            await self._detailed_log(execution, f"💼 Business metrics validation completed", "BUSINESS_VALIDATION", {
                "revenue_flow_status": business_validation["revenue_flow"],
                "customer_experience_score": business_validation["customer_experience"],
                "operational_capacity": business_validation["operational_capacity"],
                "business_continuity_score": business_validation["continuity_score"]
            })
            execution.progress = 45
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # Technical system validation
            technical_validation = self._validate_technical_systems(incident)
            
            await self._detailed_log(execution, f"🔧 Technical system validation completed", "TECHNICAL_VALIDATION", {
                "system_health_status": technical_validation["health_status"],
                "performance_metrics": technical_validation["performance"],
                "integration_status": technical_validation["integrations"],
                "monitoring_status": technical_validation["monitoring"]
            })
            execution.progress = 70
            await asyncio.sleep(random.uniform(1.5, 2.0))
            
            # Customer impact validation
            customer_validation = self._validate_customer_impact(incident)
            
            await self._detailed_log(execution, f"👥 Customer impact validation completed", "CUSTOMER_VALIDATION", {
                "customer_experience_restored": customer_validation["experience_restored"],
                "support_volume_status": customer_validation["support_volume"],
                "satisfaction_metrics": customer_validation["satisfaction"],
                "churn_risk_assessment": customer_validation["churn_risk"]
            })
            execution.progress = 90
            await asyncio.sleep(random.uniform(1.0, 1.5))
            
            # Enhanced success rate based on business criticality and system confidence
            base_success_rate = 0.80
            business_boost = 0.15 if overall_confidence > 0.85 else 0.10
            severity_factor = {
                "critical": -0.05, 
                "high": 0.0, 
                "medium": 0.05, 
                "low": 0.10
            }.get(incident.severity.value, 0.0)
            
            final_success_rate = base_success_rate + business_boost + severity_factor
            resolution_successful = random.random() < final_success_rate
            
            # Comprehensive validation results
            validation_results = {
                "business_validation": business_validation,
                "technical_validation": technical_validation,
                "customer_validation": customer_validation,
                "overall_health_score": self._calculate_overall_health_score(
                    business_validation, technical_validation, customer_validation
                )
            }
            
            execution.output_data = {
                "business_continuity_validation": {
                    "resolution_successful": resolution_successful,
                    "validation_results": validation_results,
                    "business_metrics_restored": resolution_successful,
                    "customer_impact_mitigated": resolution_successful,
                    "revenue_protection_effective": resolution_successful
                },
                "comprehensive_analysis": {
                    "mcp_enhanced": True,
                    "cross_agent_validation": True,
                    "confidence_factors_used": len(confidence_factors),
                    "overall_system_confidence": overall_confidence,
                    "validation_depth": "comprehensive_business_focused"
                },
                "validation_score": random.uniform(0.94, 0.99) if resolution_successful else random.uniform(0.75, 0.89),
                "business_impact_resolved": resolution_successful,
                "post_incident_actions": self._get_business_post_incident_actions(incident.incident_type)
            }
            
            # Final comprehensive MCP context update
            if mcp_context:
                mcp_context.update_context("validation", execution.output_data, 0.97)
                mcp_context.shared_knowledge["final_business_resolution"] = {
                    "status": "business_resolved" if resolution_successful else "business_partially_resolved",
                    "overall_confidence": overall_confidence,
                    "validation_score": execution.output_data["validation_score"],
                    "business_continuity_restored": resolution_successful,
                    "customer_impact_mitigated": resolution_successful,
                    "validated_at": datetime.now().isoformat()
                }
                
                await self._detailed_log(execution, f"🧠 Final MCP context update with comprehensive business validation", "MCP_FINAL_UPDATE", {
                    "confidence_score": 0.97,
                    "business_resolution_status": "resolved" if resolution_successful else "partially_resolved",
                    "final_validation": True
                })
            
            # Set incident resolution with business context
            if resolution_successful:
                incident.resolution = f"Business incident {incident.incident_type} fully resolved using comprehensive MCP+A2A enhanced analysis. Business continuity restored with {overall_confidence:.1%} system confidence. Revenue protection effective, customer impact mitigated. Validation score: {execution.output_data['validation_score']:.1%}."
                incident.status = "resolved"
            else:
                incident.resolution = f"Business incident {incident.incident_type} partially resolved - MCP enhanced analysis indicates continued business monitoring required. Revenue protection active, customer impact being managed. Validation score: {execution.output_data['validation_score']:.1%}. Business continuity procedures remain active."
                incident.status = "partially_resolved"
            
            execution.progress = 100
            execution.status = AgentStatus.SUCCESS
            status_msg = "fully resolved with business continuity restored" if resolution_successful else "partially resolved with business continuity active"
            await self._detailed_log(execution, f"✅ Comprehensive business validation completed - Issue {status_msg}", "SUCCESS", {
                "resolution_status": "resolved" if resolution_successful else "partially_resolved",
                "business_continuity": "restored" if resolution_successful else "active",
                "validation_confidence": f"{execution.output_data['validation_score']:.1%}",
                "customer_impact": "mitigated" if resolution_successful else "managed",
                "revenue_protection": "effective",
                "total_log_entries": len(execution.logs)
            })
            
        except Exception as e:
            execution.status = AgentStatus.ERROR
            execution.error_message = str(e)
            await self._detailed_log(execution, f"❌ Business validation failed: {str(e)}", "ERROR")
        
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        return execution
    
    def _validate_business_metrics(self, incident: Incident) -> Dict[str, Any]:
        """Validate business metrics recovery"""
        if incident.incident_type == "business_critical":
            return {
                "revenue_flow": "restored" if random.random() < 0.85 else "partially_restored",
                "customer_experience": "baseline_achieved" if random.random() < 0.80 else "improving",
                "operational_capacity": "90%" if random.random() < 0.85 else "75%",
                "continuity_score": random.uniform(0.85, 0.95) if random.random() < 0.85 else random.uniform(0.70, 0.85)
            }
        elif incident.incident_type == "payment_critical":
            return {
                "revenue_flow": "payment_processing_restored" if random.random() < 0.90 else "backup_provider_active",
                "customer_experience": "payment_flow_normal" if random.random() < 0.85 else "some_delays",
                "operational_capacity": "95%" if random.random() < 0.90 else "80%",
                "continuity_score": random.uniform(0.88, 0.96) if random.random() < 0.90 else random.uniform(0.75, 0.88)
            }
        else:
            return {
                "revenue_flow": "stable" if random.random() < 0.80 else "recovering",
                "customer_experience": "acceptable" if random.random() < 0.75 else "degraded",
                "operational_capacity": "85%" if random.random() < 0.80 else "70%",
                "continuity_score": random.uniform(0.80, 0.90) if random.random() < 0.80 else random.uniform(0.65, 0.80)
            }
    
    def _validate_technical_systems(self, incident: Incident) -> Dict[str, Any]:
        """Validate technical system recovery"""
        success_rate = 0.85 if incident.severity == IncidentSeverity.CRITICAL else 0.90
        
        return {
            "health_status": "healthy" if random.random() < success_rate else "recovering",
            "performance": "optimal" if random.random() < success_rate else "acceptable",
            "integrations": "all_connected" if random.random() < success_rate else "mostly_connected",
            "monitoring": "enhanced_active" if random.random() < 0.95 else "standard_active"
        }
    
    def _validate_customer_impact(self, incident: Incident) -> Dict[str, Any]:
        """Validate customer impact mitigation"""
        return {
            "experience_restored": random.random() < 0.85,
            "support_volume": "normalized" if random.random() < 0.80 else "elevated",
            "satisfaction": "recovering" if random.random() < 0.75 else "monitoring_required",
            "churn_risk": "mitigated" if random.random() < 0.80 else "monitoring"
        }
    
    def _calculate_overall_health_score(self, business_val: Dict, technical_val: Dict, customer_val: Dict) -> float:
        """Calculate overall health score"""
        business_score = business_val.get("continuity_score", 0.8)
        
        technical_score = 0.9 if technical_val["health_status"] == "healthy" else 0.7
        customer_score = 0.85 if customer_val["experience_restored"] else 0.65
        
        return (business_score * 0.4 + technical_score * 0.35 + customer_score * 0.25)
    
    def _get_business_post_incident_actions(self, incident_type: str) -> List[str]:
        """Get business-focused post-incident actions"""
        actions = {
            "business_critical": [
                "business_continuity_plan_review",
                "revenue_protection_optimization",
                "customer_experience_improvement_analysis",
                "payment_processing_resilience_enhancement"
            ],
            "payment_critical": [
                "payment_provider_diversification_review",
                "financial_resilience_assessment",
                "merchant_relationship_strengthening",
                "payment_security_enhancement"
            ],
            "performance_critical": [
                "customer_experience_optimization",
                "search_performance_improvement",
                "conversion_funnel_analysis",
                "user_journey_optimization"
            ]
        }
        
        return actions.get(incident_type, [
            "business_impact_assessment",
            "operational_resilience_review",
            "customer_satisfaction_improvement",
            "revenue_protection_enhancement"
        ])
    
    # ENHANCED LOGGING UTILITY
    async def _detailed_log(self, execution: AgentExecution, message: str, log_type: str = "INFO", additional_data: Dict[str, Any] = None):
        """Enhanced detailed logging with business context and structured data"""
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
        
        # Enhanced console logging with emoji and structure
        log_prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "ANALYSIS": "🔍",
            "BUSINESS_ANALYSIS": "💼",
            "TECHNICAL": "🔧",
            "COLLABORATION": "🤝",
            "A2A_COMMUNICATION": "📨",
            "A2A_SHARE": "📤",
            "A2A_COORDINATION": "🔗",
            "MCP_ANALYSIS": "🧠",
            "MCP_UPDATE": "🧠📝",
            "MCP_FINAL_UPDATE": "🧠✨",
            "STAKEHOLDER_ANALYSIS": "👥",
            "TEAM_ASSIGNMENT": "🎯",
            "CLASSIFICATION": "📊",
            "TICKET_CREATION": "🎫",
            "COMMUNICATION_PLANNING": "📋",
            "MESSAGE_CREATION": "✍️",
            "BUSINESS_METRICS": "📈",
            "CONTINUITY_ASSESSMENT": "🏢",
            "REMEDIATION_PLANNING": "⚡",
            "CONTINUITY_EXECUTION": "🚀",
            "BUSINESS_VALIDATION": "💼✅",
            "TECHNICAL_VALIDATION": "🔧✅",
            "CUSTOMER_VALIDATION": "👥✅",
            "FINANCIAL_ANALYSIS": "💰",
            "CUSTOMER_ANALYSIS": "👥📊",
            "ROOT_CAUSE": "🎯",
            "PEER_ANALYSIS": "🤝🔍"
        }.get(log_type, "📝")
        
        formatted_message = f"{log_prefix} [{execution.agent_id.upper()}] {message}"
        if additional_data:
            formatted_message += f" | Data: {json.dumps(additional_data, default=str)}"
        
        logger.info(f"[{execution.incident_id}] {formatted_message}")

# Global complete enhanced workflow engine
workflow_engine = CompleteEnhancedWorkflowEngine()

# =============================================================================
# COMPLETE ENHANCED FASTAPI APPLICATION WITH AGENT LOG ENDPOINTS
# =============================================================================

class CompleteEnhancedMonitoringApp:
    def __init__(self):
        self.app = FastAPI(
            title="Complete MCP + A2A Enhanced AI Monitoring System with Business Intelligence",
            description="ALL Previous Features + Model Context Protocol + Agent-to-Agent Communication + Business-Centric Incidents + Detailed Agent Logging",
            version="4.0.0",
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
        # Enhanced incident triggering with business scenarios
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
                "message": f"Business-critical enhanced incident {incident.id} workflow initiated with comprehensive MCP+A2A analysis",
                "affected_systems": len(incident.affected_systems),
                "enhanced_features": [
                    "Business-Centric Incident Scenarios", 
                    "Model Context Protocol", 
                    "Agent-to-Agent Communication", 
                    "Detailed Agent Console Logs",
                    "Real-time WebSocket Updates",
                    "All 7 Specialized Agents",
                    "Comprehensive Business Analysis"
                ]
            }
        
        # ENHANCED AGENT LOG ENDPOINTS - THE KEY FEATURE REQUESTED
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
                    "collaboration_sessions": execution.collaboration_sessions,
                    "total_collaborations": len(execution.collaboration_sessions)
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
                    "business_focused_logs": sum(1 for log in execution.logs if log.get("business_context")),
                    "mcp_related_logs": sum(1 for log in execution.logs if "MCP" in log.get("log_type", "")),
                    "a2a_related_logs": sum(1 for log in execution.logs if "A2A" in log.get("log_type", ""))
                }
            }
        
        # Get agent execution history with detailed logging
        @self.app.get("/api/agents/{agent_id}/history")
        async def get_agent_execution_history_with_logs(agent_id: str, limit: int = 20):
            if agent_id not in workflow_engine.agent_execution_history:
                return {"error": "Agent not found"}
            
            executions = workflow_engine.agent_execution_history[agent_id][-limit:]
            
            return {
                "agent_id": agent_id,
                "agent_name": f"Business Enhanced {agent_id.title()} Agent",
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
                        "business_enhanced": True,
                        "mcp_enhanced": bool(exec.contextual_insights_used),
                        "a2a_messages": exec.a2a_messages_sent + exec.a2a_messages_received,
                        "collaborations": len(exec.collaboration_sessions),
                        "log_types": list(set(log.get("log_type", "INFO") for log in exec.logs)),
                        "business_context_logs": sum(1 for log in exec.logs if log.get("business_context"))
                    }
                    for exec in executions
                ],
                "agent_capabilities": workflow_engine.a2a_protocol.agent_capabilities.get(agent_id, []),
                "business_focus": True
            }
        
        # Complete enhanced incident status with business context
        @self.app.get("/api/incidents/{incident_id}/status")
        async def get_complete_business_incident_status(incident_id: str):
            incident = None
            if incident_id in workflow_engine.active_incidents:
                incident = workflow_engine.active_incidents[incident_id]
            else:
                incident = next((i for i in workflow_engine.incident_history if i.id == incident_id), None)
            
            if not incident:
                return {"error": "Business incident not found"}
            
            # Get comprehensive MCP context data
            mcp_data = {}
            if incident.mcp_context_id:
                context = workflow_engine.mcp_registry.get_context(incident.mcp_context_id)
                if context:
                    mcp_data = {
                        "context_id": context.context_id,
                        "context_version": context.context_version,
                        "agent_insights_count": len(context.agent_insights),
                        "avg_confidence": sum(context.confidence_scores.values()) / len(context.confidence_scores) if context.confidence_scores else 0.0,
                        "correlation_patterns": len(context.correlation_patterns),
                        "shared_knowledge": context.shared_knowledge,
                        "agent_insights_detail": context.agent_insights,
                        "confidence_scores": context.confidence_scores,
                        "business_context": True
                    }
            
            # Get comprehensive A2A data
            a2a_data = {
                "total_messages_sent": sum(exec.a2a_messages_sent for exec in incident.executions.values()),
                "total_messages_received": sum(exec.a2a_messages_received for exec in incident.executions.values()),
                "active_collaborations": len(incident.a2a_collaborations),
                "cross_agent_insights": len(incident.cross_agent_insights),
                "collaboration_details": incident.a2a_collaborations,
                "message_breakdown": {
                    agent_id: {
                        "sent": execution.a2a_messages_sent,
                        "received": execution.a2a_messages_received,
                        "collaborations": len(execution.collaboration_sessions)
                    }
                    for agent_id, execution in incident.executions.items()
                },
                "business_collaboration_focus": True
            }
            
            # Enhanced execution details with log summaries
            execution_details = {}
            for agent_id, execution in incident.executions.items():
                execution_details[agent_id] = {
                    "agent_name": execution.agent_name,
                    "status": execution.status.value,
                    "progress": execution.progress,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "duration": execution.duration_seconds,
                    "error": execution.error_message,
                    "business_enhanced": True,
                    "detailed_logging": {
                        "total_log_entries": len(execution.logs),
                        "log_types": list(set(log.get("log_type", "INFO") for log in execution.logs)),
                        "business_context_logs": sum(1 for log in execution.logs if log.get("business_context")),
                        "mcp_related_logs": sum(1 for log in execution.logs if "MCP" in log.get("log_type", "")),
                        "a2a_related_logs": sum(1 for log in execution.logs if "A2A" in log.get("log_type", "")),
                        "analysis_logs": sum(1 for log in execution.logs if "ANALYSIS" in log.get("log_type", "")),
                        "logs_available": True
                    },
                    "mcp_enhanced": bool(execution.contextual_insights_used),
                    "a2a_messages": {
                        "sent": execution.a2a_messages_sent,
                        "received": execution.a2a_messages_received
                    },
                    "collaborations": len(execution.collaboration_sessions),
                    "contextual_insights": execution.contextual_insights_used,
                    "output_data": execution.output_data
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
                "p