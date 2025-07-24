# Requirements Agent - Specialized SDLC Agent
# Handles requirements gathering, analysis, and validation

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

# LangChain imports
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import BaseTool, tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Core framework imports
from core_agent_framework import (
    BaseSDLCAgent, AgentConfiguration, AgentCapability, 
    AgentContext, LLMProvider, AgentState
)

# Tool integrations
import requests
import httpx
from jira import JIRA
import confluence

class RequirementsAnalysisTool(BaseTool):
    """Tool for analyzing and structuring requirements"""
    
    name = "requirements_analysis"
    description = "Analyze raw requirements and structure them according to best practices"
    
    def _run(self, requirements_text: str, project_context: str = "") -> Dict:
        """Analyze requirements text"""
        # This would use NLP and domain knowledge to structure requirements
        analysis = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "business_requirements": [],
            "technical_requirements": [],
            "stakeholder_requirements": [],
            "gaps_identified": [],
            "conflicts_detected": [],
            "recommendations": []
        }
        
        # Simplified analysis logic (would be more sophisticated in production)
        lines = requirements_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(keyword in line.lower() for keyword in ['user', 'customer', 'interface']):
                analysis["functional_requirements"].append({
                    "text": line,
                    "priority": "medium",
                    "category": "user_interaction"
                })
            elif any(keyword in line.lower() for keyword in ['performance', 'scalability', 'security']):
                analysis["non_functional_requirements"].append({
                    "text": line,
                    "priority": "high",
                    "category": "system_quality"
                })
        
        return analysis
    
    async def _arun(self, requirements_text: str, project_context: str = "") -> Dict:
        """Async version"""
        return self._run(requirements_text, project_context)

class StakeholderInterviewTool(BaseTool):
    """Tool for conducting structured stakeholder interviews"""
    
    name = "stakeholder_interview"
    description = "Conduct structured interviews with stakeholders to gather requirements"
    
    def _run(self, stakeholder_type: str, interview_questions: List[str]) -> Dict:
        """Simulate stakeholder interview"""
        # In production, this would integrate with video conferencing APIs
        # or provide structured forms for stakeholders
        
        interview_results = {
            "stakeholder": stakeholder_type,
            "timestamp": datetime.utcnow().isoformat(),
            "responses": [],
            "follow_up_questions": [],
            "requirements_identified": [],
            "concerns_raised": []
        }
        
        # Simulate responses based on stakeholder type
        if stakeholder_type.lower() == "product_manager":
            interview_results["responses"] = [
                "We need to support 1000 concurrent users",
                "The system should integrate with our existing CRM",
                "Mobile-first design is critical",
                "We need real-time analytics dashboard"
            ]
            interview_results["requirements_identified"] = [
                "Concurrent user support (1000 users)",
                "CRM integration requirement",
                "Mobile-responsive design",
                "Real-time analytics"
            ]
        elif stakeholder_type.lower() == "end_user":
            interview_results["responses"] = [
                "The interface should be simple and intuitive",
                "I want to complete tasks quickly",
                "Notifications are important",
                "Works well on mobile devices"
            ]
            interview_results["requirements_identified"] = [
                "Intuitive user interface",
                "Task completion efficiency",
                "Notification system",
                "Mobile compatibility"
            ]
        
        return interview_results
    
    async def _arun(self, stakeholder_type: str, interview_questions: List[str]) -> Dict:
        """Async version"""
        return self._run(stakeholder_type, interview_questions)

class JiraIntegrationTool(BaseTool):
    """Tool for integrating with Jira for requirements management"""
    
    name = "jira_integration"
    description = "Create and manage requirements in Jira"
    
    def __init__(self, jira_url: str, username: str, api_token: str):
        super().__init__()
        self.jira_url = jira_url
        self.username = username
        self.api_token = api_token
        self.jira_client = None
        
    def _connect_jira(self):
        """Connect to Jira"""
        if not self.jira_client:
            self.jira_client = JIRA(
                server=self.jira_url,
                basic_auth=(self.username, self.api_token)
            )
        return self.jira_client
    
    def _run(self, action: str, **kwargs) -> Dict:
        """Execute Jira actions"""
        try:
            jira = self._connect_jira()
            
            if action == "create_epic":
                epic_data = {
                    'project': kwargs.get('project_key'),
                    'summary': kwargs.get('title'),
                    'description': kwargs.get('description'),
                    'issuetype': {'name': 'Epic'},
                    'customfield_10011': kwargs.get('epic_name')  # Epic Name field
                }
                
                epic = jira.create_issue(fields=epic_data)
                return {
                    "action": "create_epic",
                    "epic_key": epic.key,
                    "epic_id": epic.id,
                    "status": "created"
                }
                
            elif action == "create_story":
                story_data = {
                    'project': kwargs.get('project_key'),
                    'summary': kwargs.get('title'),
                    'description': kwargs.get('description'),
                    'issuetype': {'name': 'Story'},
                    'customfield_10014': kwargs.get('epic_link')  # Epic Link
                }
                
                story = jira.create_issue(fields=story_data)
                return {
                    "action": "create_story",
                    "story_key": story.key,
                    "story_id": story.id,
                    "status": "created"
                }
                
            elif action == "get_project_requirements":
                project_key = kwargs.get('project_key')
                issues = jira.search_issues(
                    f'project = {project_key} AND issuetype in (Epic, Story)',
                    expand='changelog'
                )
                
                requirements = []
                for issue in issues:
                    requirements.append({
                        "key": issue.key,
                        "summary": issue.fields.summary,
                        "description": issue.fields.description,
                        "status": issue.fields.status.name,
                        "priority": issue.fields.priority.name if issue.fields.priority else "Medium",
                        "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None
                    })
                
                return {
                    "action": "get_requirements",
                    "requirements": requirements,
                    "count": len(requirements)
                }
                
        except Exception as e:
            return {
                "action": action,
                "status": "error",
                "error": str(e)
            }
    
    async def _arun(self, action: str, **kwargs) -> Dict:
        """Async version"""
        return self._run(action, **kwargs)

class ConfluenceIntegrationTool(BaseTool):
    """Tool for creating and managing requirements documentation in Confluence"""
    
    name = "confluence_integration"
    description = "Create and manage requirements documentation in Confluence"
    
    def __init__(self, confluence_url: str, username: str, api_token: str):
        super().__init__()
        self.confluence_url = confluence_url
        self.username = username
        self.api_token = api_token
        
    def _run(self, action: str, **kwargs) -> Dict:
        """Execute Confluence actions"""
        # Simplified implementation
        if action == "create_requirements_page":
            return {
                "action": "create_page",
                "page_id": "12345",
                "page_url": f"{self.confluence_url}/pages/12345",
                "status": "created"
            }
        return {"action": action, "status": "not_implemented"}
    
    async def _arun(self, action: str, **kwargs) -> Dict:
        """Async version"""
        return self._run(action, **kwargs)

class RequirementsAgent(BaseSDLCAgent):
    """Requirements gathering and analysis agent"""
    
    def __init__(self, config: AgentConfiguration):
        # Define capabilities
        capabilities = [
            AgentCapability(
                name="gather_requirements",
                description="Gather requirements from stakeholders and documentation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "stakeholders": {"type": "array", "items": {"type": "string"}},
                        "existing_docs": {"type": "array", "items": {"type": "string"}}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "requirements": {"type": "array"},
                        "stakeholder_feedback": {"type": "object"},
                        "documentation": {"type": "string"}
                    }
                },
                tools=["requirements_analysis", "stakeholder_interview", "jira_integration"]
            ),
            AgentCapability(
                name="validate_requirements",
                description="Validate and prioritize requirements",
                input_schema={
                    "type": "object",
                    "properties": {
                        "requirements": {"type": "array"},
                        "business_objectives": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object", 
                    "properties": {
                        "validated_requirements": {"type": "array"},
                        "priority_matrix": {"type": "object"},
                        "conflicts": {"type": "array"}
                    }
                },
                tools=["requirements_analysis"]
            )
        ]
        
        super().__init__(config, capabilities)
        
        # Initialize specialized tools
        self.tools = self._initialize_tools()
        
        # Create LangChain agent
        self.langchain_agent = self._create_langchain_agent()
        
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize specialized tools for requirements agent"""
        tools = [
            RequirementsAnalysisTool(),
            StakeholderInterviewTool()
        ]
        
        # Add Jira integration if configured
        jira_config = self.config.tools_config.get('jira', {})
        if jira_config.get('enabled', False):
            tools.append(JiraIntegrationTool(
                jira_url=jira_config['url'],
                username=jira_config['username'],
                api_token=jira_config['api_token']
            ))
        
        # Add Confluence integration if configured
        confluence_config = self.config.tools_config.get('confluence', {})
        if confluence_config.get('enabled', False):
            tools.append(ConfluenceIntegrationTool(
                confluence_url=confluence_config['url'],
                username=confluence_config['username'],
                api_token=confluence_config['api_token']
            ))
        
        return tools
    
    def _create_langchain_agent(self) -> AgentExecutor:
        """Create LangChain agent with specialized prompt"""
        
        system_prompt = """You are a specialized Requirements Agent for software development lifecycle management.
        
        Your primary responsibilities:
        1. Gather comprehensive requirements from stakeholders
        2. Analyze and structure requirements according to best practices  
        3. Identify gaps, conflicts, and dependencies
        4. Validate requirements against business objectives
        5. Create clear, testable, and prioritized requirements documentation
        
        Available tools: {tool_names}
        
        When gathering requirements:
        - Conduct thorough stakeholder interviews
        - Analyze existing documentation
        - Identify functional, non-functional, and business requirements
        - Consider technical constraints and dependencies
        - Validate requirements for clarity, completeness, and feasibility
        
        Always provide structured output with clear reasoning for your decisions.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm_manager.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    async def reason(self, input_data: Dict) -> Dict:
        """Reasoning phase: Analyze requirements gathering context"""
        self.log_execution("reasoning_start", {"input": input_data})
        
        reasoning_prompt = f"""
        Analyze the following requirements gathering task and project context:
        
        Task: {json.dumps(input_data, indent=2)}
        Project Context: {json.dumps(self.context.shared_memory.get('project_context', {}), indent=2)}
        
        Provide analysis covering:
        1. Stakeholder identification and prioritization
        2. Requirements gathering approach and methods
        3. Expected types of requirements (functional, non-functional, business)
        4. Potential challenges and risks
        5. Success criteria and validation approach
        6. Timeline and resource requirements
        
        Consider:
        - Project complexity and scope
        - Stakeholder availability and expertise
        - Existing documentation and systems
        - Business objectives and constraints
        - Technical requirements and dependencies
        
        Provide structured reasoning with confidence scores for each analysis point.
        """
        
        # Use LLM for reasoning
        reasoning_response = await self.llm_manager.llm.ainvoke([
            HumanMessage(content=reasoning_prompt)
        ])
        
        # Parse reasoning response
        reasoning_result = {
            "task_understanding": "Requirements gathering and validation",
            "complexity_assessment": "medium",
            "stakeholder_analysis": {
                "primary_stakeholders": ["product_manager", "end_users", "technical_lead"],
                "secondary_stakeholders": ["business_analyst", "qa_lead"],
                "interview_priority": "high"
            },
            "requirements_scope": {
                "functional_requirements": "expected_high_volume",
                "non_functional_requirements": "critical_for_architecture",
                "business_requirements": "alignment_needed"
            },
            "approach_recommendation": "hybrid_stakeholder_interviews_and_document_analysis",
            "risks_identified": ["stakeholder_availability", "requirement_changes", "scope_creep"],
            "success_criteria": ["complete_requirements_coverage", "stakeholder_sign_off", "clear_priorities"],
            "confidence_score": 0.85,
            "reasoning_text": reasoning_response.content
        }
        
        self.log_execution("reasoning_complete", reasoning_result)
        return reasoning_result
    
    async def plan(self, reasoning_output: Dict) -> Dict:
        """Planning phase: Create requirements gathering plan"""
        self.log_execution("planning_start", {"reasoning": reasoning_output})
        
        planning_prompt = f"""
        Based on this reasoning analysis: {json.dumps(reasoning_output, indent=2)}
        
        Create a detailed step-by-step plan for requirements gathering including:
        
        1. Stakeholder Interview Plan:
           - Interview schedule and priorities
           - Structured questions for each stakeholder type
           - Interview methods and formats
        
        2. Documentation Analysis Plan:
           - Existing documents to review
           - Information extraction approach
           - Gap identification strategy
        
        3. Requirements Structuring Plan:
           - Classification approach (functional/non-functional/business)
           - Prioritization methodology
           - Validation and approval process
        
        4. Tool Integration Plan:
           - Jira epic and story creation
           - Confluence documentation structure
           - Traceability and version control
        
        5. Timeline and Milestones:
           - Task sequencing and dependencies
           - Estimated durations
           - Review and approval gates
        
        Provide specific, actionable steps with clear deliverables.
        """
        
        # Use LangChain agent for planning
        planning_response = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.langchain_agent.invoke,
            {"input": planning_prompt, "chat_history": []}
        )
        
        # Structure the plan
        plan = {
            "plan_id": f"req_plan_{int(time.time())}",
            "approach": "comprehensive_requirements_gathering",
            "phases": [
                {
                    "phase": "stakeholder_interviews",
                    "duration_hours": 8,
                    "steps": [
                        "schedule_interviews",
                        "prepare_questions",
                        "conduct_interviews", 
                        "analyze_responses"
                    ]
                },
                {
                    "phase": "document_analysis",
                    "duration_hours": 4,
                    "steps": [
                        "collect_existing_docs",
                        "analyze_content",
                        "extract_requirements",
                        "identify_gaps"
                    ]
                },
                {
                    "phase": "requirements_structuring",
                    "duration_hours": 6,
                    "steps": [
                        "categorize_requirements",
                        "prioritize_requirements",
                        "validate_completeness",
                        "resolve_conflicts"
                    ]
                },
                {
                    "phase": "documentation_creation",
                    "duration_hours": 4,
                    "steps": [
                        "create_jira_epics",
                        "create_user_stories", 
                        "document_in_confluence",
                        "get_stakeholder_approval"
                    ]
                }
            ],
            "tools_to_use": ["stakeholder_interview", "requirements_analysis", "jira_integration", "confluence_integration"],
            "deliverables": [
                "stakeholder_interview_reports",
                "requirements_specification_document",
                "jira_epic_and_stories", 
                "confluence_documentation",
                "requirements_traceability_matrix"
            ],
            "success_metrics": {
                "requirements_gathered": "target_50_plus",
                "stakeholder_satisfaction": "target_90_percent",
                "requirement_clarity": "all_testable"
            },
            "estimated_total_hours": 22,
            "planning_response": planning_response["output"]
        }
        
        self.log_execution("planning_complete", plan)
        return plan
    
    async def act(self, plan: Dict) -> Dict:
        """Action phase: Execute requirements gathering plan"""
        self.log_execution("acting_start", {"plan": plan})
        
        results = {
            "execution_id": f"req_exec_{int(time.time())}",
            "plan_id": plan["plan_id"],
            "phase_results": {},
            "overall_metrics": {},
            "deliverables_created": [],
            "issues_encountered": []
        }
        
        try:
            # Execute each phase
            for phase in plan["phases"]:
                phase_name = phase["phase"]
                self.log_execution(f"phase_start_{phase_name}", phase)
                
                phase_result = await self._execute_phase(phase, plan)
                results["phase_results"][phase_name] = phase_result
                
                self.log_execution(f"phase_complete_{phase_name}", phase_result)
            
            # Compile overall results
            results["overall_metrics"] = await self._compile_metrics(results)
            results["success"] = True
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            self.log_execution("acting_error", {"error": str(e)})
            
        self.log_execution("acting_complete", results)
        return results
    
    async def _execute_phase(self, phase: Dict, overall_plan: Dict) -> Dict:
        """Execute a specific phase of the requirements gathering plan"""
        phase_name = phase["phase"]
        
        if phase_name == "stakeholder_interviews":
            return await self._execute_stakeholder_interviews()
        elif phase_name == "document_analysis":
            return await self._execute_document_analysis()
        elif phase_name == "requirements_structuring":
            return await self._execute_requirements_structuring()
        elif phase_name == "documentation_creation":
            return await self._execute_documentation_creation()
        else:
            return {"status": "not_implemented", "phase": phase_name}
    
    async def _execute_stakeholder_interviews(self) -> Dict:
        """Execute stakeholder interviews"""
        interview_tool = next((tool for tool in self.tools if tool.name == "stakeholder_interview"), None)
        
        stakeholders = ["product_manager", "end_user", "technical_lead"]
        interview_results = {}
        
        for stakeholder in stakeholders:
            questions = [
                "What are the main objectives for this project?",
                "What are your key requirements and priorities?",
                "What are your main concerns or constraints?",
                "How do you measure success for this project?"
            ]
            
            result = await interview_tool._arun(stakeholder, questions)
            interview_results[stakeholder] = result
        
        return {
            "interviews_conducted": len(stakeholders),
            "stakeholders": stakeholders,
            "results": interview_results,
            "requirements_identified": sum(len(r["requirements_identified"]) for r in interview_results.values())
        }
    
    async def _execute_document_analysis(self) -> Dict:
        """Execute document analysis"""
        # Simulate document analysis
        return {
            "documents_analyzed": 3,
            "requirements_extracted": 12,
            "gaps_identified": 2,
            "conflicts_found": 1
        }
    
    async def _execute_requirements_structuring(self) -> Dict:
        """Execute requirements structuring"""
        analysis_tool = next((tool for tool in self.tools if tool.name == "requirements_analysis"), None)
        
        # Simulate requirements text from previous phases
        requirements_text = """
        User authentication and registration system
        Product catalog with search functionality
        Shopping cart and checkout process
        Payment processing integration
        Order tracking and management
        Admin dashboard for inventory management
        Mobile responsive design
        Performance requirements: 1000 concurrent users
        Security requirements: PCI DSS compliance
        Integration with existing CRM system
        """
        
        analysis_result = await analysis_tool._arun(requirements_text, "e-commerce project")
        
        return {
            "requirements_structured": True,
            "functional_requirements": len(analysis_result["functional_requirements"]),
            "non_functional_requirements": len(analysis_result["non_functional_requirements"]),
            "business_requirements": len(analysis_result["business_requirements"]),
            "conflicts_resolved": len(analysis_result["conflicts_detected"]),
            "analysis_details": analysis_result
        }
    
    async def _execute_documentation_creation(self) -> Dict:
        """Execute documentation creation"""
        documentation_results = {
            "jira_epics_created": 0,
            "jira_stories_created": 0,
            "confluence_pages_created": 0,
            "deliverables": []
        }
        
        # Create Jira epics and stories if tool is available
        jira_tool = next((tool for tool in self.tools if tool.name == "jira_integration"), None)
        if jira_tool:
            # Create epic
            epic_result = await jira_tool._arun(
                action="create_epic",
                project_key="ECOM",
                title="E-commerce Platform Requirements",
                description="Complete requirements for e-commerce platform project",
                epic_name="E-commerce Requirements"
            )
            
            if epic_result["status"] == "created":
                documentation_results["jira_epics_created"] = 1
                documentation_results["deliverables"].append(f"Jira Epic: {epic_result['epic_key']}")
                
                # Create user stories
                stories = [
                    {"title": "User Authentication System", "description": "Secure user registration and login"},
                    {"title": "Product Catalog", "description": "Browse and search products"},
                    {"title": "Shopping Cart", "description": "Add/remove products to cart"}
                ]
                
                for story in stories:
                    story_result = await jira_tool._arun(
                        action="create_story",
                        project_key="ECOM",
                        title=story["title"],
                        description=story["description"],
                        epic_link=epic_result["epic_key"]
                    )
                    
                    if story_result["status"] == "created":
                        documentation_results["jira_stories_created"] += 1
                        documentation_results["deliverables"].append(f"Jira Story: {story_result['story_key']}")
        
        # Create Confluence documentation if tool is available
        confluence_tool = next((tool for tool in self.tools if tool.name == "confluence_integration"), None)
        if confluence_tool:
            confluence_result = await confluence_tool._arun(
                action="create_requirements_page",
                space_key="PROJ",
                title="E-commerce Platform Requirements Specification",
                content="Detailed requirements specification document"
            )
            
            if confluence_result["status"] == "created":
                documentation_results["confluence_pages_created"] = 1
                documentation_results["deliverables"].append(f"Confluence Page: {confluence_result['page_id']}")
        
        return documentation_results
    
    async def _compile_metrics(self, results: Dict) -> Dict:
        """Compile overall execution metrics"""
        phase_results = results["phase_results"]
        
        total_requirements = 0
        total_stakeholders = 0
        total_deliverables = 0
        
        # Count from stakeholder interviews
        if "stakeholder_interviews" in phase_results:
            si_results = phase_results["stakeholder_interviews"]
            total_requirements += si_results.get("requirements_identified", 0)
            total_stakeholders += si_results.get("interviews_conducted", 0)
        
        # Count from document analysis
        if "document_analysis" in phase_results:
            da_results = phase_results["document_analysis"]
            total_requirements += da_results.get("requirements_extracted", 0)
        
        # Count from requirements structuring
        if "requirements_structuring" in phase_results:
            rs_results = phase_results["requirements_structuring"]
            total_requirements += rs_results.get("functional_requirements", 0)
            total_requirements += rs_results.get("non_functional_requirements", 0)
            total_requirements += rs_results.get("business_requirements", 0)
        
        # Count deliverables
        if "documentation_creation" in phase_results:
            dc_results = phase_results["documentation_creation"]
            total_deliverables = len(dc_results.get("deliverables", []))
        
        return {
            "total_requirements_gathered": total_requirements,
            "stakeholders_interviewed": total_stakeholders,
            "deliverables_created": total_deliverables,
            "execution_time_minutes": 45,  # Simulated
            "success_rate": 0.95,
            "completeness_score": 0.88,
            "stakeholder_satisfaction": 0.92
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_requirements_agent():
        # Configuration for testing
        config = AgentConfiguration(
            agent_id="requirements_agent_001",
            agent_type="requirements",
            llm_provider=LLMProvider.OPENAI,
            llm_model="gpt-4",
            api_key="your-openai-api-key",  # Replace with actual key
            enable_mcp=True,
            enable_a2a=True,
            tools_config={
                "jira": {
                    "enabled": False,  # Set to True with actual credentials
                    "url": "https://your-domain.atlassian.net",
                    "username": "your-email@company.com",
                    "api_token": "your-jira-api-token"
                },
                "confluence": {
                    "enabled": False,  # Set to True with actual credentials
                    "url": "https://your-domain.atlassian.net/wiki",
                    "username": "your-email@company.com",
                    "api_token": "your-confluence-api-token"
                }
            }
        )
        
        # Create agent
        agent = RequirementsAgent(config)
        
        # Test context
        context = AgentContext(
            project_id="ecommerce_project_001",
            session_id="test_session_001",
            workflow_id="test_workflow_001",
            shared_memory={
                "project_context": {
                    "name": "E-commerce Platform",
                    "description": "Modern e-commerce platform with microservices architecture",
                    "stakeholders": ["Product Manager", "End Users", "Technical Lead", "QA Lead"],
                    "business_objectives": ["Increase sales", "Improve user experience", "Scale to 1M users"]
                }
            }
        )
        
        # Test task
        task = {
            "type": "gather_requirements",
            "project_id": "ecommerce_project_001",
            "stakeholders": ["product_manager", "end_users", "technical_lead"],
            "existing_docs": ["business_case.pdf", "technical_spec_draft.md"],
            "deadline": "2025-02-15",
            "scope": "full_platform_requirements"
        }
        
        try:
            print("🚀 Starting Requirements Agent Test")
            print(f"Agent ID: {agent.agent_id}")
            print(f"Tools available: {[tool.name for tool in agent.tools]}")
            
            # Execute agent
            result = await agent.process(task, context)
            
            print("\n✅ Requirements Agent Execution Complete!")
            print(f"Success: {result['success']}")
            print(f"Execution time: {result['execution_time']:.2f}s")
            
            if result['success']:
                # Print reasoning summary
                reasoning = result['reasoning']
                print(f"\n🧠 Reasoning Summary:")
                print(f"  - Complexity: {reasoning['complexity_assessment']}")
                print(f"  - Stakeholders: {len(reasoning['stakeholder_analysis']['primary_stakeholders'])}")
                print(f"  - Confidence: {reasoning['confidence_score']}")
                
                # Print planning summary
                plan = result['plan']
                print(f"\n📋 Plan Summary:")
                print(f"  - Approach: {plan['approach']}")
                print(f"  - Phases: {len(plan['phases'])}")
                print(f"  - Estimated hours: {plan['estimated_total_hours']}")
                print(f"  - Tools to use: {', '.join(plan['tools_to_use'])}")
                
                # Print execution results
                execution_result = result['result']
                if execution_result['success']:
                    metrics = execution_result['overall_metrics']
                    print(f"\n📊 Execution Results:")
                    print(f"  - Requirements gathered: {metrics['total_requirements_gathered']}")
                    print(f"  - Stakeholders interviewed: {metrics['stakeholders_interviewed']}")
                    print(f"  - Deliverables created: {metrics['deliverables_created']}")
                    print(f"  - Success rate: {metrics['success_rate']:.1%}")
                    
                    # Print deliverables
                    for phase_name, phase_result in execution_result['phase_results'].items():
                        print(f"\n  📁 {phase_name.replace('_', ' ').title()}:")
                        if 'deliverables' in phase_result:
                            for deliverable in phase_result['deliverables']:
                                print(f"    - {deliverable}")
                
            else:
                print(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
                
            # Print recent logs
            print(f"\n📋 Recent Execution Logs:")
            for log_entry in result['logs'][-5:]:  # Last 5 log entries
                timestamp = datetime.fromisoformat(log_entry['timestamp']).strftime('%H:%M:%S')
                print(f"  [{timestamp}] {log_entry['stage']}: {log_entry['data'].get('status', 'in progress')}")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_requirements_agent())

# Integration patterns for production deployment
class RequirementsAgentFactory:
    """Factory for creating configured requirements agents"""
    
    @staticmethod
    def create_for_project(project_type: str, tools_config: Dict) -> RequirementsAgent:
        """Create requirements agent configured for specific project type"""
        
        # Base configuration
        config = AgentConfiguration(
            agent_id=f"requirements_agent_{project_type}_{int(time.time())}",
            agent_type="requirements",
            llm_provider=LLMProvider.OPENAI,
            llm_model="gpt-4",
            tools_config=tools_config
        )
        
        # Customize based on project type
        if project_type == "web_application":
            config.tools_config.update({
                "stakeholder_types": ["product_manager", "ux_designer", "end_users", "technical_lead"],
                "requirement_categories": ["functional", "non_functional", "ui_ux", "performance", "security"]
            })
        elif project_type == "mobile_application":
            config.tools_config.update({
                "stakeholder_types": ["product_manager", "mobile_users", "platform_engineers", "qa_mobile"],
                "requirement_categories": ["functional", "platform_specific", "performance", "offline_support"]
            })
        elif project_type == "data_platform":
            config.tools_config.update({
                "stakeholder_types": ["data_scientist", "business_analyst", "data_engineer", "compliance_officer"],
                "requirement_categories": ["data_requirements", "processing_requirements", "compliance", "performance"]
            })
        
        return RequirementsAgent(config)
    
    @staticmethod
    def create_with_integrations(jira_config: Dict, confluence_config: Dict, **kwargs) -> RequirementsAgent:
        """Create requirements agent with tool integrations"""
        tools_config = {
            "jira": jira_config,
            "confluence": confluence_config
        }
        tools_config.update(kwargs)
        
        config = AgentConfiguration(
            agent_id=f"requirements_agent_integrated_{int(time.time())}",
            agent_type="requirements",
            llm_provider=LLMProvider.OPENAI,
            llm_model="gpt-4",
            tools_config=tools_config
        )
        
        return RequirementsAgent(config)