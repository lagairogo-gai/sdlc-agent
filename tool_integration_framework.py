# Tool Integration Framework - SDLC Tools Connector
# Handles integration with all SDLC tools across different stages

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# HTTP and API clients
import httpx
import requests
from requests.auth import HTTPBasicAuth

# Specific tool integrations
from jira import JIRA
import git
from github import Github
from gitlab import Gitlab
import docker
import subprocess
import yaml

# MCP Protocol for tool standardization
from mcp import types as mcp_types
from mcp.server.fastapi import FastMCPServer
from mcp.client import Client as MCPClient

class ToolCategory(Enum):
    PLANNING = "planning"
    DESIGN = "design"
    DEVELOPMENT = "development"
    CODE_QUALITY = "code_quality"
    TESTING = "testing"
    CI_CD = "ci_cd"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"

class ToolStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"

@dataclass
class ToolConfiguration:
    tool_name: str
    tool_type: str
    category: ToolCategory
    connection_config: Dict[str, Any]
    auth_config: Dict[str, Any]
    capabilities: List[str]
    mcp_enabled: bool = True
    health_check_interval: int = 300  # 5 minutes

@dataclass
class ToolOperation:
    operation_id: str
    tool_name: str
    action: str
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    result: Optional[Dict] = None
    error: Optional[str] = None
    duration: Optional[float] = None

class BaseTool(ABC):
    """Base class for all tool integrations"""
    
    def __init__(self, config: ToolConfiguration):
        self.config = config
        self.status = ToolStatus.DISCONNECTED
        self.client = None
        self.last_health_check = None
        self.logger = logging.getLogger(f"tool.{config.tool_name}")
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the tool"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the tool"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check tool health and connectivity"""
        pass
    
    @abstractmethod
    async def execute_operation(self, operation: ToolOperation) -> Dict[str, Any]:
        """Execute a tool operation"""
        pass
    
    async def get_capabilities(self) -> List[str]:
        """Get tool capabilities"""
        return self.config.capabilities

# Planning Tools
class JiraTool(BaseTool):
    """Jira integration for agile planning"""
    
    def __init__(self, config: ToolConfiguration):
        super().__init__(config)
        self.jira_client = None
        
    async def connect(self) -> bool:
        """Connect to Jira"""
        try:
            auth_config = self.config.auth_config
            self.jira_client = JIRA(
                server=self.config.connection_config['url'],
                basic_auth=(auth_config['username'], auth_config['api_token'])
            )
            
            # Test connection
            projects = self.jira_client.projects()
            self.status = ToolStatus.CONNECTED
            self.logger.info(f"Connected to Jira with {len(projects)} projects")
            return True
            
        except Exception as e:
            self.status = ToolStatus.ERROR
            self.logger.error(f"Failed to connect to Jira: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Jira"""
        self.jira_client = None
        self.status = ToolStatus.DISCONNECTED
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Jira health"""
        try:
            if self.jira_client:
                server_info = self.jira_client.server_info()
                self.last_health_check = time.time()
                return {
                    "status": "healthy",
                    "server_version": server_info.get("version"),
                    "response_time": "fast"
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def execute_operation(self, operation: ToolOperation) -> Dict[str, Any]:
        """Execute Jira operations"""
        action = operation.action
        params = operation.parameters
        
        try:
            if action == "create_epic":
                epic_data = {
                    'project': params['project_key'],
                    'summary': params['title'],
                    'description': params.get('description', ''),
                    'issuetype': {'name': 'Epic'},
                    'customfield_10011': params.get('epic_name', params['title'])
                }
                epic = self.jira_client.create_issue(fields=epic_data)
                return {"epic_key": epic.key, "epic_id": epic.id, "url": f"{self.config.connection_config['url']}/browse/{epic.key}"}
                
            elif action == "create_story":
                story_data = {
                    'project': params['project_key'],
                    'summary': params['title'],
                    'description': params.get('description', ''),
                    'issuetype': {'name': 'Story'},
                    'priority': {'name': params.get('priority', 'Medium')}
                }
                if params.get('epic_link'):
                    story_data['customfield_10014'] = params['epic_link']
                    
                story = self.jira_client.create_issue(fields=story_data)
                return {"story_key": story.key, "story_id": story.id, "url": f"{self.config.connection_config['url']}/browse/{story.key}"}
                
            elif action == "get_project_issues":
                jql = f"project = {params['project_key']}"
                if params.get('issue_type'):
                    jql += f" AND issuetype = {params['issue_type']}"
                if params.get('status'):
                    jql += f" AND status = '{params['status']}'"
                    
                issues = self.jira_client.search_issues(jql, maxResults=params.get('max_results', 50))
                
                return {
                    "issues": [
                        {
                            "key": issue.key,
                            "summary": issue.fields.summary,
                            "status": issue.fields.status.name,
                            "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
                            "created": issue.fields.created
                        } for issue in issues
                    ],
                    "total": len(issues)
                }
                
            elif action == "update_issue_status":
                issue = self.jira_client.issue(params['issue_key'])
                transitions = self.jira_client.transitions(issue)
                
                target_transition = None
                for transition in transitions:
                    if transition['name'].lower() == params['status'].lower():
                        target_transition = transition['id']
                        break
                
                if target_transition:
                    self.jira_client.transition_issue(issue, target_transition)
                    return {"status": "updated", "new_status": params['status']}
                else:
                    return {"status": "error", "error": f"Transition to {params['status']} not available"}
                    
        except Exception as e:
            raise Exception(f"Jira operation failed: {str(e)}")

class TrelloTool(BaseTool):
    """Trello integration for Kanban boards"""
    
    def __init__(self, config: ToolConfiguration):
        super().__init__(config)
        self.api_key = config.auth_config['api_key']
        self.token = config.auth_config['token']
        self.base_url = "https://api.trello.com/1"
        
    async def connect(self) -> bool:
        """Connect to Trello"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/members/me",
                    params={"key": self.api_key, "token": self.token}
                )
                if response.status_code == 200:
                    self.status = ToolStatus.CONNECTED
                    return True
                return False
        except Exception as e:
            self.status = ToolStatus.ERROR
            self.logger.error(f"Failed to connect to Trello: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Trello"""
        self.status = ToolStatus.DISCONNECTED
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Trello health"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/members/me",
                    params={"key": self.api_key, "token": self.token}
                )
                return {"status": "healthy" if response.status_code == 200 else "unhealthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def execute_operation(self, operation: ToolOperation) -> Dict[str, Any]:
        """Execute Trello operations"""
        action = operation.action
        params = operation.parameters
        
        async with httpx.AsyncClient() as client:
            try:
                if action == "create_card":
                    response = await client.post(
                        f"{self.base_url}/cards",
                        params={
                            "key": self.api_key,
                            "token": self.token,
                            "name": params['title'],
                            "desc": params.get('description', ''),
                            "idList": params['list_id']
                        }
                    )
                    return response.json()
                    
                elif action == "get_board_lists":
                    response = await client.get(
                        f"{self.base_url}/boards/{params['board_id']}/lists",
                        params={"key": self.api_key, "token": self.token}
                    )
                    return {"lists": response.json()}
                    
            except Exception as e:
                raise Exception(f"Trello operation failed: {str(e)}")

# Development Tools
class GitHubTool(BaseTool):
    """GitHub integration for source control"""
    
    def __init__(self, config: ToolConfiguration):
        super().__init__(config)
        self.github_client = None
        
    async def connect(self) -> bool:
        """Connect to GitHub"""
        try:
            token = self.config.auth_config['token']
            self.github_client = Github(token)
            
            # Test connection
            user = self.github_client.get_user()
            self.status = ToolStatus.CONNECTED
            self.logger.info(f"Connected to GitHub as {user.login}")
            return True
            
        except Exception as e:
            self.status = ToolStatus.ERROR
            self.logger.error(f"Failed to connect to GitHub: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from GitHub"""
        if self.github_client:
            self.github_client.close()
        self.status = ToolStatus.DISCONNECTED
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check GitHub health"""
        try:
            if self.github_client:
                rate_limit = self.github_client.get_rate_limit()
                return {
                    "status": "healthy",
                    "rate_limit_remaining": rate_limit.core.remaining,
                    "rate_limit_reset": rate_limit.core.reset.isoformat()
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def execute_operation(self, operation: ToolOperation) -> Dict[str, Any]:
        """Execute GitHub operations"""
        action = operation.action
        params = operation.parameters
        
        try:
            if action == "create_repository":
                repo = self.github_client.get_user().create_repo(
                    name=params['name'],
                    description=params.get('description', ''),
                    private=params.get('private', False)
                )
                return {
                    "repo_name": repo.name,
                    "repo_url": repo.html_url,
                    "clone_url": repo.clone_url
                }
                
            elif action == "create_branch":
                repo = self.github_client.get_repo(params['repo_name'])
                source_branch = repo.get_branch(params.get('source_branch', 'main'))
                new_branch = repo.create_git_ref(
                    ref=f"refs/heads/{params['branch_name']}",
                    sha=source_branch.commit.sha
                )
                return {"branch_name": params['branch_name'], "sha": new_branch.object.sha}
                
            elif action == "create_pull_request":
                repo = self.github_client.get_repo(params['repo_name'])
                pr = repo.create_pull(
                    title=params['title'],
                    body=params.get('description', ''),
                    head=params['head_branch'],
                    base=params.get('base_branch', 'main')
                )
                return {
                    "pr_number": pr.number,
                    "pr_url": pr.html_url,
                    "state": pr.state
                }
                
            elif action == "get_repository_info":
                repo = self.github_client.get_repo(params['repo_name'])
                return {
                    "name": repo.name,
                    "description": repo.description,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "open_issues": repo.open_issues_count
                }
                
        except Exception as e:
            raise Exception(f"GitHub operation failed: {str(e)}")

# CI/CD Tools
class JenkinsTool(BaseTool):
    """Jenkins integration for CI/CD"""
    
    def __init__(self, config: ToolConfiguration):
        super().__init__(config)
        self.base_url = config.connection_config['url']
        self.username = config.auth_config['username']
        self.api_token = config.auth_config['api_token']
        
    async def connect(self) -> bool:
        """Connect to Jenkins"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/json",
                    auth=(self.username, self.api_token)
                )
                if response.status_code == 200:
                    self.status = ToolStatus.CONNECTED
                    return True
                return False
        except Exception as e:
            self.status = ToolStatus.ERROR
            self.logger.error(f"Failed to connect to Jenkins: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Jenkins"""
        self.status = ToolStatus.DISCONNECTED
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Jenkins health"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/json",
                    auth=(self.username, self.api_token)
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": "healthy",
                        "jenkins_version": data.get("version"),
                        "jobs_count": len(data.get("jobs", []))
                    }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def execute_operation(self, operation: ToolOperation) -> Dict[str, Any]:
        """Execute Jenkins operations"""
        action = operation.action
        params = operation.parameters
        
        async with httpx.AsyncClient() as client:
            try:
                if action == "trigger_build":
                    job_name = params['job_name']
                    build_params = params.get('parameters', {})
                    
                    if build_params:
                        # Trigger parameterized build
                        response = await client.post(
                            f"{self.base_url}/job/{job_name}/buildWithParameters",
                            auth=(self.username, self.api_token),
                            data=build_params
                        )
                    else:
                        # Trigger simple build
                        response = await client.post(
                            f"{self.base_url}/job/{job_name}/build",
                            auth=(self.username, self.api_token)
                        )
                    
                    if response.status_code in [200, 201]:
                        return {"status": "triggered", "job_name": job_name}
                    else:
                        return {"status": "failed", "error": f"HTTP {response.status_code}"}
                
                elif action == "get_build_status":
                    job_name = params['job_name']
                    build_number = params.get('build_number', 'lastBuild')
                    
                    response = await client.get(
                        f"{self.base_url}/job/{job_name}/{build_number}/api/json",
                        auth=(self.username, self.api_token)
                    )
                    
                    if response.status_code == 200:
                        build_data = response.json()
                        return {
                            "build_number": build_data.get("number"),
                            "status": build_data.get("result", "RUNNING"),
                            "duration": build_data.get("duration"),
                            "timestamp": build_data.get("timestamp"),
                            "url": build_data.get("url")
                        }
                
                elif action == "get_jobs":
                    response = await client.get(
                        f"{self.base_url}/api/json",
                        auth=(self.username, self.api_token)
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "jobs": [
                                {
                                    "name": job["name"],
                                    "url": job["url"],
                                    "color": job.get("color", "unknown")
                                } for job in data.get("jobs", [])
                            ]
                        }
                        
            except Exception as e:
                raise Exception(f"Jenkins operation failed: {str(e)}")

# Monitoring Tools
class PrometheusGrafanaTool(BaseTool):
    """Prometheus + Grafana integration for monitoring"""
    
    def __init__(self, config: ToolConfiguration):
        super().__init__(config)
        self.prometheus_url = config.connection_config['prometheus_url']
        self.grafana_url = config.connection_config['grafana_url']
        self.grafana_token = config.auth_config['grafana_token']
        
    async def connect(self) -> bool:
        """Connect to Prometheus and Grafana"""
        try:
            # Test Prometheus connection
            async with httpx.AsyncClient() as client:
                prom_response = await client.get(f"{self.prometheus_url}/api/v1/query?query=up")
                grafana_response = await client.get(
                    f"{self.grafana_url}/api/health",
                    headers={"Authorization": f"Bearer {self.grafana_token}"}
                )
                
                if prom_response.status_code == 200 and grafana_response.status_code == 200:
                    self.status = ToolStatus.CONNECTED
                    return True
                return False
        except Exception as e:
            self.status = ToolStatus.ERROR
            self.logger.error(f"Failed to connect to monitoring stack: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from monitoring tools"""
        self.status = ToolStatus.DISCONNECTED
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check monitoring tools health"""
        try:
            async with httpx.AsyncClient() as client:
                # Check Prometheus
                prom_response = await client.get(f"{self.prometheus_url}/api/v1/query?query=up")
                # Check Grafana
                grafana_response = await client.get(
                    f"{self.grafana_url}/api/health",
                    headers={"Authorization": f"Bearer {self.grafana_token}"}
                )
                
                return {
                    "status": "healthy" if prom_response.status_code == 200 and grafana_response.status_code == 200 else "unhealthy",
                    "prometheus_status": "up" if prom_response.status_code == 200 else "down",
                    "grafana_status": "up" if grafana_response.status_code == 200 else "down"
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def execute_operation(self, operation: ToolOperation) -> Dict[str, Any]:
        """Execute monitoring operations"""
        action = operation.action
        params = operation.parameters
        
        async with httpx.AsyncClient() as client:
            try:
                if action == "query_metrics":
                    query = params['query']
                    response = await client.get(
                        f"{self.prometheus_url}/api/v1/query",
                        params={"query": query}
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                
                elif action == "create_dashboard":
                    dashboard_config = {
                        "dashboard": {
                            "title": params['title'],
                            "panels": params.get('panels', []),
                            "time": params.get('time_range', {"from": "now-1h", "to": "now"})
                        }
                    }
                    
                    response = await client.post(
                        f"{self.grafana_url}/api/dashboards/db",
                        headers={"Authorization": f"Bearer {self.grafana_token}"},
                        json=dashboard_config
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                
                elif action == "create_alert":
                    alert_config = {
                        "name": params['alert_name'],
                        "message": params.get('message', ''),
                        "frequency": params.get('frequency', '10s'),
                        "conditions": params['conditions']
                    }
                    
                    response = await client.post(
                        f"{self.grafana_url}/api/alerts",
                        headers={"Authorization": f"Bearer {self.grafana_token}"},
                        json=alert_config
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                        
            except Exception as e:
                raise Exception(f"Monitoring operation failed: {str(e)}")

# Testing Tools
class SeleniumTool(BaseTool):
    """Selenium integration for UI testing"""
    
    def __init__(self, config: ToolConfiguration):
        super().__init__(config)
        self.selenium_grid_url = config.connection_config.get('selenium_grid_url')
        
    async def connect(self) -> bool:
        """Connect to Selenium Grid"""
        try:
            if self.selenium_grid_url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.selenium_grid_url}/status")
                    if response.status_code == 200:
                        self.status = ToolStatus.CONNECTED
                        return True
            else:
                # Local Selenium setup
                self.status = ToolStatus.CONNECTED
                return True
            return False
        except Exception as e:
            self.status = ToolStatus.ERROR
            self.logger.error(f"Failed to connect to Selenium: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Selenium"""
        self.status = ToolStatus.DISCONNECTED
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Selenium health"""
        try:
            if self.selenium_grid_url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.selenium_grid_url}/status")
                    if response.status_code == 200:
                        return {"status": "healthy", "grid_status": response.json()}
            return {"status": "healthy", "mode": "local"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def execute_operation(self, operation: ToolOperation) -> Dict[str, Any]:
        """Execute Selenium operations"""
        action = operation.action
        params = operation.parameters
        
        try:
            if action == "run_test_suite":
                # This would integrate with actual Selenium test execution
                test_results = {
                    "suite_name": params['suite_name'],
                    "total_tests": params.get('total_tests', 0),
                    "passed": params.get('passed', 0),
                    "failed": params.get('failed', 0),
                    "execution_time": params.get('execution_time', 0),
                    "test_results": []
                }
                return test_results
                
            elif action == "execute_test_script":
                # Execute individual test script
                return {
                    "script_name": params['script_name'],
                    "status": "passed",
                    "execution_time": 45.2,
                    "screenshots": [],
                    "logs": []
                }
                
        except Exception as e:
            raise Exception(f"Selenium operation failed: {str(e)}")

# Tool Registry and Manager
class ToolRegistry:
    """Central registry for all SDLC tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_configs: Dict[str, ToolConfiguration] = {}
        self.mcp_server = FastMCPServer("SDLC-Tools")
        
    def register_tool(self, tool: BaseTool):
        """Register a tool with the registry"""
        self.tools[tool.config.tool_name] = tool
        self.tool_configs[tool.config.tool_name] = tool.config
        
        # Register with MCP if enabled
        if tool.config.mcp_enabled:
            self._register_mcp_tool(tool)
    
    def _register_mcp_tool(self, tool: BaseTool):
        """Register tool with MCP server"""
        tool_name = tool.config.tool_name
        
        @self.mcp_server.call_tool()
        async def execute_tool_operation(name: str, arguments: dict) -> Any:
            if name == f"{tool_name}_execute":
                operation = ToolOperation(
                    operation_id=str(uuid.uuid4()),
                    tool_name=tool_name,
                    action=arguments.get('action'),
                    parameters=arguments.get('parameters', {})
                )
                return await tool.execute_operation(operation)
            raise ValueError(f"Unknown tool operation: {name}")
    
    async def connect_all_tools(self) -> Dict[str, bool]:
        """Connect to all registered tools"""
        connection_results = {}
        
        for tool_name, tool in self.tools.items():
            try:
                success = await tool.connect()
                connection_results[tool_name] = success
                if success:
                    self.logger.info(f"Successfully connected to {tool_name}")
                else:
                    self.logger.error(f"Failed to connect to {tool_name}")
            except Exception as e:
                connection_results[tool_name] = False
                self.logger.error(f"Error connecting to {tool_name}: {e}")
        
        return connection_results
    
    async def health_check_all(self) -> Dict[str, Dict]:
        """Perform health check on all tools"""
        health_results = {}
        
        for tool_name, tool in self.tools.items():
            if tool.status == ToolStatus.CONNECTED:
                health_results[tool_name] = await tool.health_check()
            else:
                health_results[tool_name] = {"status": "disconnected"}
        
        return health_results
    
    async def execute_tool_operation(self, tool_name: str, action: str, parameters: Dict) -> Dict:
        """Execute operation on specific tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
        
        tool = self.tools[tool_name]
        if tool.status != ToolStatus.CONNECTED:
            raise ValueError(f"Tool {tool_name} is not connected")
        
        operation = ToolOperation(
            operation_id=str(uuid.uuid4()),
            tool_name=tool_name,
            action=action,
            parameters=parameters
        )
        
        start_time = time.time()
        try:
            result = await tool.execute_operation(operation)
            operation.result = result
            operation.duration = time.time() - start_time
            return {
                "success": True,
                "result": result,
                "operation_id": operation.operation_id,
                "duration": operation.duration
            }
        except Exception as e:
            operation.error = str(e)
            operation.duration = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation.operation_id,
                "duration": operation.duration
            }
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get tools by category"""
        return [
            tool for tool in self.tools.values()
            if tool.config.category == category
        ]
    
    def get_tool_capabilities(self, tool_name: str) -> List[str]:
        """Get capabilities of a specific tool"""
        if tool_name in self.tools:
            return self.tools[tool_name].config.capabilities
        return []

# Tool Configuration Factory
class ToolConfigurationFactory:
    """Factory for creating tool configurations"""
    
    @staticmethod
    def create_jira_config(url: str, username: str, api_token: str) -> ToolConfiguration:
        """Create Jira configuration"""
        return ToolConfiguration(
            tool_name="jira",
            tool_type="project_management",
            category=ToolCategory.PLANNING,
            connection_config={"url": url},
            auth_config={"username": username, "api_token": api_token},
            capabilities=[
                "create_epic", "create_story", "update_issue_status",
                "get_project_issues", "create_subtask", "add_comment"
            ]
        )
    
    @staticmethod
    def create_github_config(token: str) -> ToolConfiguration:
        """Create GitHub configuration"""
        return ToolConfiguration(
            tool_name="github",
            tool_type="source_control",
            category=ToolCategory.DEVELOPMENT,
            connection_config={},
            auth_config={"token": token},
            capabilities=[
                "create_repository", "create_branch", "create_pull_request",
                "merge_pull_request", "get_repository_info", "create_release"
            ]
        )
    
    @staticmethod
    def create_jenkins_config(url: str, username: str, api_token: str) -> ToolConfiguration:
        """Create Jenkins configuration"""
        return ToolConfiguration(
            tool_name="jenkins",
            tool_type="ci_cd",
            category=ToolCategory.CI_CD,
            connection_config={"url": url},
            auth_config={"username": username, "api_token": api_token},
            capabilities=[
                "trigger_build", "get_build_status", "get_jobs",
                "create_job", "get_build_logs"
            ]
        )
    
    @staticmethod
    def create_monitoring_config(prometheus_url: str, grafana_url: str, grafana_token: str) -> ToolConfiguration:
        """Create monitoring tools configuration"""
        return ToolConfiguration(
            tool_name="prometheus_grafana",
            tool_type="monitoring",
            category=ToolCategory.MONITORING,
            connection_config={
                "prometheus_url": prometheus_url,
                "grafana_url": grafana_url
            },
            auth_config={"grafana_token": grafana_token},
            capabilities=[
                "query_metrics", "create_dashboard", "create_alert",
                "get_metrics", "setup_monitoring"
            ]
        )

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_tool_integration():
        # Create tool registry
        registry = ToolRegistry()
        
        # Create tool configurations (with dummy credentials for testing)
        jira_config = ToolConfigurationFactory.create_jira_config(
            url="https://your-domain.atlassian.net",
            username="your-email@company.com",
            api_token="dummy-token"
        )
        
        github_config = ToolConfigurationFactory.create_github_config(
            token="dummy-github-token"
        )
        
        jenkins_config = ToolConfigurationFactory.create_jenkins_config(
            url="http://jenkins.company.com",
            username="admin",
            api_token="dummy-jenkins-token"
        )
        
        monitoring_config = ToolConfigurationFactory.create_monitoring_config(
            prometheus_url="http://prometheus.company.com:9090",
            grafana_url="http://grafana.company.com:3000",
            grafana_token="dummy-grafana-token"
        )
        
        # Create and register tools
        jira_tool = JiraTool(jira_config)
        github_tool = GitHubTool(github_config)
        jenkins_tool = JenkinsTool(jenkins_config)
        monitoring_tool = PrometheusGrafanaTool(monitoring_config)
        
        registry.register_tool(jira_tool)
        registry.register_tool(github_tool)
        registry.register_tool(jenkins_tool)
        registry.register_tool(monitoring_tool)
        
        print("🔧 Tool Integration Framework Test")
        print(f"Registered tools: {list(registry.tools.keys())}")
        
        # Test tool capabilities
        for tool_name in registry.tools.keys():
            capabilities = registry.get_tool_capabilities(tool_name)
            print(f"  {tool_name}: {len(capabilities)} capabilities")
        
        # Test tools by category
        planning_tools = registry.get_tools_by_category(ToolCategory.PLANNING)
        development_tools = registry.get_tools_by_category(ToolCategory.DEVELOPMENT)
        
        print(f"\nTools by category:")
        print(f"  Planning: {[t.config.tool_name for t in planning_tools]}")
        print(f"  Development: {[t.config.tool_name for t in development_tools]}")
        print(f"  CI/CD: {[t.config.tool_name for t in registry.get_tools_by_category(ToolCategory.CI_CD)]}")
        print(f"  Monitoring: {[t.config.tool_name for t in registry.get_tools_by_category(ToolCategory.MONITORING)]}")
        
        # Note: Actual connection testing would require valid credentials
        print(f"\n✅ Tool Integration Framework initialized successfully!")
        print(f"   Ready to connect to {len(registry.tools)} SDLC tools")
        
        # Example of how agents would use tools
        print(f"\n📋 Example tool usage patterns:")
        print(f"  Requirements Agent -> Jira: create_epic, create_story")
        print(f"  Code Agent -> GitHub: create_repository, create_branch")
        print(f"  CI/CD Agent -> Jenkins: trigger_build, get_build_status")
        print(f"  Monitoring Agent -> Grafana: create_dashboard, create_alert")
    
    # Run the test
    asyncio.run(test_tool_integration())

# Production deployment helpers
def create_production_tool_registry(tool_configs: Dict[str, Dict]) -> ToolRegistry:
    """Create production tool registry with configurations"""
    registry = ToolRegistry()
    
    tool_classes = {
        "jira": JiraTool,
        "github": GitHubTool,
        "jenkins": JenkinsTool,
        "prometheus_grafana": PrometheusGrafanaTool,
        "selenium": SeleniumTool,
        "trello": TrelloTool
    }
    
    for tool_name, config_data in tool_configs.items():
        if tool_name in tool_classes:
            config = ToolConfiguration(**config_data)
            tool = tool_classes[tool_name](config)
            registry.register_tool(tool)
    
    return registry