# Knowledge Graph & Context Management System
# Manages project context, agent decisions, and learning from outcomes

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import networkx as nx
from neo4j import GraphDatabase, AsyncGraphDatabase
import redis.asyncio as redis
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer

class EntityType(Enum):
    PROJECT = "project"
    REQUIREMENT = "requirement"
    DESIGN = "design"
    CODE = "code"
    TEST = "test"
    DEPLOYMENT = "deployment"
    AGENT = "agent"
    DECISION = "decision"
    ARTIFACT = "artifact"
    TOOL = "tool"

class RelationType(Enum):
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    DEPLOYS = "deploys"
    CREATED_BY = "created_by"
    USED_BY = "used_by"
    LEARNED_FROM = "learned_from"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"

@dataclass
class Entity:
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class Relationship:
    id: str
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)

@dataclass
class ContextQuery:
    entity_id: str
    relation_types: List[RelationType] = field(default_factory=list)
    max_depth: int = 2
    max_results: int = 50
    include_embeddings: bool = False

class KnowledgeGraph:
    """Neo4j-based knowledge graph for project context"""
    
    def __init__(self, neo4j_uri: str, neo4j_auth: Tuple[str, str]):
        self.driver = AsyncGraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def close(self):
        """Close database connection"""
        await self.driver.close()
        
    async def create_entity(self, entity: Entity) -> str:
        """Create entity in knowledge graph"""
        # Generate embedding if text properties exist
        text_content = self._extract_text_content(entity)
        if text_content:
            entity.embedding = self.embedding_model.encode(text_content).tolist()
            
        async with self.driver.session() as session:
            result = await session.run(
                """
                CREATE (e:Entity {
                    id: $id,
                    type: $type,
                    name: $name,
                    properties: $properties,
                    embedding: $embedding,
                    created_at: $created_at,
                    updated_at: $updated_at
                })
                RETURN e.id as id
                """,
                id=entity.id,
                type=entity.type.value,
                name=entity.name,
                properties=json.dumps(entity.properties),
                embedding=entity.embedding,
                created_at=entity.created_at,
                updated_at=entity.updated_at
            )
            return entity.id
    
    async def create_relationship(self, relationship: Relationship) -> str:
        """Create relationship in knowledge graph"""
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (a:Entity {id: $source_id})
                MATCH (b:Entity {id: $target_id})
                CREATE (a)-[r:RELATES {
                    id: $id,
                    type: $type,
                    properties: $properties,
                    weight: $weight,
                    created_at: $created_at
                }]->(b)
                """,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                id=relationship.id,
                type=relationship.type.value,
                properties=json.dumps(relationship.properties),
                weight=relationship.weight,
                created_at=relationship.created_at
            )
            return relationship.id
    
    async def get_context(self, query: ContextQuery) -> Dict[str, Any]:
        """Get related context for an entity"""
        async with self.driver.session() as session:
            # Build dynamic query based on parameters
            cypher_query = """
            MATCH (center:Entity {id: $entity_id})
            CALL apoc.path.expandConfig(center, {
                maxLevel: $max_depth,
                limit: $max_results
            }) YIELD path
            WITH nodes(path) as path_nodes, relationships(path) as path_rels
            UNWIND path_nodes as node
            UNWIND path_rels as rel
            RETURN DISTINCT
                node.id as node_id,
                node.type as node_type,
                node.name as node_name,
                node.properties as node_properties,
                node.embedding as node_embedding,
                rel.id as rel_id,
                rel.type as rel_type,
                rel.properties as rel_properties,
                rel.weight as rel_weight
            """
            
            result = await session.run(
                cypher_query,
                entity_id=query.entity_id,
                max_depth=query.max_depth,
                max_results=query.max_results
            )
            
            # Process results
            entities = {}
            relationships = []
            
            async for record in result:
                if record["node_id"]:
                    entities[record["node_id"]] = {
                        "id": record["node_id"],
                        "type": record["node_type"],
                        "name": record["node_name"],
                        "properties": json.loads(record["node_properties"] or "{}"),
                        "embedding": record["node_embedding"] if query.include_embeddings else None
                    }
                
                if record["rel_id"]:
                    relationships.append({
                        "id": record["rel_id"],
                        "type": record["rel_type"],
                        "properties": json.loads(record["rel_properties"] or "{}"),
                        "weight": record["rel_weight"]
                    })
            
            return {
                "center_entity": query.entity_id,
                "entities": entities,
                "relationships": relationships,
                "query_params": {
                    "max_depth": query.max_depth,
                    "max_results": query.max_results
                }
            }
    
    async def find_similar_entities(self, entity_id: str, limit: int = 10) -> List[Dict]:
        """Find similar entities using embeddings"""
        async with self.driver.session() as session:
            # Get entity embedding
            result = await session.run(
                "MATCH (e:Entity {id: $entity_id}) RETURN e.embedding as embedding",
                entity_id=entity_id
            )
            
            record = await result.single()
            if not record or not record["embedding"]:
                return []
                
            target_embedding = record["embedding"]
            
            # Find similar entities using cosine similarity
            similar_result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.id <> $entity_id AND e.embedding IS NOT NULL
                WITH e, apoc.coll.cosine(e.embedding, $target_embedding) as similarity
                WHERE similarity > 0.7
                RETURN e.id as id, e.name as name, e.type as type, 
                       similarity, e.properties as properties
                ORDER BY similarity DESC
                LIMIT $limit
                """,
                entity_id=entity_id,
                target_embedding=target_embedding,
                limit=limit
            )
            
            similar_entities = []
            async for record in similar_result:
                similar_entities.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "similarity": record["similarity"],
                    "properties": json.loads(record["properties"] or "{}")
                })
                
            return similar_entities
    
    async def add_agent_decision(self, agent_id: str, decision: Dict) -> str:
        """Record agent decision for learning"""
        decision_id = str(uuid.uuid4())
        
        # Create decision entity
        decision_entity = Entity(
            id=decision_id,
            type=EntityType.DECISION,
            name=f"Decision by {agent_id}",
            properties={
                "agent_id": agent_id,
                "context": decision.get("context", {}),
                "reasoning": decision.get("reasoning", ""),
                "action_taken": decision.get("action", ""),
                "outcome": decision.get("outcome", ""),
                "confidence": decision.get("confidence", 0.5),
                "success": decision.get("success", False)
            }
        )
        
        await self.create_entity(decision_entity)
        
        # Link to agent
        agent_relationship = Relationship(
            id=str(uuid.uuid4()),
            source_id=agent_id,
            target_id=decision_id,
            type=RelationType.CREATED_BY,
            properties={"decision_type": "autonomous"}
        )
        
        await self.create_relationship(agent_relationship)
        
        return decision_id
    
    async def get_agent_learning_context(self, agent_id: str, context_type: str) -> List[Dict]:
        """Get learning context for agent based on past decisions"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (agent:Entity {id: $agent_id})-[:CREATED_BY]->(decision:Entity {type: 'decision'})
                WHERE decision.properties CONTAINS $context_type
                WITH decision
                ORDER BY decision.created_at DESC
                LIMIT 20
                RETURN decision.properties as decision_data, decision.created_at as timestamp
                """,
                agent_id=agent_id,
                context_type=context_type
            )
            
            learning_data = []
            async for record in result:
                decision_data = json.loads(record["decision_data"])
                learning_data.append({
                    "timestamp": record["timestamp"],
                    "context": decision_data.get("context", {}),
                    "reasoning": decision_data.get("reasoning", ""),
                    "action": decision_data.get("action_taken", ""),
                    "outcome": decision_data.get("outcome", ""),
                    "success": decision_data.get("success", False),
                    "confidence": decision_data.get("confidence", 0.5)
                })
            
            return learning_data
    
    def _extract_text_content(self, entity: Entity) -> str:
        """Extract text content for embedding generation"""
        text_parts = [entity.name]
        
        # Extract text from properties
        for key, value in entity.properties.items():
            if isinstance(value, str) and len(value) > 10:
                text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)

class ContextManager:
    """Manages contextual information for agents"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, redis_client):
        self.knowledge_graph = knowledge_graph
        self.redis_client = redis_client
        self.context_cache = {}
        
    async def create_project_context(self, project_id: str, project_data: Dict) -> str:
        """Create project context in knowledge graph"""
        # Create project entity
        project_entity = Entity(
            id=project_id,
            type=EntityType.PROJECT,
            name=project_data.get("name", f"Project {project_id}"),
            properties={
                "description": project_data.get("description", ""),
                "requirements": project_data.get("requirements", {}),
                "technology_stack": project_data.get("technology_stack", []),
                "timeline": project_data.get("timeline", {}),
                "stakeholders": project_data.get("stakeholders", [])
            }
        )
        
        await self.knowledge_graph.create_entity(project_entity)
        
        # Create requirement entities
        requirements = project_data.get("requirements", {})
        for req_id, req_data in requirements.items():
            req_entity = Entity(
                id=f"{project_id}_req_{req_id}",
                type=EntityType.REQUIREMENT,
                name=req_data.get("name", f"Requirement {req_id}"),
                properties=req_data
            )
            
            await self.knowledge_graph.create_entity(req_entity)
            
            # Link to project
            req_relationship = Relationship(
                id=str(uuid.uuid4()),
                source_id=project_id,
                target_id=req_entity.id,
                type=RelationType.PART_OF
            )
            
            await self.knowledge_graph.create_relationship(req_relationship)
        
        # Cache project context
        await self._cache_project_context(project_id, project_entity)
        
        return project_id
    
    async def get_agent_context(self, agent_id: str, project_id: str, task_type: str) -> Dict:
        """Get contextual information for agent task"""
        # Get project context
        project_context = await self._get_project_context(project_id)
        
        # Get agent's learning context
        learning_context = await self.knowledge_graph.get_agent_learning_context(
            agent_id, task_type
        )
        
        # Get related entities and context
        context_query = ContextQuery(
            entity_id=project_id,
            max_depth=3,
            max_results=100,
            include_embeddings=False
        )
        
        graph_context = await self.knowledge_graph.get_context(context_query)
        
        # Find similar projects for reference
        similar_projects = await self.knowledge_graph.find_similar_entities(
            project_id, limit=5
        )
        
        return {
            "project": project_context,
            "learning_history": learning_context,
            "related_entities": graph_context,
            "similar_projects": similar_projects,
            "context_generated_at": time.time()
        }
    
    async def update_context_with_result(self, agent_id: str, project_id: str, 
                                       task_result: Dict) -> str:
        """Update context with task result"""
        # Create artifact entity for the result
        artifact_id = str(uuid.uuid4())
        artifact_entity = Entity(
            id=artifact_id,
            type=EntityType.ARTIFACT,
            name=f"Artifact from {agent_id}",
            properties={
                "agent_id": agent_id,
                "project_id": project_id,
                "task_type": task_result.get("task_type", ""),
                "result_data": task_result.get("result", {}),
                "success": task_result.get("success", False),
                "execution_time": task_result.get("execution_time", 0)
            }
        )
        
        await self.knowledge_graph.create_entity(artifact_entity)
        
        # Link artifact to project
        artifact_project_rel = Relationship(
            id=str(uuid.uuid4()),
            source_id=project_id,
            target_id=artifact_id,
            type=RelationType.PART_OF
        )
        
        await self.knowledge_graph.create_relationship(artifact_project_rel)
        
        # Link artifact to agent
        artifact_agent_rel = Relationship(
            id=str(uuid.uuid4()),
            source_id=agent_id,
            target_id=artifact_id,
            type=RelationType.CREATED_BY
        )
        
        await self.knowledge_graph.create_relationship(artifact_agent_rel)
        
        # Record agent decision
        decision_data = {
            "context": task_result.get("context", {}),
            "reasoning": task_result.get("reasoning", {}),
            "action": task_result.get("plan", {}),
            "outcome": task_result.get("result", {}),
            "success": task_result.get("success", False),
            "confidence": task_result.get("confidence", 0.8)
        }
        
        await self.knowledge_graph.add_agent_decision(agent_id, decision_data)
        
        # Invalidate cached context
        await self._invalidate_project_cache(project_id)
        
        return artifact_id
    
    async def _get_project_context(self, project_id: str) -> Dict:
        """Get cached or fresh project context"""
        # Try cache first
        cached = await self.redis_client.get(f"project_context:{project_id}")
        if cached:
            return json.loads(cached)
        
        # Get from knowledge graph
        context_query = ContextQuery(
            entity_id=project_id,
            max_depth=2,
            max_results=50
        )
        
        context = await self.knowledge_graph.get_context(context_query)
        
        # Cache for 1 hour
        await self.redis_client.setex(
            f"project_context:{project_id}",
            3600,
            json.dumps(context)
        )
        
        return context
    
    async def _cache_project_context(self, project_id: str, project_entity: Entity):
        """Cache project context"""
        context_data = {
            "id": project_entity.id,
            "name": project_entity.name,
            "properties": project_entity.properties,
            "cached_at": time.time()
        }
        
        await self.redis_client.setex(
            f"project_context:{project_id}",
            3600,  # 1 hour
            json.dumps(context_data)
        )
    
    async def _invalidate_project_cache(self, project_id: str):
        """Invalidate project context cache"""
        await self.redis_client.delete(f"project_context:{project_id}")

class ContextAPI:
    """API for context management operations"""
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        
    async def create_project(self, project_data: Dict) -> Dict:
        """Create new project context"""
        project_id = project_data.get("id", str(uuid.uuid4()))
        
        await self.context_manager.create_project_context(project_id, project_data)
        
        return {
            "project_id": project_id,
            "status": "created",
            "timestamp": time.time()
        }
    
    async def get_context_for_agent(self, agent_id: str, project_id: str, 
                                  task_type: str) -> Dict:
        """Get context for agent task"""
        context = await self.context_manager.get_agent_context(
            agent_id, project_id, task_type
        )
        
        return {
            "agent_id": agent_id,
            "project_id": project_id,
            "task_type": task_type,
            "context": context,
            "retrieved_at": time.time()
        }
    
    async def update_with_result(self, agent_id: str, project_id: str, 
                               result: Dict) -> Dict:
        """Update context with task result"""
        artifact_id = await self.context_manager.update_context_with_result(
            agent_id, project_id, result
        )
        
        return {
            "artifact_id": artifact_id,
            "status": "updated",
            "timestamp": time.time()
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Initialize components
        knowledge_graph = KnowledgeGraph(
            neo4j_uri="bolt://localhost:7687",
            neo4j_auth=("neo4j", "password")
        )
        
        redis_client = redis.from_url("redis://localhost:6379")
        context_manager = ContextManager(knowledge_graph, redis_client)
        context_api = ContextAPI(context_manager)
        
        # Example project data
        project_data = {
            "id": "ecommerce_project_001",
            "name": "E-commerce Platform",
            "description": "Modern e-commerce platform with microservices architecture",
            "requirements": {
                "req_001": {
                    "name": "User Authentication",
                    "description": "Secure user registration and login system",
                    "priority": "high",
                    "type": "functional"
                },
                "req_002": {
                    "name": "Product Catalog",
                    "description": "Product browsing and search functionality",
                    "priority": "high",
                    "type": "functional"
                },
                "req_003": {
                    "name": "Shopping Cart",
                    "description": "Add/remove products to cart functionality",
                    "priority": "medium",
                    "type": "functional"
                }
            },
            "technology_stack": ["React", "Node.js", "PostgreSQL", "Redis", "Docker"],
            "timeline": {
                "start_date": "2025-01-01",
                "end_date": "2025-06-30",
                "phases": {
                    "planning": "2025-01-01 to 2025-01-15",
                    "development": "2025-01-16 to 2025-05-31",
                    "testing": "2025-05-01 to 2025-06-15",
                    "deployment": "2025-06-16 to 2025-06-30"
                }
            },
            "stakeholders": ["Product Manager", "Lead Developer", "QA Lead", "DevOps Engineer"]
        }
        
        try:
            # Create project context
            print("Creating project context...")
            result = await context_api.create_project(project_data)
            print(f"Project created: {result}")
            
            # Simulate getting context for requirements agent
            print("\nGetting context for requirements agent...")
            agent_context = await context_api.get_context_for_agent(
                agent_id="requirements_agent_001",
                project_id="ecommerce_project_001",
                task_type="gather_requirements"
            )
            print(f"Agent context retrieved with {len(agent_context['context']['related_entities']['entities'])} related entities")
            
            # Simulate task result
            task_result = {
                "task_type": "gather_requirements",
                "success": True,
                "execution_time": 145.2,
                "result": {
                    "requirements_gathered": 15,
                    "stakeholders_interviewed": 4,
                    "requirements_validated": True,
                    "additional_requirements": [
                        "Payment processing integration",
                        "Order tracking system",
                        "Admin dashboard"
                    ]
                },
                "reasoning": {
                    "approach": "stakeholder_interviews",
                    "validation_method": "requirement_review_sessions",
                    "confidence_score": 0.92
                },
                "plan": {
                    "steps_completed": ["interview_product_manager", "interview_lead_dev", "validate_requirements"],
                    "total_steps": 3
                },
                "confidence": 0.92
            }
            
            # Update context with result
            print("\nUpdating context with task result...")
            update_result = await context_api.update_with_result(
                agent_id="requirements_agent_001",
                project_id="ecommerce_project_001",
                result=task_result
            )
            print(f"Context updated: {update_result}")
            
            # Test similarity search
            print("\nTesting similarity search...")
            similar_entities = await knowledge_graph.find_similar_entities(
                "ecommerce_project_001", limit=3
            )
            print(f"Found {len(similar_entities)} similar entities")
            
            # Get learning context for agent
            print("\nGetting learning context for agent...")
            learning_context = await knowledge_graph.get_agent_learning_context(
                "requirements_agent_001", "gather_requirements"
            )
            print(f"Learning context contains {len(learning_context)} past decisions")
            
            # Demonstrate context retrieval for different agent
            print("\nGetting context for design agent...")
            design_context = await context_api.get_context_for_agent(
                agent_id="design_agent_001",
                project_id="ecommerce_project_001",
                task_type="create_architecture"
            )
            print(f"Design agent context includes project info and related requirements")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            await knowledge_graph.close()
            await redis_client.close()
    
    # Run the example
    asyncio.run(main())

# Integration with FastAPI for production use
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class ProjectCreate(BaseModel):
    name: str
    description: str
    requirements: Dict[str, Any]
    technology_stack: List[str]
    timeline: Dict[str, Any]
    stakeholders: List[str]

class ContextRequest(BaseModel):
    agent_id: str
    project_id: str
    task_type: str

class ResultUpdate(BaseModel):
    agent_id: str
    project_id: str
    result: Dict[str, Any]

def create_context_management_app(context_api: ContextAPI) -> FastAPI:
    """Create FastAPI app for context management"""
    app = FastAPI(title="Context Management API")
    
    @app.post("/projects")
    async def create_project(project: ProjectCreate):
        """Create new project context"""
        try:
            project_data = project.dict()
            project_data["id"] = str(uuid.uuid4())
            result = await context_api.create_project(project_data)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/context")
    async def get_agent_context(request: ContextRequest):
        """Get context for agent"""
        try:
            context = await context_api.get_context_for_agent(
                request.agent_id,
                request.project_id,
                request.task_type
            )
            return context
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/context/update")
    async def update_context(update: ResultUpdate):
        """Update context with result"""
        try:
            result = await context_api.update_with_result(
                update.agent_id,
                update.project_id,
                update.result
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/projects/{project_id}/context")
    async def get_project_context(project_id: str):
        """Get full project context"""
        try:
            # This would be implemented to return comprehensive project context
            return {"project_id": project_id, "status": "not_implemented"}
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    return app