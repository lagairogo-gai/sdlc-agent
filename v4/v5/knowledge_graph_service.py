import asyncio
from typing import List, Dict, Any, Optional, Tuple
from neo4j import AsyncGraphDatabase
import logging
import json
import re
from datetime import datetime
from langchain.schema import Document

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """Knowledge Graph service using Neo4j for entity and relationship management"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password123"):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._initialize_constraints()
    
    async def _initialize_constraints(self):
        """Initialize Neo4j constraints and indexes"""
        async with self.driver.session() as session:
            try:
                # Create constraints for unique entities
                await session.run("""
                    CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                
                # Create indexes for performance
                await session.run("""
                    CREATE INDEX entity_type_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """)
                
                await session.run("""
                    CREATE INDEX entity_name_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """)
                
                logger.info("Knowledge graph constraints and indexes initialized")
                
            except Exception as e:
                logger.error(f"Error initializing constraints: {e}")
    
    async def add_document_knowledge(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Extract and add knowledge from document content"""
        try:
            # Extract entities and relationships
            entities = await self._extract_entities(content, metadata)
            relationships = await self._extract_relationships(content, entities)
            
            async with self.driver.session() as session:
                # Add entities
                for entity in entities:
                    await self._create_entity(session, entity, doc_id)
                
                # Add relationships
                for relationship in relationships:
                    await self._create_relationship(session, relationship, doc_id)
            
            logger.info(f"Added knowledge for document {doc_id}: {len(entities)} entities, {len(relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document knowledge: {e}")
            return False
    
    async def _extract_entities(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from document content using rule-based and pattern matching"""
        entities = []
        
        # Simple rule-based entity extraction (in production, use NER models)
        patterns = {
            'feature': r'\b(?:feature|functionality|capability|module|component)\s+([A-Z][a-zA-Z\s]+)',
            'user_role': r'\b(?:as\s+(?:a|an)\s+)([a-zA-Z\s]+?)(?:\s*,|\s+I\s+want)',
            'system': r'\b(?:system|application|platform|service|API)\s+([A-Z][a-zA-Z\s]+)',
            'requirement': r'\b(?:requirement|shall|must|should)\s+([a-zA-Z\s]+)',
            'technology': r'\b(?:using|with|via|through)\s+([A-Z][a-zA-Z0-9\s]+)',
        }
        
        entity_id_counter = 0
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(1).strip()
                if len(entity_name) > 2 and len(entity_name) < 50:  # Basic validation
                    entity_id_counter += 1
                    entities.append({
                        'id': f"{metadata.get('project_id', 'unknown')}_{entity_type}_{entity_id_counter}",
                        'name': entity_name,
                        'type': entity_type,
                        'description': f"Extracted from document content",
                        'source_position': match.start(),
                        'confidence': 0.7,
                        'properties': {
                            'project_id': metadata.get('project_id'),
                            'source': metadata.get('source', 'unknown'),
                            'extracted_at': datetime.utcnow().isoformat()
                        }
                    })
        
        # Add document as an entity
        entities.append({
            'id': f"doc_{metadata.get('project_id', 'unknown')}_{doc_id}",
            'name': metadata.get('filename', f"Document {doc_id}"),
            'type': 'document',
            'description': content[:200] + "..." if len(content) > 200 else content,
            'confidence': 1.0,
            'properties': {
                **metadata,
                'doc_id': doc_id,
                'content_length': len(content),
                'created_at': datetime.utcnow().isoformat()
            }
        })
        
        return entities
    
    async def _extract_relationships(self, content: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Simple relationship patterns
        relationship_patterns = [
            (r'(\w+)\s+depends\s+on\s+(\w+)', 'DEPENDS_ON'),
            (r'(\w+)\s+implements\s+(\w+)', 'IMPLEMENTS'),
            (r'(\w+)\s+uses\s+(\w+)', 'USES'),
            (r'(\w+)\s+contains\s+(\w+)', 'CONTAINS'),
            (r'(\w+)\s+relates\s+to\s+(\w+)', 'RELATES_TO'),
        ]
        
        # Cross-reference entities for relationships
        entity_names = [entity['name'].lower() for entity in entities]
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # Check for co-occurrence in content
                    if (entity1['name'].lower() in content.lower() and 
                        entity2['name'].lower() in content.lower()):
                        
                        # Simple heuristic: entities mentioned in the same sentence are related
                        sentences = content.split('.')
                        for sentence in sentences:
                            if (entity1['name'].lower() in sentence.lower() and 
                                entity2['name'].lower() in sentence.lower()):
                                
                                relationships.append({
                                    'from_entity_id': entity1['id'],
                                    'to_entity_id': entity2['id'],
                                    'relationship_type': 'MENTIONED_WITH',
                                    'confidence': 0.6,
                                    'properties': {
                                        'context': sentence.strip(),
                                        'extracted_at': datetime.utcnow().isoformat()
                                    }
                                })
                                break  # Only one relationship per entity pair
        
        return relationships
    
    async def _create_entity(self, session, entity: Dict[str, Any], doc_id: str):
        """Create or update an entity in the knowledge graph"""
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.name = $name,
            e.type = $type,
            e.description = $description,
            e.confidence = $confidence,
            e.properties = $properties,
            e.updated_at = datetime()
        WITH e
        MATCH (d:Document {id: $doc_id})
        MERGE (e)-[:EXTRACTED_FROM]->(d)
        """
        
        await session.run(query, 
            entity_id=entity['id'],
            name=entity['name'],
            type=entity['type'],
            description=entity['description'],
            confidence=entity['confidence'],
            properties=json.dumps(entity.get('properties', {})),
            doc_id=doc_id
        )
    
    async def _create_relationship(self, session, relationship: Dict[str, Any], doc_id: str):
        """Create a relationship between entities"""
        query = f"""
        MATCH (e1:Entity {{id: $from_id}})
        MATCH (e2:Entity {{id: $to_id}})
        MERGE (e1)-[r:{relationship['relationship_type']}]->(e2)
        SET r.confidence = $confidence,
            r.properties = $properties,
            r.source_doc = $doc_id,
            r.created_at = datetime()
        """
        
        await session.run(query,
            from_id=relationship['from_entity_id'],
            to_id=relationship['to_entity_id'],
            confidence=relationship['confidence'],
            properties=json.dumps(relationship.get('properties', {})),
            doc_id=doc_id
        )
    
    async def get_project_entities(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all entities for a project"""
        async with self.driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.properties CONTAINS $project_filter
            RETURN e.id as id, e.name as name, e.type as type, 
                   e.description as description, e.confidence as confidence,
                   e.properties as properties
            ORDER BY e.confidence DESC, e.name
            """
            
            result = await session.run(query, project_filter=f'"project_id":{project_id}')
            entities = []
            async for record in result:
                entities.append({
                    'id': record['id'],
                    'name': record['name'],
                    'type': record['type'],
                    'description': record['description'],
                    'confidence': record['confidence'],
                    'properties': json.loads(record['properties']) if record['properties'] else {}
                })
            
            return entities
    
    async def get_entity_relations(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Get relationships between specified entities"""
        async with self.driver.session() as session:
            query = """
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
            RETURN e1.id as from_id, e2.id as to_id, type(r) as relationship_type,
                   r.confidence as confidence, r.properties as properties
            """
            
            result = await session.run(query, entity_ids=entity_ids)
            relationships = []
            async for record in result:
                relationships.append({
                    'from_id': record['from_id'],
                    'to_id': record['to_id'],
                    'type': record['relationship_type'],
                    'confidence': record['confidence'],
                    'properties': json.loads(record['properties']) if record['properties'] else {}
                })
            
            return relationships
    
    async def get_project_insights(self, project_id: int) -> Dict[str, Any]:
        """Generate insights and analytics for a project"""
        async with self.driver.session() as session:
            # Get entity type distribution
            type_query = """
            MATCH (e:Entity)
            WHERE e.properties CONTAINS $project_filter
            RETURN e.type as type, count(e) as count
            ORDER BY count DESC
            """
            
            type_result = await session.run(type_query, project_filter=f'"project_id":{project_id}')
            entity_types = {}
            async for record in type_result:
                entity_types[record['type']] = record['count']
            
            # Get relationship statistics
            rel_query = """
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE e1.properties CONTAINS $project_filter 
            AND e2.properties CONTAINS $project_filter
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
            """
            
            rel_result = await session.run(rel_query, project_filter=f'"project_id":{project_id}')
            relationship_types = {}
            async for record in rel_result:
                relationship_types[record['relationship_type']] = record['count']
            
            # Get most connected entities (hubs)
            hub_query = """
            MATCH (e:Entity)
            WHERE e.properties CONTAINS $project_filter
            OPTIONAL MATCH (e)-[r]-()
            RETURN e.name as name, e.type as type, count(r) as connections
            ORDER BY connections DESC
            LIMIT 10
            """
            
            hub_result = await session.run(hub_query, project_filter=f'"project_id":{project_id}')
            hubs = []
            async for record in hub_result:
                hubs.append({
                    'name': record['name'],
                    'type': record['type'],
                    'connections': record['connections']
                })
            
            return {
                'entity_types': entity_types,
                'relationship_types': relationship_types,
                'most_connected_entities': hubs,
                'total_entities': sum(entity_types.values()),
                'total_relationships': sum(relationship_types.values()),
                'generated_at': datetime.utcnow().isoformat()
            }
    
    async def find_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to a given entity within specified depth"""
        async with self.driver.session() as session:
            query = """
            MATCH path = (start:Entity {id: $entity_id})-[*1..%d]-(related:Entity)
            RETURN DISTINCT related.id as id, related.name as name, related.type as type,
                   length(path) as distance
            ORDER BY distance, related.name
            """ % max_depth
            
            result = await session.run(query, entity_id=entity_id)
            related_entities = []
            async for record in result:
                related_entities.append({
                    'id': record['id'],
                    'name': record['name'],
                    'type': record['type'],
                    'distance': record['distance']
                })
            
            return related_entities
    
    async def get_graph_visualization_data(self, project_id: int) -> Dict[str, Any]:
        """Get data formatted for graph visualization"""
        entities = await self.get_project_entities(project_id)
        entity_ids = [entity['id'] for entity in entities]
        relationships = await self.get_entity_relations(entity_ids)
        
        # Format for frontend visualization
        nodes = []
        for entity in entities:
            nodes.append({
                'id': entity['id'],
                'label': entity['name'],
                'type': entity['type'],
                'size': min(max(entity['confidence'] * 20, 10), 50),
                'color': self._get_node_color(entity['type'])
            })
        
        edges = []
        for rel in relationships:
            edges.append({
                'from': rel['from_id'],
                'to': rel['to_id'],
                'label': rel['type'],
                'weight': rel['confidence']
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }
    
    def _get_node_color(self, entity_type: str) -> str:
        """Get color for entity type"""
        color_map = {
            'feature': '#8B5CF6',      # Purple
            'user_role': '#06B6D4',    # Cyan
            'system': '#10B981',       # Green
            'requirement': '#F59E0B',  # Amber
            'technology': '#EF4444',   # Red
            'document': '#6B7280',     # Gray
            'default': '#9CA3AF'       # Light gray
        }
        return color_map.get(entity_type, color_map['default'])
    
    async def close(self):
        """Close the database connection"""
        await self.driver.close()