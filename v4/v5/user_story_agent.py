import asyncio
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import json
import logging
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """State for the user story generation agent"""
    project_id: int
    requirements: str
    context: str
    user_prompt: str
    
    # Retrieved information
    relevant_documents: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_entities: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_relations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Generated content
    user_stories: List[Dict[str, Any]] = Field(default_factory=list)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent state
    current_step: str = "initialize"
    error: Optional[str] = None
    retry_count: int = 0

class UserStoryAgent:
    """LangGraph-based agent for generating user stories using RAG and knowledge graphs"""
    
    def __init__(self, rag_service, kg_service, llm_service):
        self.rag_service = rag_service
        self.kg_service = kg_service
        self.llm_service = llm_service
        self.graph = self._build_graph()
        
        # Prompts
        self.analysis_prompt = PromptTemplate(
            input_variables=["requirements", "context", "documents", "entities"],
            template="""
            Analyze the following requirements and context to understand what user stories need to be generated.
            
            Requirements: {requirements}
            Context: {context}
            
            Relevant Documents:
            {documents}
            
            Knowledge Entities:
            {entities}
            
            Identify:
            1. Key features and functionalities
            2. User personas and roles
            3. Business objectives
            4. Technical constraints
            5. Dependencies between features
            
            Return your analysis as a structured JSON object.
            """
        )
        
        self.generation_prompt = PromptTemplate(
            input_variables=["analysis", "requirements", "context", "user_prompt", "examples"],
            template="""
            Based on the following analysis, generate comprehensive user stories.
            
            Analysis: {analysis}
            Requirements: {requirements}
            Context: {context}
            User Instructions: {user_prompt}
            
            Example User Stories:
            {examples}
            
            Generate user stories following this format:
            - Title: As a [user role], I want [functionality] so that [benefit]
            - Description: Detailed description of the feature
            - Acceptance Criteria: Clear, testable criteria
            - Priority: High/Medium/Low based on business value
            - Story Points: Estimated complexity (1, 2, 3, 5, 8, 13)
            - Epic: Group related stories under epics
            
            Return as a JSON array of user story objects.
            """
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("retrieve_knowledge", self._retrieve_knowledge)
        workflow.add_node("analyze_requirements", self._analyze_requirements)
        workflow.add_node("generate_stories", self._generate_stories)
        workflow.add_node("validate_stories", self._validate_stories)
        workflow.add_node("enrich_stories", self._enrich_stories)
        
        # Define the flow
        workflow.set_entry_point("retrieve_documents")
        
        workflow.add_edge("retrieve_documents", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "analyze_requirements")
        workflow.add_edge("analyze_requirements", "generate_stories")
        workflow.add_edge("generate_stories", "validate_stories")
        workflow.add_edge("validate_stories", "enrich_stories")
        workflow.add_edge("enrich_stories", END)
        
        return workflow.compile()
    
    async def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents using RAG"""
        try:
            logger.info(f"Retrieving documents for project {state.project_id}")
            
            # Combine requirements, context, and user prompt for search
            search_query = f"{state.requirements} {state.context} {state.user_prompt}"
            
            # Retrieve relevant documents
            documents = await self.rag_service.retrieve_documents(
                query=search_query,
                project_id=state.project_id,
                limit=10
            )
            
            state.relevant_documents = documents
            state.current_step = "retrieve_knowledge"
            
            logger.info(f"Retrieved {len(documents)} relevant documents")
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            state.error = f"Document retrieval failed: {str(e)}"
            return state
    
    async def _retrieve_knowledge(self, state: AgentState) -> AgentState:
        """Retrieve relevant knowledge entities and relations"""
        try:
            logger.info("Retrieving knowledge graph data")
            
            # Get entities related to the project
            entities = await self.kg_service.get_project_entities(state.project_id)
            
            # Get relations between entities
            relations = await self.kg_service.get_entity_relations(
                [entity['id'] for entity in entities]
            )
            
            state.knowledge_entities = entities
            state.knowledge_relations = relations
            state.current_step = "analyze_requirements"
            
            logger.info(f"Retrieved {len(entities)} entities and {len(relations)} relations")
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            state.error = f"Knowledge retrieval failed: {str(e)}"
            return state
    
    async def _analyze_requirements(self, state: AgentState) -> AgentState:
        """Analyze requirements using LLM"""
        try:
            logger.info("Analyzing requirements")
            
            # Format documents for prompt
            docs_text = "\n".join([
                f"- {doc['metadata'].get('filename', 'Unknown')}: {doc['content'][:500]}..."
                for doc in state.relevant_documents
            ])
            
            # Format entities for prompt
            entities_text = "\n".join([
                f"- {entity['name']} ({entity['type']}): {entity.get('description', '')}"
                for entity in state.knowledge_entities
            ])
            
            # Generate analysis
            analysis_response = await self.llm_service.generate(
                prompt=self.analysis_prompt.format(
                    requirements=state.requirements,
                    context=state.context,
                    documents=docs_text,
                    entities=entities_text
                ),
                temperature=0.3
            )
            
            # Parse analysis (assuming JSON response)
            try:
                analysis = json.loads(analysis_response)
            except json.JSONDecodeError:
                analysis = {"analysis": analysis_response}
            
            state.generation_metadata["analysis"] = analysis
            state.current_step = "generate_stories"
            
            logger.info("Requirements analysis completed")
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing requirements: {str(e)}")
            state.error = f"Requirements analysis failed: {str(e)}"
            return state
    
    async def _generate_stories(self, state: AgentState) -> AgentState:
        """Generate user stories using LLM"""
        try:
            logger.info("Generating user stories")
            
            # Get example stories for few-shot learning
            examples = await self._get_example_stories()
            
            # Generate stories
            stories_response = await self.llm_service.generate(
                prompt=self.generation_prompt.format(
                    analysis=json.dumps(state.generation_metadata.get("analysis", {})),
                    requirements=state.requirements,
                    context=state.context,
                    user_prompt=state.user_prompt,
                    examples=examples
                ),
                temperature=0.7
            )
            
            # Parse stories
            try:
                stories = json.loads(stories_response)
                if not isinstance(stories, list):
                    stories = [stories]
            except json.JSONDecodeError:
                # Fallback: create a single story from the response
                stories = [{
                    "title": "Generated User Story",
                    "description": stories_response,
                    "acceptance_criteria": "To be defined",
                    "priority": "Medium",
                    "story_points": 3
                }]
            
            state.user_stories = stories
            state.current_step = "validate_stories"
            
            logger.info(f"Generated {len(stories)} user stories")
            return state
            
        except Exception as e:
            logger.error(f"Error generating stories: {str(e)}")
            state.error = f"Story generation failed: {str(e)}"
            return state
    
    async def _validate_stories(self, state: AgentState) -> AgentState:
        """Validate and improve generated stories"""
        try:
            logger.info("Validating user stories")
            
            validated_stories = []
            
            for story in state.user_stories:
                # Basic validation
                if not story.get("title") or not story.get("description"):
                    continue
                
                # Ensure required fields
                validated_story = {
                    "title": story.get("title", ""),
                    "description": story.get("description", ""),
                    "acceptance_criteria": story.get("acceptance_criteria", "To be defined"),
                    "priority": story.get("priority", "Medium"),
                    "story_points": story.get("story_points", 3),
                    "epic": story.get("epic", ""),
                    "labels": story.get("labels", []),
                    "confidence_score": 0.8  # Default confidence
                }
                
                # Validate priority
                if validated_story["priority"] not in ["Low", "Medium", "High", "Critical"]:
                    validated_story["priority"] = "Medium"
                
                # Validate story points
                valid_points = [1, 2, 3, 5, 8, 13, 21]
                if validated_story["story_points"] not in valid_points:
                    validated_story["story_points"] = 3
                
                validated_stories.append(validated_story)
            
            state.user_stories = validated_stories
            state.current_step = "enrich_stories"
            
            logger.info(f"Validated {len(validated_stories)} user stories")
            return state
            
        except Exception as e:
            logger.error(f"Error validating stories: {str(e)}")
            state.error = f"Story validation failed: {str(e)}"
            return state
    
    async def _enrich_stories(self, state: AgentState) -> AgentState:
        """Enrich stories with additional metadata"""
        try:
            logger.info("Enriching user stories")
            
            enriched_stories = []
            
            for story in state.user_stories:
                # Add generation metadata
                story["generation_context"] = {
                    "source_documents": [doc["id"] for doc in state.relevant_documents],
                    "knowledge_entities": [entity["id"] for entity in state.knowledge_entities],
                    "generated_at": datetime.utcnow().isoformat(),
                    "requirements_hash": hash(state.requirements),
                    "model_used": self.llm_service.current_model
                }
                
                # Determine epic if not set
                if not story.get("epic"):
                    story["epic"] = await self._determine_epic(story, state.knowledge_entities)
                
                # Add dependency analysis
                story["dependencies"] = await self._analyze_dependencies(
                    story, state.user_stories, state.knowledge_relations
                )
                
                enriched_stories.append(story)
            
            state.user_stories = enriched_stories
            state.current_step = "completed"
            
            logger.info(f"Enriched {len(enriched_stories)} user stories")
            return state
            
        except Exception as e:
            logger.error(f"Error enriching stories: {str(e)}")
            state.error = f"Story enrichment failed: {str(e)}"
            return state
    
    async def _get_example_stories(self) -> str:
        """Get example user stories for few-shot learning"""
        examples = [
            {
                "title": "As a user, I want to log in to the system so that I can access my personalized dashboard",
                "description": "Users need to authenticate themselves to access the application features and see their personalized content.",
                "acceptance_criteria": [
                    "User can enter username and password",
                    "System validates credentials",
                    "User is redirected to dashboard on successful login",
                    "Error message shown for invalid credentials"
                ],
                "priority": "High",
                "story_points": 5,
                "epic": "User Authentication"
            },
            {
                "title": "As an admin, I want to manage user accounts so that I can control system access",
                "description": "Administrators need the ability to create, update, and deactivate user accounts to maintain system security and user management.",
                "acceptance_criteria": [
                    "Admin can create new user accounts",
                    "Admin can edit existing user details",
                    "Admin can deactivate/activate user accounts",
                    "Admin can assign roles to users"
                ],
                "priority": "Medium",
                "story_points": 8,
                "epic": "User Management"
            }
        ]
        
        return json.dumps(examples, indent=2)
    
    async def _determine_epic(self, story: Dict[str, Any], entities: List[Dict[str, Any]]) -> str:
        """Determine epic for a story based on knowledge entities"""
        # Simple heuristic: use the most relevant feature entity
        story_text = f"{story['title']} {story['description']}".lower()
        
        for entity in entities:
            if entity.get("type") == "feature" and entity["name"].lower() in story_text:
                return entity["name"]
        
        return "General"
    
    async def _analyze_dependencies(
        self, 
        story: Dict[str, Any], 
        all_stories: List[Dict[str, Any]], 
        relations: List[Dict[str, Any]]
    ) -> List[str]:
        """Analyze dependencies between stories"""
        dependencies = []
        
        # Simple keyword-based dependency detection
        story_keywords = set(story["title"].lower().split() + story["description"].lower().split())
        
        for other_story in all_stories:
            if other_story == story:
                continue
                
            other_keywords = set(other_story["title"].lower().split() + other_story["description"].lower().split())
            
            # Check for common keywords indicating dependency
            common_keywords = story_keywords.intersection(other_keywords)
            dependency_indicators = {"login", "authentication", "user", "account", "permission"}
            
            if dependency_indicators.intersection(common_keywords):
                dependencies.append(other_story["title"])
        
        return dependencies[:3]  # Limit to top 3 dependencies
    
    async def generate_user_stories(
        self,
        project_id: int,
        requirements: str,
        context: str,
        user_prompt: str = ""
    ) -> List[Dict[str, Any]]:
        """Main method to generate user stories"""
        try:
            # Initialize state
            initial_state = AgentState(
                project_id=project_id,
                requirements=requirements,
                context=context,
                user_prompt=user_prompt
            )
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            if result.error:
                raise Exception(result.error)
            
            return result.user_stories
            
        except Exception as e:
            logger.error(f"Error in user story generation: {str(e)}")
            raise e