#!/usr/bin/env python3
"""
JIRA Ticket Creation Chatbot with Teams Integration and AI Enhancement

This module provides a comprehensive solution for creating JIRA tickets with optional
Microsoft Teams notifications and AI-powered description enhancement.

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass, asdict
from enum import Enum

# Third-party imports
import httpx
import pymsteams
from atlassian import Jira
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from dotenv import load_dotenv

# AI Libraries (conditional imports)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jira_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIAgent(str, Enum):
    """Supported AI agents for description enhancement."""
    CLAUDE = "claude"
    OPENAI = "openai"
    NONE = "none"


class IssueType(str, Enum):
    """JIRA issue types."""
    BUG = "Bug"
    TASK = "Task"
    STORY = "Story"
    EPIC = "Epic"
    SUBTASK = "Sub-task"
    IMPROVEMENT = "Improvement"


class Priority(str, Enum):
    """JIRA priority levels."""
    HIGHEST = "Highest"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    LOWEST = "Lowest"


@dataclass
class JiraConfig:
    """JIRA configuration settings."""
    url: str
    username: str
    api_token: str
    timeout: int = 30


@dataclass
class TeamsConfig:
    """Microsoft Teams webhook configuration."""
    webhook_url: Optional[str] = None
    enabled: bool = False


@dataclass
class AIConfig:
    """AI service configuration."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_agent: AIAgent = AIAgent.NONE


class JiraTicketRequest(BaseModel):
    """Request model for JIRA ticket creation."""
    
    # Required fields
    summary: str = Field(..., min_length=1, max_length=255, description="Brief ticket summary")
    project: str = Field(..., min_length=1, description="JIRA project key")
    issue_type: IssueType = Field(..., description="Type of issue")
    
    # Optional fields
    description: Optional[str] = Field(None, description="Detailed description")
    priority: Priority = Field(Priority.MEDIUM, description="Issue priority")
    assignee: Optional[str] = Field(None, description="Assignee username")
    reporter: Optional[str] = Field(None, description="Reporter username")
    labels: Optional[List[str]] = Field(default_factory=list, description="Issue labels")
    components: Optional[List[str]] = Field(default_factory=list, description="Project components")
    fix_versions: Optional[List[str]] = Field(default_factory=list, description="Fix versions")
    custom_fields: Optional[Dict[str, str]] = Field(default_factory=dict, description="Custom fields")
    
    # Options
    enhance_with_ai: bool = Field(False, description="Enable AI description enhancement")
    ai_agent: AIAgent = Field(AIAgent.CLAUDE, description="AI agent for enhancement")
    notify_teams: bool = Field(False, description="Send Teams notification")
    
    @validator('labels')
    def validate_labels(cls, v):
        """Validate and clean labels."""
        if v:
            return [label.strip() for label in v if label.strip()]
        return []
    
    @validator('summary')
    def validate_summary(cls, v):
        """Validate and clean summary."""
        return v.strip()


class JiraTicketResponse(BaseModel):
    """Response model for JIRA ticket creation."""
    
    success: bool
    ticket_key: Optional[str] = None
    ticket_url: Optional[str] = None
    message: str
    enhanced_description: Optional[str] = None
    ai_suggestions: Optional[List[str]] = None
    teams_notification_sent: bool = False
    errors: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIDescriptionEnhancer:
    """Handles AI-powered description enhancement for JIRA tickets."""
    
    def __init__(self, config: AIConfig):
        """
        Initialize AI enhancer with configuration.
        
        Args:
            config: AI configuration settings
        """
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize AI clients
        if OPENAI_AVAILABLE and config.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        
        if ANTHROPIC_AVAILABLE and config.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    
    async def enhance_description(
        self, 
        ticket_data: JiraTicketRequest, 
        agent: AIAgent = AIAgent.CLAUDE
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Enhance ticket description using AI.
        
        Args:
            ticket_data: Original ticket data
            agent: AI agent to use
            
        Returns:
            Dictionary containing enhanced description and suggestions
        """
        try:
            if agent == AIAgent.NONE:
                return {
                    "enhanced_description": ticket_data.description or "",
                    "suggestions": []
                }
            
            prompt = self._build_enhancement_prompt(ticket_data)
            
            if agent == AIAgent.CLAUDE and self.anthropic_client:
                return await self._enhance_with_claude(prompt)
            elif agent == AIAgent.OPENAI and self.openai_client:
                return await self._enhance_with_openai(prompt)
            else:
                logger.warning(f"AI agent {agent} not available, returning original description")
                return {
                    "enhanced_description": ticket_data.description or "",
                    "suggestions": ["AI enhancement not available"]
                }
        
        except Exception as e:
            logger.error(f"AI enhancement failed: {str(e)}")
            return {
                "enhanced_description": ticket_data.description or "",
                "suggestions": [f"AI enhancement failed: {str(e)}"]
            }
    
    def _build_enhancement_prompt(self, ticket_data: JiraTicketRequest) -> str:
        """Build enhancement prompt for AI."""
        return f"""
You are a professional JIRA ticket writer. Please enhance the following ticket information:

**Ticket Information:**
- Summary: {ticket_data.summary}
- Issue Type: {ticket_data.issue_type}
- Priority: {ticket_data.priority}
- Description: {ticket_data.description or 'Not provided'}
- Labels: {', '.join(ticket_data.labels) if ticket_data.labels else 'None'}
- Components: {', '.join(ticket_data.components) if ticket_data.components else 'None'}

**Requirements:**
1. Provide a clear, detailed, and professional description
2. Include acceptance criteria if applicable
3. Suggest any missing information
4. Use proper formatting for JIRA (bullet points, numbered lists)
5. Keep it concise but comprehensive

Please respond with a JSON object containing:
- "enhanced_description": The improved description
- "suggestions": Array of suggestions for improvement
"""
    
    async def _enhance_with_claude(self, prompt: str) -> Dict[str, Union[str, List[str]]]:
        """Enhance description using Claude."""
        try:
            message = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            response_text = message.content[0].text
            # Try to parse as JSON, fallback to text
            try:
                result = json.loads(response_text)
                return {
                    "enhanced_description": result.get("enhanced_description", ""),
                    "suggestions": result.get("suggestions", [])
                }
            except json.JSONDecodeError:
                return {
                    "enhanced_description": response_text,
                    "suggestions": ["Response parsing completed successfully"]
                }
        
        except Exception as e:
            logger.error(f"Claude enhancement failed: {str(e)}")
            raise
    
    async def _enhance_with_openai(self, prompt: str) -> Dict[str, Union[str, List[str]]]:
        """Enhance description using OpenAI."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
            )
            
            response_text = response.choices[0].message.content
            # Try to parse as JSON, fallback to text
            try:
                result = json.loads(response_text)
                return {
                    "enhanced_description": result.get("enhanced_description", ""),
                    "suggestions": result.get("suggestions", [])
                }
            except json.JSONDecodeError:
                return {
                    "enhanced_description": response_text,
                    "suggestions": ["Response parsing completed successfully"]
                }
        
        except Exception as e:
            logger.error(f"OpenAI enhancement failed: {str(e)}")
            raise


class TeamsNotifier:
    """Handles Microsoft Teams notifications."""
    
    def __init__(self, config: TeamsConfig):
        """
        Initialize Teams notifier.
        
        Args:
            config: Teams configuration
        """
        self.config = config
        self.webhook_url = config.webhook_url
    
    async def send_ticket_notification(
        self, 
        ticket_key: str, 
        ticket_url: str, 
        ticket_data: JiraTicketRequest
    ) -> bool:
        """
        Send ticket creation notification to Teams.
        
        Args:
            ticket_key: JIRA ticket key
            ticket_url: JIRA ticket URL
            ticket_data: Original ticket data
            
        Returns:
            True if notification sent successfully
        """
        if not self.config.enabled or not self.webhook_url:
            return False
        
        try:
            # Create Teams message
            teams_message = pymsteams.connectorcard(self.webhook_url)
            
            # Set title and summary
            teams_message.title(f"New JIRA Ticket Created: {ticket_key}")
            teams_message.summary(f"JIRA ticket {ticket_key} has been created")
            
            # Set color based on priority
            color_map = {
                Priority.HIGHEST: "FF0000",  # Red
                Priority.HIGH: "FF8C00",     # Orange
                Priority.MEDIUM: "FFD700",   # Yellow
                Priority.LOW: "32CD32",      # Green
                Priority.LOWEST: "87CEEB"    # Light Blue
            }
            teams_message.color(color_map.get(ticket_data.priority, "FFD700"))
            
            # Add main card section
            card_section = pymsteams.cardsection()
            card_section.activityTitle(f"📋 {ticket_data.summary}")
            card_section.activitySubtitle(f"Type: {ticket_data.issue_type} | Priority: {ticket_data.priority}")
            
            if ticket_data.description:
                description = ticket_data.description
                if len(description) > 200:
                    description = description[:200] + "..."
                card_section.activityText(description)
            
            # Add facts
            facts = []
            if ticket_data.assignee:
                facts.append(("Assignee", ticket_data.assignee))
            if ticket_data.labels:
                facts.append(("Labels", ", ".join(ticket_data.labels)))
            if ticket_data.components:
                facts.append(("Components", ", ".join(ticket_data.components)))
            
            for name, value in facts:
                card_section.addFact(name, value)
            
            teams_message.addSection(card_section)
            
            # Add action button
            teams_message.addLinkButton("View Ticket", ticket_url)
            
            # Send message
            teams_message.send()
            logger.info(f"Teams notification sent for ticket {ticket_key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {str(e)}")
            return False


class JiraChatbot:
    """
    Professional JIRA Ticket Creation Chatbot.
    
    This class provides a comprehensive solution for creating JIRA tickets with
    optional AI enhancement and Teams notifications.
    """
    
    def __init__(
        self,
        jira_config: JiraConfig,
        teams_config: TeamsConfig = None,
        ai_config: AIConfig = None
    ):
        """
        Initialize the JIRA Chatbot.
        
        Args:
            jira_config: JIRA connection configuration
            teams_config: Teams notification configuration (optional)
            ai_config: AI enhancement configuration (optional)
        """
        self.jira_config = jira_config
        self.teams_config = teams_config or TeamsConfig()
        self.ai_config = ai_config or AIConfig()
        
        # Initialize JIRA client
        self.jira = Jira(
            url=jira_config.url,
            username=jira_config.username,
            password=jira_config.api_token,
            cloud=True
        )
        
        # Initialize optional components
        self.ai_enhancer = AIDescriptionEnhancer(self.ai_config)
        self.teams_notifier = TeamsNotifier(self.teams_config)
        
        logger.info("JIRA Chatbot initialized successfully")
    
    async def create_ticket(self, ticket_request: JiraTicketRequest) -> JiraTicketResponse:
        """
        Create a JIRA ticket with optional enhancements.
        
        Args:
            ticket_request: Ticket creation request
            
        Returns:
            JiraTicketResponse with creation results
        """
        try:
            logger.info(f"Creating JIRA ticket: {ticket_request.summary}")
            
            # Validate project and issue type
            await self._validate_ticket_request(ticket_request)
            
            # Enhance description with AI if requested
            enhanced_data = None
            if ticket_request.enhance_with_ai:
                enhanced_data = await self.ai_enhancer.enhance_description(
                    ticket_request, ticket_request.ai_agent
                )
                if enhanced_data.get("enhanced_description"):
                    ticket_request.description = enhanced_data["enhanced_description"]
            
            # Prepare issue data
            issue_data = self._prepare_issue_data(ticket_request)
            
            # Create JIRA issue
            issue = self.jira.create_issue(fields=issue_data)
            ticket_key = issue.key
            ticket_url = f"{self.jira_config.url}/browse/{ticket_key}"
            
            logger.info(f"JIRA ticket created successfully: {ticket_key}")
            
            # Send Teams notification if requested
            teams_sent = False
            if ticket_request.notify_teams:
                teams_sent = await self.teams_notifier.send_ticket_notification(
                    ticket_key, ticket_url, ticket_request
                )
            
            # Prepare response
            response = JiraTicketResponse(
                success=True,
                ticket_key=ticket_key,
                ticket_url=ticket_url,
                message=f"JIRA ticket {ticket_key} created successfully",
                enhanced_description=enhanced_data.get("enhanced_description") if enhanced_data else None,
                ai_suggestions=enhanced_data.get("suggestions", []) if enhanced_data else None,
                teams_notification_sent=teams_sent
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Failed to create JIRA ticket: {str(e)}")
            return JiraTicketResponse(
                success=False,
                message="Failed to create JIRA ticket",
                errors=[str(e)]
            )
    
    async def _validate_ticket_request(self, ticket_request: JiraTicketRequest) -> None:
        """
        Validate ticket request against JIRA configuration.
        
        Args:
            ticket_request: Ticket request to validate
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate project exists
            project = self.jira.project(ticket_request.project)
            if not project:
                raise ValueError(f"Project '{ticket_request.project}' not found")
            
            # Validate issue type exists in project
            issue_types = [it.name for it in project.issueTypes]
            if ticket_request.issue_type not in issue_types:
                raise ValueError(
                    f"Issue type '{ticket_request.issue_type}' not available in project. "
                    f"Available types: {', '.join(issue_types)}"
                )
            
            logger.info(f"Ticket request validated successfully for project {ticket_request.project}")
        
        except Exception as e:
            logger.error(f"Ticket validation failed: {str(e)}")
            raise ValueError(f"Ticket validation failed: {str(e)}")
    
    def _prepare_issue_data(self, ticket_request: JiraTicketRequest) -> Dict:
        """
        Prepare JIRA issue data from ticket request.
        
        Args:
            ticket_request: Ticket request data
            
        Returns:
            Dictionary formatted for JIRA API
        """
        issue_data = {
            'project': {'key': ticket_request.project},
            'summary': ticket_request.summary,
            'issuetype': {'name': ticket_request.issue_type},
            'priority': {'name': ticket_request.priority}
        }
        
        # Add optional fields
        if ticket_request.description:
            issue_data['description'] = ticket_request.description
        
        if ticket_request.assignee:
            issue_data['assignee'] = {'name': ticket_request.assignee}
        
        if ticket_request.reporter:
            issue_data['reporter'] = {'name': ticket_request.reporter}
        
        if ticket_request.labels:
            issue_data['labels'] = ticket_request.labels
        
        if ticket_request.components:
            issue_data['components'] = [{'name': comp} for comp in ticket_request.components]
        
        if ticket_request.fix_versions:
            issue_data['fixVersions'] = [{'name': ver} for ver in ticket_request.fix_versions]
        
        # Add custom fields
        if ticket_request.custom_fields:
            issue_data.update(ticket_request.custom_fields)
        
        return issue_data
    
    async def get_project_metadata(self, project_key: str) -> Dict:
        """
        Get project metadata including available issue types and components.
        
        Args:
            project_key: JIRA project key
            
        Returns:
            Dictionary containing project metadata
        """
        try:
            project = self.jira.project(project_key)
            
            metadata = {
                'key': project.key,
                'name': project.name,
                'description': getattr(project, 'description', ''),
                'issue_types': [it.name for it in project.issueTypes],
                'components': [comp.name for comp in project.components],
                'versions': [ver.name for ver in project.versions]
            }
            
            return metadata
        
        except Exception as e:
            logger.error(f"Failed to get project metadata: {str(e)}")
            raise ValueError(f"Failed to get project metadata: {str(e)}")
    
    async def health_check(self) -> Dict[str, Union[bool, str]]:
        """
        Perform health check on all services.
        
        Returns:
            Dictionary containing health status
        """
        status = {
            'jira': False,
            'teams': False,
            'ai_claude': False,
            'ai_openai': False,
            'overall': False
        }
        
        # Check JIRA connection
        try:
            self.jira.myself()
            status['jira'] = True
        except Exception as e:
            logger.error(f"JIRA health check failed: {str(e)}")
        
        # Check Teams webhook
        if self.teams_config.enabled and self.teams_config.webhook_url:
            try:
                # Simple webhook test (just check if URL is accessible)
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.teams_config.webhook_url.split('/')[2])
                    status['teams'] = response.status_code < 500
            except Exception as e:
                logger.error(f"Teams health check failed: {str(e)}")
        
        # Check AI services
        if self.ai_enhancer.anthropic_client:
            status['ai_claude'] = True
        
        if self.ai_enhancer.openai_client:
            status['ai_openai'] = True
        
        status['overall'] = status['jira']  # JIRA is the core requirement
        
        return status


# FastAPI Application Setup
def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Load configuration from environment
    jira_config = JiraConfig(
        url=os.getenv('JIRA_URL', ''),
        username=os.getenv('JIRA_USERNAME', ''),
        api_token=os.getenv('JIRA_API_TOKEN', '')
    )
    
    teams_config = TeamsConfig(
        webhook_url=os.getenv('TEAMS_WEBHOOK_URL'),
        enabled=bool(os.getenv('TEAMS_WEBHOOK_URL'))
    )
    
    ai_config = AIConfig(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        default_agent=AIAgent(os.getenv('DEFAULT_AI_AGENT', 'claude'))
    )
    
    # Validate required configuration
    if not all([jira_config.url, jira_config.username, jira_config.api_token]):
        raise ValueError("Missing required JIRA configuration")
    
    # Initialize chatbot
    chatbot = JiraChatbot(jira_config, teams_config, ai_config)
    
    # Create FastAPI app
    app = FastAPI(
        title="JIRA Ticket Creation Chatbot",
        description="Professional JIRA ticket creation with AI enhancement and Teams notifications",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # API Routes
    @app.post("/tickets", response_model=JiraTicketResponse)
    async def create_ticket(
        ticket_request: JiraTicketRequest,
        background_tasks: BackgroundTasks
    ):
        """Create a new JIRA ticket with optional enhancements."""
        response = await chatbot.create_ticket(ticket_request)
        return response
    
    @app.get("/projects/{project_key}/metadata")
    async def get_project_metadata(project_key: str):
        """Get project metadata including available issue types and components."""
        try:
            metadata = await chatbot.get_project_metadata(project_key)
            return {"success": True, "data": metadata}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """Perform health check on all services."""
        status = await chatbot.health_check()
        return status
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "JIRA Ticket Creation Chatbot",
            "version": "1.0.0",
            "description": "Professional JIRA ticket creation with AI enhancement and Teams notifications",
            "endpoints": {
                "create_ticket": "/tickets",
                "project_metadata": "/projects/{project_key}/metadata",
                "health_check": "/health"
            }
        }
    
    return app


# CLI Interface
def jira_main():
    """Main CLI interface for the JIRA chatbot."""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="JIRA Ticket Creation Chatbot")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--log-level', default='info', help='Log level')
    
    args = parser.parse_args()
    
    # Create and run the app
    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
