#!/usr/bin/env python3
"""
JIRA Ticket Creation Chatbot with Teams Integration and AI Enhancement

This module provides a comprehensive solution for creating JIRA tickets with optional
Microsoft Teams notifications, AI-powered description enhancement, and sprint support.

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
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import httpx
import pymsteams
from atlassian import Jira
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
    sprint_custom_field: Optional[str] = 'customfield_10020'
    team_custom_field: Optional[str] = 'customfield_10010'
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
    assignee: Optional[str] = Field(None, description="Assignee account ID")
    reporter: Optional[str] = Field(None, description="Reporter account ID")
    labels: Optional[List[str]] = Field(default_factory=list, description="Issue labels")
    components: Optional[List[str]] = Field(default_factory=list, description="Project components")
    fix_versions: Optional[List[str]] = Field(default_factory=list, description="Fix versions")
    team: Optional[str] = Field(None, description="Team ID for team-managed projects (string ID)")
    custom_fields: Optional[Dict[str, Union[str, int, List]]] = Field(default_factory=dict, description="Custom fields")

    # Sprint options
    assign_to_current_sprint: bool = Field(False, description="Assign to current active sprint")

    # AI options
    enhance_with_ai: bool = Field(False, description="Enable AI description enhancement")
    ai_agent: AIAgent = Field(AIAgent.CLAUDE, description="AI agent for enhancement")

    # Notification options
    notify_teams: bool = Field(False, description="Send Teams notification")

    @field_validator('labels')
    def validate_labels(cls, v):
        """Validate and clean labels."""
        if v:
            return [label.strip() for label in v if label.strip()]
        return []

    @field_validator('summary')
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

        # Initialize AI clients if available
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
        Enhance ticket description using selected AI agent.

        Args:
            ticket_data: Original ticket data
            agent: AI agent to use

        Returns:
            Dictionary containing enhanced description and suggestions
        """
        if agent == AIAgent.NONE or (agent == AIAgent.CLAUDE and not self.anthropic_client) or (agent == AIAgent.OPENAI and not self.openai_client):
            logger.info(f"AI enhancement skipped or not available for agent: {agent}")
            return {
                "enhanced_description": ticket_data.description or "",
                "suggestions": []
            }

        prompt = self._build_enhancement_prompt(ticket_data)

        try:
            if agent == AIAgent.CLAUDE:
                return await self._enhance_with_claude(prompt)
            elif agent == AIAgent.OPENAI:
                return await self._enhance_with_openai(prompt)
        except Exception as e:
            logger.error(f"AI enhancement failed for {agent}: {str(e)}")
            return {
                "enhanced_description": ticket_data.description or "",
                "suggestions": [f"AI enhancement failed: {str(e)}"]
            }

    def _build_enhancement_prompt(self, ticket_data: JiraTicketRequest) -> str:
        """Build prompt for AI description enhancement."""
        return f"""
You are a professional JIRA ticket writer. Enhance the following ticket information:

**Ticket Information:**
- Summary: {ticket_data.summary}
- Issue Type: {ticket_data.issue_type}
- Priority: {ticket_data.priority}
- Description: {ticket_data.description or 'Not provided'}
- Labels: {', '.join(ticket_data.labels) if ticket_data.labels else 'None'}
- Components: {', '.join(ticket_data.components) if ticket_data.components else 'None'}

**Requirements:**
1. Provide a clear, detailed, and professional description.
2. Include acceptance criteria if applicable.
3. Suggest any missing information or improvements.
4. Use proper JIRA formatting (e.g., bullet points, numbered lists, code blocks).
5. Keep it concise yet comprehensive.

Respond with a JSON object containing:
- "enhanced_description": The improved description (string)
- "suggestions": Array of suggestion strings
"""

    async def _enhance_with_claude(self, prompt: str) -> Dict[str, Union[str, List[str]]]:
        """Enhance description using Anthropic's Claude model."""
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        response_text = message.content[0].text
        return self._parse_ai_response(response_text)

    async def _enhance_with_openai(self, prompt: str) -> Dict[str, Union[str, List[str]]]:
        """Enhance description using OpenAI's GPT model."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
        )
        response_text = response.choices[0].message.content
        return self._parse_ai_response(response_text)

    def _parse_ai_response(self, response_text: str) -> Dict[str, Union[str, List[str]]]:
        """Parse AI response text to extract enhanced description and suggestions."""
        try:
            result = json.loads(response_text)
            return {
                "enhanced_description": result.get("enhanced_description", ""),
                "suggestions": result.get("suggestions", [])
            }
        except json.JSONDecodeError:
            logger.warning("AI response not valid JSON, using raw text as description")
            return {
                "enhanced_description": response_text.strip(),
                "suggestions": ["Response was not in expected JSON format"]
            }


class TeamsNotifier:
    """Handles notifications to Microsoft Teams."""

    def __init__(self, config: TeamsConfig):
        """
        Initialize Teams notifier with configuration.

        Args:
            config: Teams configuration settings
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
        Send notification about new ticket to Teams channel.

        Args:
            ticket_key: JIRA ticket key
            ticket_url: Full URL to the ticket
            ticket_data: Ticket creation data

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.config.enabled or not self.webhook_url:
            logger.info("Teams notifications disabled or webhook not configured")
            return False

        try:
            teams_message = pymsteams.connectorcard(self.webhook_url)
            teams_message.title(f"New JIRA Ticket: {ticket_key}")
            teams_message.summary(f"Ticket {ticket_key} created in {ticket_data.project}")

            color_map = {
                Priority.HIGHEST: "FF0000",
                Priority.HIGH: "FF8C00",
                Priority.MEDIUM: "FFD700",
                Priority.LOW: "32CD32",
                Priority.LOWEST: "87CEEB"
            }
            teams_message.color(color_map.get(ticket_data.priority, "FFD700"))

            card_section = pymsteams.cardsection()
            card_section.activityTitle(ticket_data.summary)
            card_section.activitySubtitle(f"{ticket_data.issue_type.value} | Priority: {ticket_data.priority.value}")

            if ticket_data.description:
                desc_preview = ticket_data.description[:200] + "..." if len(ticket_data.description) > 200 else ticket_data.description
                card_section.activityText(desc_preview)

            facts = []
            if ticket_data.assignee:
                facts.append(("Assignee", ticket_data.assignee))
            if ticket_data.reporter:
                facts.append(("Reporter", ticket_data.reporter))
            if ticket_data.labels:
                facts.append(("Labels", ", ".join(ticket_data.labels)))
            if ticket_data.components:
                facts.append(("Components", ", ".join(ticket_data.components)))

            for name, value in facts:
                card_section.addFact(name, value)

            teams_message.addSection(card_section)
            teams_message.addLinkButton("View in JIRA", ticket_url)

            await asyncio.get_event_loop().run_in_executor(None, teams_message.send)
            logger.info(f"Teams notification sent for {ticket_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Teams notification for {ticket_key}: {str(e)}")
            return False


class JiraChatbot:
    """
    Main class for JIRA ticket creation with enhancements.
    """

    def __init__(
        self,
        jira_config: JiraConfig,
        teams_config: Optional[TeamsConfig] = None,
        ai_config: Optional[AIConfig] = None
    ):
        """
        Initialize the JIRA Chatbot.

        Args:
            jira_config: JIRA connection settings
            teams_config: Teams notification settings (optional)
            ai_config: AI enhancement settings (optional)
        """
        self.jira_config = jira_config
        self.teams_config = teams_config or TeamsConfig()
        self.ai_config = ai_config or AIConfig()

        self.jira = Jira(
            url=jira_config.url,
            username=jira_config.username,
            password=jira_config.api_token,
            cloud=True
        )

        self.ai_enhancer = AIDescriptionEnhancer(self.ai_config)
        self.teams_notifier = TeamsNotifier(self.teams_config)

        self.executor = ThreadPoolExecutor(max_workers=5)

        # Fetch available priorities
        self.priority_map = self._get_priority_map()

        logger.info("JIRA Chatbot initialized")

    def _get_priority_map(self) -> Dict[str, str]:
        """Fetch available priorities from JIRA and map enum values to API names."""
        try:
            priorities = self.jira.get_all_priorities()
            priority_map = {}
            for priority in priorities:
                api_name = priority['name'].lower()
                # Map known Priority enum values to JIRA priority names
                for enum_priority in Priority:
                    if enum_priority.value.lower() == api_name:
                        priority_map[enum_priority.value] = priority['name']
                        break
                else:
                    priority_map[priority['name']] = priority['name']
            logger.info(f"Available priorities: {list(priority_map.values())}")
            return priority_map
        except Exception as e:
            logger.error(f"Failed to fetch priorities: {str(e)}")
            return {p.value: p.value for p in Priority}  # Fallback to enum values

    async def create_ticket(self, ticket_request: JiraTicketRequest) -> JiraTicketResponse:
        """
        Create a new JIRA ticket with optional AI enhancement, sprint assignment, and Teams notification.

        Args:
            ticket_request: Ticket creation request data

        Returns:
            Response with ticket details or error information
        """
        errors = []
        try:
            await self._validate_ticket_request(ticket_request)

            enhanced_data = None
            if ticket_request.enhance_with_ai:
                enhanced_data = await self.ai_enhancer.enhance_description(
                    ticket_request,
                    ticket_request.ai_agent
                )
                if enhanced_data["enhanced_description"]:
                    ticket_request.description = enhanced_data["enhanced_description"]

            issue_data = self._prepare_issue_data(ticket_request)

            if ticket_request.assign_to_current_sprint:
                sprint_id = await self.get_active_sprint(ticket_request.project)
                if sprint_id:
                    sprint_field = self.jira_config.sprint_custom_field
                    issue_data[sprint_field] = [sprint_id]
                else:
                    errors.append("No active sprint found; ticket created without sprint assignment")
                    logger.warning(f"No active sprint for project {ticket_request.project}")

            loop = asyncio.get_event_loop()
            try:
                issue = await loop.run_in_executor(
                    self.executor,
                    lambda: self.jira.create_issue(fields=issue_data)
                )
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response.text:
                    try:
                        error_details = json.loads(e.response.text)
                        error_msg = error_details.get('errorMessages', [error_msg])
                    except json.JSONDecodeError:
                        pass
                errors.append(f"JIRA API error: {error_msg}")
                logger.error(f"JIRA API error: {error_msg}")
                raise

            ticket_key = issue.key
            ticket_url = f"{self.jira_config.url}/browse/{ticket_key}"

            teams_sent = False
            if ticket_request.notify_teams:
                teams_sent = await self.teams_notifier.send_ticket_notification(
                    ticket_key, ticket_url, ticket_request
                )

            return JiraTicketResponse(
                success=True,
                ticket_key=ticket_key,
                ticket_url=ticket_url,
                message="Ticket created successfully",
                enhanced_description=enhanced_data.get("enhanced_description") if enhanced_data else None,
                ai_suggestions=enhanced_data.get("suggestions") if enhanced_data else None,
                teams_notification_sent=teams_sent,
                errors=errors if errors else None
            )

        except ValueError as ve:
            errors.append(str(ve))
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")
            logger.error(f"Ticket creation failed: {str(e)}")

        return JiraTicketResponse(
            success=False,
            message="Ticket creation failed",
            errors=errors
        )

    async def _validate_ticket_request(self, ticket_request: JiraTicketRequest) -> None:
        """Validate ticket request against JIRA."""
        loop = asyncio.get_event_loop()
        project = await loop.run_in_executor(
            self.executor,
            lambda: self.jira.project(ticket_request.project)
        )
        if not project:
            raise ValueError(f"Project '{ticket_request.project}' does not exist")

        # Handle both dictionary and object responses for issue types
        if isinstance(project, dict):
            issue_types = [it.get('name') for it in project.get('issueTypes', [])]
        else:
            issue_types = [it.name for it in project.issueTypes]

        if not issue_types:
            raise ValueError(f"No issue types found for project '{ticket_request.project}'")

        if ticket_request.issue_type.value not in issue_types:
            raise ValueError(f"Issue type '{ticket_request.issue_type}' not available. Available: {', '.join(issue_types)}")

        # Validate priority
        priority_name = self.priority_map.get(ticket_request.priority.value, ticket_request.priority.value)
        available_priorities = list(self.priority_map.values())
        if priority_name not in available_priorities:
            raise ValueError(f"Priority '{priority_name}' not available. Available: {', '.join(available_priorities)}")

    def _prepare_issue_data(self, ticket_request: JiraTicketRequest) -> Dict:
        """Prepare data dictionary for JIRA issue creation."""
        issue_data = {
            'project': {'key': ticket_request.project},
            'summary': ticket_request.summary,
            'issuetype': {'name': ticket_request.issue_type.value},
            'priority': {'name': self.priority_map.get(ticket_request.priority.value, ticket_request.priority.value)}
        }

        if ticket_request.description:
            issue_data['description'] = ticket_request.description
        if ticket_request.assignee:
            issue_data['assignee'] = {'accountId': ticket_request.assignee}
        if ticket_request.reporter:
            issue_data['reporter'] = {'accountId': ticket_request.reporter}
        if ticket_request.labels:
            issue_data['labels'] = ticket_request.labels
        if ticket_request.components:
            issue_data['components'] = [{'name': comp} for comp in ticket_request.components]
        if ticket_request.fix_versions:
            issue_data['fixVersions'] = [{'name': ver} for ver in ticket_request.fix_versions]
        if ticket_request.team:
            issue_data[self.jira_config.team_custom_field] = ticket_request.team
        if ticket_request.custom_fields:
            issue_data.update(ticket_request.custom_fields)

        return issue_data

    async def get_active_sprint(self, project_key: str) -> Optional[int]:
        """
        Get the ID of the active sprint for the project.

        Assumes the first board and first active sprint.

        Args:
            project_key: JIRA project key

        Returns:
            Active sprint ID or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            boards = await loop.run_in_executor(
                self.executor,
                lambda: self.jira.get_all_agile_boards(projectKey=project_key)
            )
            if not boards.get('values'):
                return None

            board_id = boards['values'][0]['id']

            sprints = await loop.run_in_executor(
                self.executor,
                lambda: self.jira.get_all_sprints(board_id, state='active')
            )
            if not sprints.get('values'):
                return None

            return sprints['values'][0]['id']

        except Exception as e:
            logger.error(f"Failed to get active sprint for {project_key}: {str(e)}")
            return None

    async def get_project_metadata(self, project_key: str) -> Dict:
        """
        Retrieve metadata for a JIRA project including current sprint.

        Args:
            project_key: JIRA project key

        Returns:
            Dictionary with project metadata
        """
        try:
            loop = asyncio.get_event_loop()
            project = await loop.run_in_executor(
                self.executor,
                lambda: self.jira.project(project_key)
            )

            # Handle dictionary or object response
            if isinstance(project, dict):
                metadata = {
                    'key': project.get('key', ''),
                    'name': project.get('name', ''),
                    'description': project.get('description', ''),
                    'issue_types': [it.get('name') for it in project.get('issueTypes', [])],
                    'components': [comp.get('name') for comp in project.get('components', [])],
                    'versions': [ver.get('name') for ver in project.get('versions', [])]
                }
            else:
                metadata = {
                    'key': project.key,
                    'name': project.name,
                    'description': getattr(project, 'description', ''),
                    'issue_types': [it.name for it in project.issueTypes],
                    'components': [comp.name for comp in project.components] if hasattr(project, 'components') else [],
                    'versions': [ver.name for ver in project.versions] if hasattr(project, 'versions') else []
                }

            sprint_id = await self.get_active_sprint(project_key)
            if sprint_id:
                sprint = await loop.run_in_executor(
                    self.executor,
                    lambda: self.jira.get_sprint(sprint_id)
                )
                metadata['current_sprint'] = {
                    'id': sprint['id'],
                    'name': sprint['name'],
                    'state': sprint['state']
                }
            else:
                metadata['current_sprint'] = None

            return metadata

        except Exception as e:
            logger.error(f"Failed to get metadata for {project_key}: {str(e)}")
            raise ValueError(f"Failed to get project metadata: {str(e)}")

    async def health_check(self) -> Dict[str, Union[bool, str]]:
        """
        Check health of connected services.

        Returns:
            Dictionary with health statuses
        """
        status = {
            'jira': False,
            'teams': self.teams_config.enabled,
            'ai_claude': ANTHROPIC_AVAILABLE and bool(self.ai_config.anthropic_api_key),
            'ai_openai': OPENAI_AVAILABLE and bool(self.ai_config.openai_api_key),
            'overall': False
        }

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.jira.myself()
            )
            status['jira'] = True
        except Exception as e:
            logger.error(f"JIRA connection check failed: {str(e)}")

        if status['teams'] and self.teams_config.webhook_url:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.head(self.teams_config.webhook_url, timeout=5)
                    status['teams'] = resp.status_code < 400
            except Exception as e:
                status['teams'] = False
                logger.error(f"Teams webhook check failed: {str(e)}")

        status['overall'] = status['jira']

        return status


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    jira_config = JiraConfig(
        url=os.getenv('JIRA_URL', ''),
        username=os.getenv('JIRA_USERNAME', ''),
        api_token=os.getenv('JIRA_API_TOKEN', ''),
        sprint_custom_field=os.getenv('JIRA_SPRINT_CUSTOM_FIELD', 'customfield_10020'),
        team_custom_field=os.getenv('JIRA_TEAM_CUSTOM_FIELD', 'customfield_10010')
    )

    teams_config = TeamsConfig(
        webhook_url=os.getenv('TEAMS_WEBHOOK_URL'),
        enabled=bool(os.getenv('TEAMS_WEBHOOK_URL'))
    )

    ai_config = AIConfig(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        default_agent=AIAgent(os.getenv('DEFAULT_AI_AGENT', 'none'))
    )

    if not (jira_config.url and jira_config.username and jira_config.api_token):
        raise ValueError("Missing required JIRA configuration variables")

    chatbot = JiraChatbot(jira_config, teams_config, ai_config)

    app = FastAPI(
        title="JIRA Ticket Creation API",
        description="API for creating JIRA tickets with AI enhancements and notifications",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/tickets", response_model=JiraTicketResponse)
    async def create_jira_ticket(ticket_request: JiraTicketRequest):
        """Create a new JIRA ticket."""
        return await chatbot.create_ticket(ticket_request)

    @app.get("/projects/{project_key}/metadata")
    async def get_project_metadata_endpoint(project_key: str):
        """Get metadata for a specific project."""
        try:
            metadata = await chatbot.get_project_metadata(project_key)
            return {"success": True, "data": metadata}
        except ValueError as ve:
            raise HTTPException(status_code=404, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/health")
    async def health_check_endpoint():
        """Check service health."""
        return await chatbot.health_check()

    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "name": "JIRA Ticket Creation API",
            "version": "1.0.0",
            "endpoints": [
                "/tickets (POST): Create ticket",
                "/projects/{project_key}/metadata (GET): Project metadata",
                "/health (GET): Health check"
            ]
        }

    return app

# For running directly if needed
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
