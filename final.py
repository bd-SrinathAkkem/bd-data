import os
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager
from datetime import datetime
import importlib
import uvicorn
from uuid import uuid4
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Body, Depends, Security, status, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pymsteams import connectorcard
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from jira import JIRA, JIRAError

# Structured logging configuration for consistent, JSON-formatted logs
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration management using Pydantic for environment variable handling
class Settings(BaseSettings):
    """
    Configuration settings loaded from environment variables or .env file.
    Provides default values and validation for required fields.
    """
    jira_server: str = Field(default="https://blackduck.atlassian.net", description="Jira server URL")
    jira_username: str = Field(..., description="Jira username for authentication")
    jira_api_token: str = Field(..., description="Jira API token for authentication")
    teams_webhook_url: Optional[str] = Field(None, description="Microsoft Teams webhook URL for notifications")
    api_key: str = Field(..., description="API key for securing endpoints")
    allowed_origins: str = Field(default="*", description="Comma-separated list of allowed CORS origins")
    ai_provider_default: str = Field(default="ollama", description="Default AI provider for ticket enhancement")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
    logger.info({
        "action": "load_settings",
        "api_key": "set" if settings.api_key else "missing",
        "jira_api_token": "set" if settings.jira_api_token else "missing",
        "teams_webhook": "set" if settings.teams_webhook_url else "missing"
    })
except Exception as e:
    logger.error({"action": "load_settings_failed", "error": str(e)})
    raise

# FastAPI application initialization
app = FastAPI(
    title="Jira Ticket Creator API",
    description="A robust API for creating Jira tickets with AI enhancement, sprint integration, and Microsoft Teams notifications.",
    version="2.4.0",
    openapi_tags=[
        {"name": "Tickets", "description": "Operations for creating and managing Jira tickets"},
        {"name": "Health", "description": "System health and connectivity checks"}
    ]
)

# CORS middleware for enabling cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Validates the API key provided in the request header.
    Raises HTTPException if the key is missing or invalid.
    """
    if not api_key:
        logger.error({"action": "api_key_validation_failed", "reason": "No API key provided"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No API key provided")
    if api_key != settings.api_key:
        logger.error({"action": "api_key_validation_failed", "provided_key": "hidden_for_security"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key

# Pydantic models for request validation
class AIConfig(BaseModel):
    """
    Configuration for AI providers used in ticket enhancement.
    """
    provider: str = Field(default=settings.ai_provider_default, description="AI provider (ollama, claude, custom)")
    api_url: Optional[str] = Field(None, description="API URL for the AI provider")
    api_key: Optional[str] = Field(None, description="API key for the AI provider")
    model: str = Field(default="llama2", description="Model name for the AI provider")

class TicketInput(BaseModel):
    """
    Input model for creating Jira tickets, with validation and example data.
    """
    summary: str = Field(..., min_length=1, max_length=255, description="Ticket summary (required, max 255 chars)")
    description: str = Field(..., min_length=1, description="Ticket description (required)")
    project: str = Field(..., min_length=1, description="Jira project key (required)")
    team_name: Optional[str] = Field(None, description="Team name for the ticket")
    issuetype: str = Field(default="Task", description="Issue type (e.g., Task, Bug)")
    priority: Optional[str] = Field(None, description="Priority (validated against project metadata)")
    labels: Optional[List[str]] = Field(None, description="List of labels for the ticket")
    assignee: Optional[str] = Field(None, description="Assignee username")
    attachment_urls: Optional[List[str]] = Field(None, description="List of URLs for attachments to add")
    board_id: Optional[int] = Field(None, description="Board ID for sprint assignment")
    other_fields: Optional[Dict[str, Any]] = Field(None, description="Additional custom fields")
    enable_ai: bool = Field(default=False, description="Enable AI enhancement for ticket data")
    ai_configs: Optional[List[AIConfig]] = Field(None, description="List of AI provider configurations")
    enable_teams_notification: bool = Field(default=False, description="Send Microsoft Teams notification")
    notification_template: Optional[str] = Field(None, description="Custom template for Teams notification")

    class Config:
        schema_extra = {
            "example": {
                "summary": "Fix login issue",
                "description": "Users cannot log in due to auth error.",
                "project": "PROJ",
                "issuetype": "Bug",
                "priority": "P1 Current priority",
                "labels": ["bug", "urgent"],
                "assignee": "username",
                "attachment_urls": ["https://example.com/image.png"],
                "board_id": 123,
                "team_name": "DevOps",
                "enable_ai": True,
                "ai_configs": [
                    {"provider": "ollama", "api_url": "http://localhost:11434", "model": "llama2"}
                ],
                "enable_teams_notification": True,
                "notification_template": "New ticket {ticket_key}: {summary}"
            }
        }

# Available Jira priorities (for reference, actual validation uses project metadata)
AVAILABLE_PRIORITIES = [
    'P0 - Urgent', 'P1 - Current priority', 'P2 - Top of the backlog', 'P3 - Negotiated or opportunistic', 
    'P4 - Negotiated or opportunistic', 'P0 - Fix Immediately!', 'P1 - High Priority', 'P2 - Medium Priority', 
    'P3 - Low Priority', 'P4 - Standard Ticket', 'Unprioritized', 'None', 'Blocker', 'Critical', 'Major', 
    'Normal', 'Minor', 'Trivial', 'Unassigned', 'P1 - Production Impact', 'P2 - High Priority', 
    'P3 - Standard Priority', 'P4 - Low Priority', 'P5 - Best Effort'
]

def text_to_adf(text: str) -> Dict[str, Any]:
    """
    Converts plain text to Atlassian Document Format (ADF), parsing for bullet points.
    
    Args:
        text: Plain text input, possibly containing bullet points with '- ' prefix.
    
    Returns:
        Dict representing the ADF structure.
    """
    content = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('- '):
            list_content = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                item_text = lines[i].strip()[2:].strip()
                list_content.append({
                    "type": "listItem",
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": item_text}]}]
                })
                i += 1
            content.append({"type": "bulletList", "content": list_content})
        else:
            para_lines = []
            while i < len(lines) and not lines[i].strip().startswith('- ') and lines[i].strip():
                para_lines.append(lines[i])
                i += 1
            para_text = '\n'.join(para_lines).strip()
            content.append({"type": "paragraph", "content": [{"type": "text", "text": para_text}]})
            if i < len(lines) and not lines[i].strip():
                i += 1
    if not content:
        content.append({"type": "paragraph", "content": [{"type": "text", "text": ""}]})
    return {"version": 1, "type": "doc", "content": content}

# AI provider registry for dynamic provider loading
class AIProviderRegistry:
    """Registry for AI providers, mapping provider names to client initialization functions."""
    providers: Dict[str, Callable] = {}

    @classmethod
    def register(cls, provider_name: str):
        def decorator(func: Callable):
            cls.providers[provider_name] = func
            return func
        return decorator

@AIProviderRegistry.register("ollama")
def get_ollama_client(config: AIConfig):
    """Initializes and returns an Ollama client."""
    try:
        ollama = importlib.import_module("ollama")
        if not config.api_url:
            raise ValueError("Ollama requires api_url")
        return ollama.AsyncClient(host=config.api_url)
    except ImportError:
        raise HTTPException(status_code=400, detail="Ollama library not installed")

@AIProviderRegistry.register("claude")
def get_claude_client(config: AIConfig):
    """Initializes and returns a Claude client."""
    try:
        anthropic = importlib.import_module("anthropic")
        if not config.api_key:
            raise ValueError("Claude requires api_key")
        return anthropic.AsyncAnthropic(api_key=config.api_key)
    except ImportError:
        raise HTTPException(status_code=400, detail="Anthropic library not installed")

@AIProviderRegistry.register("custom")
def get_custom_client(config: AIConfig):
    """Initializes and returns a custom AI client."""
    if not config.api_url:
        raise ValueError("Custom provider requires api_url")
    async def custom_call(prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
            payload = {"model": config.model, "prompt": prompt}
            response = await client.post(config.api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("response", "")
    return custom_call

# AI processing for ticket enhancement
class AIProcessor:
    """Handles AI-based enhancement of ticket data using configured providers."""
    def __init__(self, configs: List[AIConfig]):
        self.clients = []
        for config in configs:
            provider_func = AIProviderRegistry.providers.get(config.provider)
            if not provider_func:
                raise HTTPException(status_code=400, detail=f"Unsupported AI provider: {config.provider}")
            self.clients.append((config, provider_func(config)))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def enhance_ticket_data(self, ticket_data: Dict[str, Any], valid_priorities: List[str]) -> Dict[str, Any]:
        """
        Enhances ticket data using AI providers, ensuring concise summary and structured description.
        Suggests priority and labels if not provided.
        
        Args:
            ticket_data: Dictionary containing ticket fields.
            valid_priorities: List of valid priorities for suggestion.
        
        Returns:
            Enhanced ticket data with ADF-formatted description and suggested fields.
        """
        prompt = (
            f"Professionally enhance this Jira ticket data: Make the summary concise (under 255 characters), "
            f"structure the description with bullet points using - and \\n for line breaks, and add logical details. "
            f"The description must be a plain text string, not a JSON object or ADF format. "
            f"If priority is missing, suggest one from: {', '.join(valid_priorities)}. "
            f"If labels is missing, suggest a list of relevant labels. "
            f"Return valid JSON matching the original structure, adding suggested fields if missing: {json.dumps(ticket_data)}"
        )
        
        for config, client in self.clients:
            try:
                if config.provider == "ollama":
                    response = await client.generate(model=config.model, prompt=prompt)
                    enhanced_data = json.loads(response['response'])
                    enhanced_data['description'] = text_to_adf(enhanced_data['description'])
                    return enhanced_data
                elif config.provider == "claude":
                    message = f"{importlib.import_module('anthropic').HUMAN_PROMPT} {prompt} {importlib.import_module('anthropic').AI_PROMPT}"
                    response = await client.completions.create(
                        model=config.model,
                        max_tokens_to_sample=500,
                        prompt=message
                    )
                    enhanced_data = json.loads(response.completion)
                    enhanced_data['description'] = text_to_adf(enhanced_data['description'])
                    return enhanced_data
                elif config.provider == "custom":
                    response_text = await client(prompt)
                    enhanced_data = json.loads(response_text)
                    enhanced_data['description'] = text_to_adf(enhanced_data['description'])
                    return enhanced_data
            except Exception as e:
                logger.warning({"provider": config.provider, "error": str(e)})
                continue
        logger.error("All AI providers failed, using original data")
        ticket_data['description'] = text_to_adf(ticket_data['description'])
        return ticket_data

# Microsoft Teams notification with Adaptive Card
class TeamsNotifier:
    """Sends notifications to Microsoft Teams using Adaptive Cards for enhanced UI."""
    def __init__(self, webhook_url: str):
        if not webhook_url:
            raise ValueError("Teams webhook URL required")
        self.webhook_url = webhook_url

    async def send_notification(self, template: str, ticket_key: str, summary: str, ticket_url: str, priority: Optional[str], assignee: Optional[str]):
        """
        Sends a colorful, interactive Adaptive Card to Microsoft Teams.
        
        Args:
            template: Custom notification message template.
            ticket_key: Jira ticket key.
            summary: Ticket summary.
            ticket_url: URL to the Jira ticket.
            priority: Ticket priority.
            assignee: Ticket assignee.
        """
        try:
            message = template.format(ticket_key=ticket_key, summary=summary) if template else f"New Jira ticket {ticket_key}: {summary}"
            teams_message = connectorcard(self.webhook_url)
            
            # Enhanced Adaptive Card JSON structure for professional and colorful UI
            adaptive_card = {
                "type": "AdaptiveCard",
                "version": "1.5",
                "body": [
                    {
                        "type": "Container",
                        "style": "good",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "🎟️ New Jira Ticket Created",
                                "weight": "bolder",
                                "size": "large",
                                "color": "good"
                            },
                            {
                                "type": "TextBlock",
                                "text": message,
                                "wrap": True,
                                "size": "medium",
                                "color": "default"
                            }
                        ]
                    },
                    {
                        "type": "Container",
                        "style": "emphasis",
                        "items": [
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Ticket Key:", "value": ticket_key},
                                    {"title": "Summary:", "value": summary},
                                    {"title": "Priority:", "value": priority or "Not set"},
                                    {"title": "Assignee:", "value": assignee or "Unassigned"}
                                ]
                            }
                        ]
                    },
                    {
                        "type": "Container",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Quick Links",
                                "weight": "bolder",
                                "size": "medium",
                                "color": "accent"
                            }
                        ]
                    }
                ],
                "actions": [
                    {
                        "type": "Action.OpenUrl",
                        "title": "View Ticket in Jira",
                        "url": ticket_url,
                        "style": "positive"
                    },
                    {
                        "type": "Action.OpenUrl",
                        "title": "Jira Dashboard",
                        "url": settings.jira_server,
                        "style": "default"
                    },
                    {
                        "type": "Action.OpenUrl",
                        "title": "Contact Support Team",
                        "url": "https://support.example.com",
                        "style": "default"
                    }
                ],
                "msteams": {
                    "width": "Full"
                }
            }
            
            teams_message.payload = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": adaptive_card
                    }
                ]
            }
            teams_message.send()
            logger.info({"action": "teams_notification_sent", "ticket_key": ticket_key})
        except Exception as e:
            logger.error({"error": str(e), "type": "TeamsNotificationError"})
            raise HTTPException(status_code=500, detail="Failed to send Teams notification")

# Dependency for Teams notifier
def get_teams_notifier() -> Optional[TeamsNotifier]:
    """Returns a TeamsNotifier instance if a webhook URL is configured."""
    if settings.teams_webhook_url:
        return TeamsNotifier(settings.teams_webhook_url)
    return None

# API endpoints
@app.post("/create_ticket", response_model=Dict[str, Any], tags=["Tickets"])
async def create_ticket(
    request: Request,
    input_data: TicketInput = Body(...),
    notifier: Optional[TeamsNotifier] = Depends(get_teams_notifier),
    api_key: str = Depends(verify_api_key)
):
    """
    Creates a Jira ticket with AI enhancement, sprint integration, and Teams notification.
    Returns full ticket details including key, URL, and fields.
    
    Args:
        request: FastAPI request object for client information.
        input_data: Ticket input data validated by Pydantic.
        notifier: Optional Teams notifier dependency.
        api_key: Verified API key.
    
    Returns:
        Dictionary containing ticket key, URL, and full ticket details.
    
    Raises:
        HTTPException: For validation errors, authentication failures, or Jira errors.
    """
    logger.info({"action": "create_ticket_start", "client_ip": request.client.host, "timestamp": datetime.now().isoformat()})
    
    # Validate AI and Teams configuration
    if input_data.enable_ai and not input_data.ai_configs:
        raise HTTPException(status_code=400, detail="AI configs required if enable_ai is true")
    if input_data.enable_teams_notification and not notifier:
        raise HTTPException(status_code=400, detail="Teams webhook not configured")

    # Prepare ticket data
    ticket_data = input_data.dict(exclude={"enable_ai", "ai_configs", "enable_teams_notification", "notification_template", "board_id", "attachment_urls"}, exclude_none=True)

    # Initialize Jira client
    try:
        jira = JIRA(
            server=settings.jira_server,
            basic_auth=(settings.jira_username, settings.jira_api_token),
            options={"rest_api_version": "3"}
        )
    except JIRAError as e:
        logger.error({"error": str(e), "type": "JIRAConnectionError"})
        raise HTTPException(status_code=500, detail=f"Jira connection failed: {str(e)}")

    # Validate project and issue type
    try:
        meta = jira.createmeta(projectKeys=ticket_data["project"], expand="projects.issuetypes.fields")
        if not meta["projects"]:
            raise HTTPException(status_code=400, detail=f"Invalid project: {ticket_data['project']}")
        
        # Validate issue type
        valid_issuetypes = [it["name"] for it in meta["projects"][0]["issuetypes"]]
        selected_issuetype = ticket_data.get("issuetype", "Task")
        if selected_issuetype not in valid_issuetypes:
            logger.warning({
                "action": "invalid_issuetype",
                "provided": selected_issuetype,
                "defaulting_to": "Task",
                "valid_options": valid_issuetypes
            })
            ticket_data["issuetype"] = "Task"
            selected_issuetype = "Task"

        # Find the fields for the selected issue type
        allowed_fields = next(
            (it["fields"] for it in meta["projects"][0]["issuetypes"] if it["name"] == selected_issuetype),
            {}
        )
        valid_priorities = [p["name"] for p in allowed_fields.get("priority", {}).get("allowedValues", [])]
    except JIRAError as e:
        logger.error({"error": str(e), "type": "ValidationError"})
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

    # AI enhancement
    if input_data.enable_ai:
        processor = AIProcessor(input_data.ai_configs)
        ticket_data = await processor.enhance_ticket_data(ticket_data, valid_priorities)
    else:
        ticket_data['description'] = text_to_adf(ticket_data['description'])

    # Prepare fields for ticket creation
    fields = {
        "project": {"key": ticket_data["project"]},
        "summary": ticket_data["summary"],
        "description": ticket_data["description"],
        "issuetype": {"name": ticket_data.get("issuetype", "Task")}
    }

    # Add and validate priority
    priority = input_data.priority or ticket_data.get("priority")
    if priority:
        if "priority" not in allowed_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Priority field not available for project {ticket_data['project']} and issue type {selected_issuetype}"
            )
        if priority not in valid_priorities:
            raise HTTPException(
                status_code=400,
                detail=f"Priority '{priority}' not allowed for issue type '{selected_issuetype}' in project {ticket_data['project']}. Valid options: {valid_priorities}"
            )
        fields["priority"] = {"name": priority}

    # Add labels
    labels = input_data.labels or ticket_data.get("labels")
    if labels:
        if "labels" not in allowed_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Labels field not available for project {ticket_data['project']} and issue type {selected_issuetype}"
            )
        fields["labels"] = labels

    # Add assignee
    assignee = input_data.assignee or ticket_data.get("assignee")
    if assignee:
        if "assignee" not in allowed_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Assignee field not available for project {ticket_data['project']} and issue type {selected_issuetype}"
            )
        fields["assignee"] = {"name": assignee}

    # Add team name and other fields
    if input_data.team_name:
        if "customfield_10001" not in allowed_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Team field (customfield_10001) not available for project {ticket_data['project']} and issue type {selected_issuetype}"
            )
        fields["customfield_10001"] = input_data.team_name
    if input_data.other_fields:
        for field in input_data.other_fields:
            if field not in allowed_fields:
                raise HTTPException(status_code=400, detail=f"Invalid field: {field} for project {ticket_data['project']} and issue type {selected_issuetype}")
        fields.update(input_data.other_fields)

    # Create ticket
    try:
        issue = jira.create_issue(fields=fields)
        ticket_key = issue.key
        ticket_url = f"{settings.jira_server}/browse/{ticket_key}"
        logger.info({"action": "ticket_created", "key": ticket_key, "url": ticket_url})
    except JIRAError as e:
        logger.error({"error": str(e), "type": "TicketCreationError"})
        raise HTTPException(status_code=400, detail=f"Ticket creation failed: {str(e)}")

    # Add attachments from URLs
    if input_data.attachment_urls:
        async with httpx.AsyncClient() as client:
            for url in input_data.attachment_urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    filename = os.path.basename(urlparse(url).path) or "attachment"
                    jira.add_attachment(issue, attachment=resp.content, filename=filename)
                    logger.info({"action": "attachment_added", "ticket": ticket_key, "url": url})
                except Exception as e:
                    logger.warning({"action": "attachment_failed", "url": url, "error": str(e)})

    # Add to current sprint
    try:
        board_id = input_data.board_id
        if not board_id:
            boards = jira.boards(projectKeyOrId=ticket_data["project"])
            if boards:
                board_id = boards[0].id
                logger.info({"action": "board_selected", "board_id": board_id})
            else:
                logger.warning({"action": "no_board_found", "project": ticket_data["project"]})

        if board_id:
            active_sprints = jira.sprints(board_id, state='active')
            if active_sprints:
                sprint_id = active_sprints[0].id
                jira.add_issues_to_sprint(sprint_id, [ticket_key])
                logger.info({"action": "added_to_sprint", "sprint_id": sprint_id, "ticket": ticket_key})
            else:
                logger.warning({"action": "no_active_sprint", "board_id": board_id})
    except JIRAError as e:
        logger.warning({"error": str(e), "type": "SprintAdditionError"})

    # Send Teams notification
    if input_data.enable_teams_notification:
        await notifier.send_notification(
            input_data.notification_template,
            ticket_key,
            ticket_data["summary"],
            ticket_url,
            priority,
            assignee
        )

    # Return full ticket details including link
    return {
        "ticket_key": ticket_key,
        "ticket_url": ticket_url,
        "ticket_details": {
            "summary": issue.fields.summary,
            "description": issue.fields.description,
            "issuetype": issue.fields.issuetype.name,
            "project": issue.fields.project.key,
            "priority": issue.fields.priority.name if issue.fields.priority else None,
            "labels": issue.fields.labels,
            "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
            "created": issue.fields.created,
            "status": issue.fields.status.name
        },
        "message": "Ticket created successfully"
    }

@app.get("/project_metadata/{project_key}", response_model=Dict[str, Any], tags=["Tickets"])
async def get_project_metadata(project_key: str, api_key: str = Depends(verify_api_key)):
    """
    Retrieves metadata for a Jira project, including available issue types and fields like priorities.
    
    Args:
        project_key: Jira project key.
        api_key: Verified API key.
    
    Returns:
        Dictionary with project metadata.
    """
    try:
        jira = JIRA(
            server=settings.jira_server,
            basic_auth=(settings.jira_username, settings.jira_api_token),
            options={"rest_api_version": "3"}
        )
        meta = jira.createmeta(projectKeys=project_key, expand="projects.issuetypes.fields")
        if not meta["projects"]:
            raise HTTPException(status_code=400, detail=f"Invalid project: {project_key}")
        
        project = meta["projects"][0]
        issuetypes = {}
        for it in project["issuetypes"]:
            fields_info = {}
            for field, info in it["fields"].items():
                if field in ["priority", "labels", "assignee"]:
                    fields_info[field] = {
                        "required": info["required"],
                        "name": info["name"],
                        "allowedValues": [v["name"] for v in info.get("allowedValues", [])] if "allowedValues" in info and field != "labels" else None
                    }
            issuetypes[it["name"]] = {
                "description": it.get("description", ""),
                "fields": fields_info
            }
        return {"project": project["key"], "issuetypes": issuetypes}
    except JIRAError as e:
        logger.error({"error": str(e), "type": "MetadataError"})
        raise HTTPException(status_code=400, detail=f"Failed to retrieve metadata: {str(e)}")

@app.get("/health", response_model=Dict[str, Any], tags=["Health"])
async def health_check(api_key: str = Depends(verify_api_key)):
    """
    Checks connectivity to Jira server and overall API health.
    
    Args:
        api_key: Verified API key.
    
    Returns:
        Dictionary with health status and Jira server information.
    """
    headers = {"Accept": "application/json"}
    async with httpx.AsyncClient(base_url=settings.jira_server, auth=httpx.BasicAuth(settings.jira_username, settings.jira_api_token), headers=headers) as client:
        try:
            resp = await client.get("/rest/api/3/serverInfo")
            resp.raise_for_status()
            info = resp.json()
            return {"status": "healthy", "jira_version": info.get("version"), "timestamp": datetime.now().isoformat()}
        except httpx.HTTPError as e:
            logger.error({"error": str(e), "type": "JIRAConnectionError"})
            return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/help", response_model=Dict[str, Any], tags=["Health"])
async def help_endpoint():
    """
    Provides help information for using the API, including endpoint details and examples.
    
    Returns:
        Dictionary containing help information and usage examples.
    """
    return {
        "message": "Jira Ticket Creator API Help",
        "version": "2.4.0",
        "interactive_docs": "You can access the interactive Swagger UI documentation at /docs to test endpoints directly by providing required information.",
        "endpoints": [
            {
                "path": "/create_ticket",
                "method": "POST",
                "description": "Creates a Jira ticket with optional AI enhancement and Teams notification.",
                "headers": {"X-API-Key": "Your API key"},
                "body_example": TicketInput.Config.schema_extra["example"]
            },
            {
                "path": "/project_metadata/{project_key}",
                "method": "GET",
                "description": "Retrieves metadata for a Jira project, including issue types and fields.",
                "headers": {"X-API-Key": "Your API key"}
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Checks API and Jira server connectivity.",
                "headers": {"X-API-Key": "Your API key"}
            }
        ],
        "priority_options": AVAILABLE_PRIORITIES,
        "contact": "support@example.com"
    }

if __name__ == "__main__":
    uvicorn.run("jira_chatbot:app", host="0.0.0.0", port=8000, reload=True) 
