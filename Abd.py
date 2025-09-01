"""
Jira Ticket Creator API (Production-Ready)

A scalable, secure FastAPI application for creating Jira tickets with AI-enhanced data,
automatic addition to the current sprint, and Microsoft Teams notifications.
Supports custom priorities, dynamic AI imports, and robust validation.

Key Features:
- Async endpoints for concurrent request handling.
- Validates custom priorities from the provided list.
- Adds tickets to the active sprint (requires board_id or auto-fetches if single board).
- Optional AI enhancement with Ollama, Claude, or custom providers.
- Rich Teams notifications with customizable templates.
- API key authentication and CORS for security.
- Input validation using Jira metadata for priorities, issue types, and custom fields.
- Structured logging for observability.
- No Redis dependency.

Dependencies:
- fastapi
- uvicorn
- jira
- pydantic
- pydantic-settings
- pymsteams
- httpx
- tenacity
- Optional: ollama, anthropic (based on AI provider)

Environment Variables:
- JIRA_SERVER: Jira instance URL
- JIRA_USERNAME: Jira username
- JIRA_API_TOKEN: Jira API token
- TEAMS_WEBHOOK_URL: Optional Teams webhook URL
- API_KEY: Required API key
- ALLOWED_ORIGINS: Comma-separated CORS origins (default: *)
- AI_PROVIDER_DEFAULT: Default AI provider (ollama/claude/custom)

Run Locally:
uvicorn main:app --reload

Production Run:
gunicorn -k uvicorn.workers.UvicornWorker -w 4 main:app

Example Request:
curl -X POST http://localhost:8000/create_ticket \
-H "X-API-Key: your-api-key" \
-H "Content-Type: application/json" \
-d '{
  "summary": "Fix login issue",
  "description": "Users cannot log in due to auth error",
  "project": "PROJ",
  "issuetype": "Bug",
  "priority": "P1 Current priority",
  "board_id": 123,
  "enable_ai": true,
  "ai_configs": [{"provider": "ollama", "api_url": "http://localhost:11434", "model": "llama2"}],
  "enable_teams_notification": true
}'

"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager
from datetime import datetime
import importlib

from fastapi import FastAPI, HTTPException, Body, Depends, Security, status, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from jira import JIRA, JIRAError
from jira.resources import Board, Sprint
from pymsteams import connectorcard
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
class Settings(BaseSettings):
    jira_server: str = Field(default="https://your-jira-instance.atlassian.net")
    jira_username: str
    jira_api_token: str
    teams_webhook_url: Optional[str] = None
    api_key: str
    allowed_origins: str = "*"
    ai_provider_default: str = "ollama"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# FastAPI app
app = FastAPI(
    title="Jira Ticket Creator API",
    description="API for creating Jira tickets with AI, sprint integration, and Teams notifications.",
    version="2.1.0",
    openapi_tags=[
        {"name": "Tickets", "description": "Jira ticket operations"},
        {"name": "Health", "description": "System health checks"}
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key

# Pydantic models
class JiraConfig(BaseModel):
    server: str = Field(default=settings.jira_server)
    username: str = Field(default=settings.jira_username)
    api_token: str = Field(default=settings.jira_api_token)

class AIConfig(BaseModel):
    provider: str = Field(default=settings.ai_provider_default)
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = Field(default="llama2")

class TicketInput(BaseModel):
    summary: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    project: str = Field(..., min_length=1)
    team_name: Optional[str] = None
    issuetype: str = Field(default="Task")
    priority: Optional[str] = Field(None, description="Priority from: ['Pe Urgent', 'P1 Current priority', 'P2 High Priority', ...]")
    board_id: Optional[int] = Field(None, description="Board ID for sprint")
    other_fields: Optional[Dict[str, Any]] = None
    enable_ai: bool = Field(default=False)
    ai_configs: Optional[List[AIConfig]] = None
    enable_teams_notification: bool = Field(default=False)
    notification_template: Optional[str] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "summary": "Fix login issue",
                "description": "Users cannot log in due to auth error",
                "project": "PROJ",
                "issuetype": "Bug",
                "priority": "P1 Current priority",
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

# Custom priorities from your log
AVAILABLE_PRIORITIES = [
    'Pe Urgent', 'P1 Current priority', 'P2 Top of the backlog', 'PS-Negotiated or opportunistic',
    'P4-iated or opportunistic', 'P9- Fix Immediately!', 'P2 High Priority', 'P2 Medium Priority',
    'P4 Low Priority', 'P4 Standard Ticket', 'Unprioritized', 'None', 'Blocker', 'tical', 'Major',
    'Normal', 'Rinor', 'Trivial', 'Unassigned', 'P1 Production Impact', 'P3 Standard Priority',
    'P5 Best Effort'
]

# Jira connection
@contextmanager
def jira_connection(config: JiraConfig):
    try:
        jira = JIRA(
            server=config.server,
            basic_auth=(config.username, config.api_token),
            options={"rest_api_version": "3"}
        )
        yield jira
    except JIRAError as e:
        logger.error({"error": str(e), "type": "JiraConnectionError"})
        raise HTTPException(status_code=500, detail=f"Jira connection error: {str(e)}")
    finally:
        pass

# AI provider registry
class AIProviderRegistry:
    providers: Dict[str, Callable] = {}

    @classmethod
    def register(cls, provider_name: str):
        def decorator(func: Callable):
            cls.providers[provider_name] = func
            return func
        return decorator

@AIProviderRegistry.register("ollama")
def get_ollama_client(config: AIConfig):
    try:
        ollama = importlib.import_module("ollama")
        if not config.api_url:
            raise ValueError("Ollama requires api_url")
        return ollama.AsyncClient(host=config.api_url)
    except ImportError:
        raise HTTPException(status_code=400, detail="Ollama library not installed")

@AIProviderRegistry.register("claude")
def get_claude_client(config: AIConfig):
    try:
        anthropic = importlib.import_module("anthropic")
        if not config.api_key:
            raise ValueError("Claude requires api_key")
        return anthropic.AsyncAnthropic(api_key=config.api_key)
    except ImportError:
        raise HTTPException(status_code=400, detail="Anthropic library not installed")

@AIProviderRegistry.register("custom")
def get_custom_client(config: AIConfig):
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

# AI processing
class AIProcessor:
    def __init__(self, configs: List[AIConfig]):
        self.clients = []
        for config in configs:
            provider_func = AIProviderRegistry.providers.get(config.provider)
            if not provider_func:
                raise HTTPException(status_code=400, detail=f"Unsupported AI provider: {config.provider}")
            self.clients.append((config, provider_func(config)))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def enhance_ticket_data(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"Professionally enhance this Jira ticket data: Make the summary concise (under 255 characters), "
            f"structure the description with bullet points, and add logical details. "
            f"Return valid JSON matching the original structure: {json.dumps(ticket_data)}"
        )
        
        for config, client in self.clients:
            try:
                if config.provider == "ollama":
                    response = await client.generate(model=config.model, prompt=prompt)
                    return json.loads(response['response'])
                elif config.provider == "claude":
                    message = f"{importlib.import_module('anthropic').HUMAN_PROMPT} {prompt} {importlib.import_module('anthropic').AI_PROMPT}"
                    response = await client.completions.create(
                        model=config.model,
                        max_tokens_to_sample=500,
                        prompt=message
                    )
                    return json.loads(response.completion)
                elif config.provider == "custom":
                    response_text = await client(prompt)
                    return json.loads(response_text)
            except Exception as e:
                logger.warning({"provider": config.provider, "error": str(e)})
                continue
        logger.error("All AI providers failed, using original data")
        return ticket_data

# Teams notifier
class TeamsNotifier:
    def __init__(self, webhook_url: str):
        if not webhook_url:
            raise ValueError("Teams webhook URL required")
        self.webhook_url = webhook_url

    async def send_notification(self, template: str, ticket_key: str, summary: str):
        try:
            message = template.format(ticket_key=ticket_key, summary=summary) if template else f"New Jira ticket {ticket_key}: {summary}"
            teams_message = connectorcard(self.webhook_url)
            card = {
                "type": "AdaptiveCard",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.0",
                "body": [
                    {"type": "TextBlock", "text": message, "weight": "bolder", "size": "medium"}
                ]
            }
            teams_message.payload = card
            teams_message.send()
            logger.info({"action": "teams_notification_sent", "ticket_key": ticket_key})
        except Exception as e:
            logger.error({"error": str(e), "type": "TeamsNotificationError"})
            raise HTTPException(status_code=500, detail="Failed to send Teams notification")

# Dependencies
def get_jira_config() -> JiraConfig:
    if not settings.jira_username or not settings.jira_api_token:
        raise HTTPException(status_code=500, detail="Jira credentials not configured")
    return JiraConfig()

def get_teams_notifier() -> Optional[TeamsNotifier]:
    if settings.teams_webhook_url:
        return TeamsNotifier(settings.teams_webhook_url)
    return None

# Endpoints
@app.post("/create_ticket", response_model=Dict[str, str], tags=["Tickets"])
async def create_ticket(
    request: Request,
    input_data: TicketInput = Body(...),
    jira_config: JiraConfig = Depends(get_jira_config),
    notifier: Optional[TeamsNotifier] = Depends(get_teams_notifier),
    api_key: str = Depends(verify_api_key)
):
    """
    Create a Jira ticket with AI enhancement, sprint addition, and Teams notification.
    Validates priority against provided list and issue type against project metadata.
    """
    logger.info({"action": "create_ticket_start", "client_ip": request.client.host})
    
    if input_data.enable_ai and not input_data.ai_configs:
        raise HTTPException(status_code=400, detail="AI configs required if enable_ai is true")
    if input_data.enable_teams_notification and not notifier:
        raise HTTPException(status_code=400, detail="Teams webhook not configured")

    ticket_data = input_data.dict(exclude={"enable_ai", "ai_configs", "enable_teams_notification", "notification_template", "board_id"}, exclude_none=True)

    # AI enhancement
    if input_data.enable_ai:
        processor = AIProcessor(input_data.ai_configs)
        ticket_data = await processor.enhance_ticket_data(ticket_data)

    # Prepare fields
    fields = {
        "project": {"key": ticket_data["project"]},
        "summary": ticket_data["summary"],
        "description": ticket_data["description"],
        "issuetype": {"name": ticket_data.get("issuetype", "Task")},
    }
    if input_data.priority:
        if input_data.priority not in AVAILABLE_PRIORITIES:
            raise HTTPException(status_code=400, detail=f"Invalid priority. Valid options: {AVAILABLE_PRIORITIES}")
        fields["priority"] = {"name": input_data.priority}
    if input_data.team_name:
        fields["customfield_10000"] = input_data.team_name  # Adjust ID
    if input_data.other_fields:
        fields.update(input_data.other_fields)

    # Create ticket and handle sprint
    with jira_connection(jira_config) as jira:
        # Validate fields and issue type
        try:
            meta = jira.createmeta(projectKeys=ticket_data["project"], issuetypeNames=fields["issuetype"]["name"], expand="projects.issuetypes.fields")
            if not meta["projects"]:
                raise HTTPException(status_code=400, detail=f"Invalid project: {ticket_data['project']}")
            allowed_fields = meta["projects"][0]["issuetypes"][0]["fields"]
            for field in fields:
                if field not in allowed_fields and field not in {"project", "issuetype", "priority"}:
                    raise HTTPException(status_code=400, detail=f"Invalid field: {field}")
        except JIRAError as e:
            logger.error({"error": str(e), "type": "ValidationError"})
            raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

        # Create issue
        try:
            issue = jira.create_issue(fields=fields)
            ticket_key = issue.key
            logger.info({"action": "ticket_created", "key": ticket_key})
        except JIRAError as e:
            logger.error({"error": str(e), "type": "TicketCreationError"})
            raise HTTPException(status_code=400, detail=f"Ticket creation failed: {str(e)}")

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

    # Teams notification
    if input_data.enable_teams_notification:
        await notifier.send_notification(input_data.notification_template, ticket_key, ticket_data["summary"])

    return {"ticket_key": ticket_key, "message": "Ticket created successfully"}

@app.get("/health", response_model=Dict[str, Any], tags=["Health"])
async def health_check(jira_config: JiraConfig = Depends(get_jira_config), api_key: str = Depends(verify_api_key)):
    """
    Check Jira connectivity and API health.
    """
    with jira_connection(jira_config) as jira:
        try:
            info = jira.server_info()
            return {"status": "healthy", "jira_version": info.get("version")}
        except JIRAError as e:
            logger.error({"error": str(e), "type": "HealthCheckError"})
            return {"status": "unhealthy", "error": str(e)}

# Unit test stub
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

client = TestClient(app)

@pytest.mark.asyncio
async def test_create_ticket():
    with patch('jira.JIRA') as mock_jira:
        mock_instance = mock_jira.return_value
        mock_instance.create_issue.return_value.key = 'PROJ-123'
        mock_instance.boards.return_value = [Board({'id': 1})]
        mock_instance.sprints.return_value = [Sprint({'id': 1})]
        mock_instance.createmeta.return_value = {
            "projects": [{"issuetypes": [{"fields": {"summary": {}, "description": {}, "issuetype": {}, "priority": {}}}]}]
        }
        headers = {"X-API-Key": settings.api_key}
        payload = {
            "summary": "Test",
            "description": "Desc",
            "project": "PROJ",
            "priority": "P1 Current priority"
        }
        response = client.post("/create_ticket", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["ticket_key"] == "PROJ-123"
"""
