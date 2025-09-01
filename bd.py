import os
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager
from datetime import datetime
import importlib
import uvicorn

from fastapi import FastAPI, HTTPException, Body, Depends, Security, status, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pymsteams import connectorcard
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from jira import JIRA, JIRAError


# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
class Settings(BaseSettings):
    jira_server: str = Field(default="https://blackduck.atlassian.net")
    jira_username: str
    jira_api_token: str
    teams_webhook_url: Optional[str] = None
    api_key: str
    allowed_origins: str = "*"
    ai_provider_default: str = "ollama"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
    logger.info({
        "action": "load_settings",
        "api_key": "set" if settings.api_key else "missing",
        "jira_api_token": "set" if settings.jira_api_token else "missing"
    })
except Exception as e:
    logger.error({"action": "load_settings_failed", "error": str(e)})
    raise

# FastAPI app
app = FastAPI(
    title="Jira Ticket Creator API",
    description="API for creating Jira tickets with AI, sprint integration, and Teams notifications.",
    version="2.3.0",
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
    if not api_key:
        logger.error({"action": "api_key_validation_failed", "reason": "No API key provided"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No API key provided")
    if api_key != settings.api_key:
        logger.error({"action": "api_key_validation_failed", "provided_key": "hidden_for_security"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key

# Pydantic models
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
    priority: Optional[str] = Field(None, description="Priority from: ['Pe Urgent', 'P1 Current priority', ...]")
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
                "description": "Users cannot log in due to auth error.",
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
    'P0 - Urgent', 'P1 - Current priority', 'P2 - Top of the backlog', 'P3 - Negotiated or opportunistic', 'P4 - Negotiated or opportunistic', 
    'P0 - Fix Immediately!', 'P1 - High Priority', 'P2 - Medium Priority', 'P3 - Low Priority', 'P4 - Standard Ticket', 
    'Unprioritized', 'None', 'Blocker', 'Critical', 'Major', 'Normal', 'Minor', 'Trivial', 'Unassigned', 
    'P1 - Production Impact', 'P2 - High Priority', 'P3 - Standard Priority', 'P4 - Low Priority', 'P5 - Best Effort'
]

# Convert plain text to Atlassian Document Format (ADF), parsing for bullet points
def text_to_adf(text: str) -> Dict[str, Any]:
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
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": item_text}
                            ]
                        }
                    ]
                })
                i += 1
            content.append({
                "type": "bulletList",
                "content": list_content
            })
        else:
            para_lines = []
            while i < len(lines) and not lines[i].strip().startswith('- ') and lines[i].strip():
                para_lines.append(lines[i])
                i += 1
            para_text = '\n'.join(para_lines).strip()
            content.append({
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": para_text}
                ]
            })
            if i < len(lines) and not lines[i].strip():
                i += 1
    if not content:
        content.append({
            "type": "paragraph",
            "content": [
                {"type": "text", "text": ""}
            ]
        })
    return {
        "version": 1,
        "type": "doc",
        "content": content
    }

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
            f"structure the description with bullet points using - and \\n for line breaks, and add logical details. "
            f"The description must be a plain text string, not a JSON object or ADF format. "
            f"Return valid JSON matching the original structure: {json.dumps(ticket_data)}"
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
            teams_message.text(message)
            teams_message.send()
            logger.info({"action": "teams_notification_sent", "ticket_key": ticket_key})
        except Exception as e:
            logger.error({"error": str(e), "type": "TeamsNotificationError"})
            raise HTTPException(status_code=500, detail="Failed to send Teams notification")

# Dependencies
def get_teams_notifier() -> Optional[TeamsNotifier]:
    if settings.teams_webhook_url:
        return TeamsNotifier(settings.teams_webhook_url)
    return None

# Endpoints
@app.post("/create_ticket", response_model=Dict[str, str], tags=["Tickets"])
async def create_ticket(
    request: Request,
    input_data: TicketInput = Body(...),
    notifier: Optional[TeamsNotifier] = Depends(get_teams_notifier),
    api_key: str = Depends(verify_api_key)
):
    """
    Create a Jira ticket with AI enhancement, sprint addition, and Teams notification.
    Validates priority against provided list and issue type against project metadata.
    """
    logger.info({"action": "create_ticket_start", "client_ip": request.client.host, "timestamp": "2025-09-01 16:22:00"})
    
    if input_data.enable_ai and not input_data.ai_configs:
        raise HTTPException(status_code=400, detail="AI configs required if enable_ai is true")
    if input_data.enable_teams_notification and not notifier:
        raise HTTPException(status_code=400, detail="Teams webhook not configured")

    ticket_data = input_data.dict(exclude={"enable_ai", "ai_configs", "enable_teams_notification", "notification_template", "board_id"}, exclude_none=True)

    # AI enhancement
    if input_data.enable_ai:
        processor = AIProcessor(input_data.ai_configs)
        ticket_data = await processor.enhance_ticket_data(ticket_data)
    else:
        ticket_data['description'] = text_to_adf(ticket_data['description'])

    # Prepare fields
    fields = {
        "project": {"key": ticket_data["project"]},
        "summary": ticket_data["summary"],
        "description": ticket_data["description"],
        "issuetype": {"name": ticket_data.get("issuetype", "Task")} if ticket_data.get("issuetype") in [issuetype["name"] for issuetype in meta["projects"][0]["issuetypes"]] else {"name": "Task"},
    }
    if input_data.priority:
        # Check if priority field exists in project metadata
        if "priority" not in allowed_fields:
            raise HTTPException(status_code=400, detail="Priority field not available for this project/issue type")
            
        # Validate priority is allowed for this project
        valid_priorities = [p["name"] for p in allowed_fields["priority"]["allowedValues"]]
        if input_data.priority not in valid_priorities:
            raise HTTPException(status_code=400, detail=f"Priority '{input_data.priority}' not allowed for this project. Valid options: {valid_priorities}")
            
        fields["priority"] = {"name": input_data.priority}
    if input_data.team_name:
        fields["customfield_10001"] = input_data.team_name
    if input_data.other_fields:
        fields.update(input_data.other_fields)

    # Create ticket and handle sprint
    try:
        jira = JIRA(
            server=settings.jira_server,
            basic_auth=(settings.jira_username, settings.jira_api_token),
            options={"rest_api_version": "3"}
        )
        meta = jira.createmeta(projectKeys=ticket_data["project"], issuetypeNames=fields["issuetype"]["name"], expand="projects.issuetypes.fields")
        if not meta["projects"]:
            raise HTTPException(status_code=400, detail=f"Invalid project: {ticket_data['project']}")
        allowed_fields = meta["projects"][0]["issuetypes"][0]["fields"]
        for field in fields:
            if field not in allowed_fields and field not in {"project", "issuetype", "priority", "description"}:
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
        await notifier.send_notification(
            input_data.notification_template,
            ticket_key,
            ticket_data["summary"] if ticket_data["summary"] else ticket_data["description"]
        )
    return {"ticket_key": ticket_key, "message": "Ticket created successfully"}

@app.get("/health", response_model=Dict[str, Any], tags=["Health"])
async def health_check(api_key: str = Depends(verify_api_key)):
    """
    Check Jira connectivity and API health.
    """
    headers = {"Accept": "application/json"}
    async with httpx.AsyncClient(base_url=settings.jira_server, auth=httpx.BasicAuth(settings.jira_username, settings.jira_api_token), headers=headers) as client:
        resp = await client.get("/rest/api/3/serverInfo")
        if resp.status_code == 200:
            info = resp.json()
            return {"status": "healthy", "jira_version": info.get("version")}
        else:
            logger.error({"error": resp.text, "type": "JIRAConnectionError"})
            return {"status": "unhealthy", "error": resp.text}

# Main block for direct execution
if __name__ == "__main__":
    uvicorn.run("jira_chatbot:app", host="0.0.0.0", port=8000, reload=True)
