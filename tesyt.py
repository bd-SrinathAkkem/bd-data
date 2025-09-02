import unittest
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from jira import JIRAError
from fastapi import HTTPException
import httpx
from datetime import datetime
from jira_chatbot import app, Settings, TicketInput, AIConfig, AIProcessor, TeamsNotifier, text_to_adf

# Initialize FastAPI test client
client = TestClient(app)

# Mock settings for testing
class TestSettings(Settings):
    jira_server = "https://test-jira.atlassian.net"
    jira_username = "test_user"
    jira_api_token = "test_token"
    api_key = "test_api_key"
    teams_webhook_url = "https://test-teams.webhook"
    allowed_origins = "*"
    ai_provider_default = "ollama"

settings = TestSettings()

# Test suite
class TestJiraTicketCreatorAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-API-Key": settings.api_key}
        self.valid_ticket_input = {
            "summary": "Test ticket",
            "description": "Test description\n- Item 1\n- Item 2",
            "project": "TEST",
            "issuetype": "Task",
            "priority": "P1 - High Priority",
            "labels": ["test", "urgent"],
            "assignee": "test_user",
            "attachment_urls": ["https://example.com/test.png"],
            "board_id": 123,
            "team_name": "TestTeam",
            "enable_ai": False,
            "enable_teams_notification": True,
            "notification_template": "New ticket {ticket_key}: {summary}"
        }

    @pytest.mark.asyncio
    @patch("jira_chatbot.JIRA")
    @patch("jira_chatbot.httpx.AsyncClient.get")
    @patch("jira_chatbot.connectorcard")
    async def test_create_ticket_success(self, mock_teams, mock_httpx_get, mock_jira):
        # Mock Jira client
        mock_issue = MagicMock()
        mock_issue.key = "TEST-123"
        mock_issue.fields.summary = "Test ticket"
        mock_issue.fields.description = text_to_adf("Test description\n- Item 1\n- Item 2")
        mock_issue.fields.issuetype.name = "Task"
        mock_issue.fields.project.key = "TEST"
        mock_issue.fields.priority.name = "P1 - High Priority"
        mock_issue.fields.labels = ["test", "urgent"]
        mock_issue.fields.assignee.displayName = "test_user"
        mock_issue.fields.created = "2025-09-02T11:26:00.000+0000"
        mock_issue.fields.status.name = "To Do"

        mock_jira_instance = MagicMock()
        mock_jira_instance.create_issue.return_value = mock_issue
        mock_jira_instance.createmeta.return_value = {
            "projects": [{
                "key": "TEST",
                "issuetypes": [{
                    "name": "Task",
                    "fields": {
                        "priority": {"allowedValues": [{"name": "P1 - High Priority"}]},
                        "labels": {"required": False},
                        "assignee": {"required": False},
                        "customfield_10001": {"required": False}
                    }
                }]
            }]
        }
        mock_jira_instance.boards.return_value = [MagicMock(id=123)]
        mock_jira_instance.sprints.return_value = [MagicMock(id=456)]
        mock_jira.return_value = mock_jira_instance

        # Mock HTTPX for attachment
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"test_content"
        mock_httpx_get.return_value = mock_response

        # Mock Teams notifier
        mock_teams_instance = MagicMock()
        mock_teams.return_value = mock_teams_instance

        # Send request
        response = self.client.post("/create_ticket", json=self.valid_ticket_input, headers=self.headers)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["ticket_key"] == "TEST-123"
        assert response_json["ticket_url"] == f"{settings.jira_server}/browse/TEST-123"
        assert response_json["ticket_details"]["summary"] == "Test ticket"
        assert response_json["message"] == "Ticket created successfully"

        # Verify Teams notification
        mock_teams_instance.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_ticket_invalid_api_key(self):
        response = self.client.post("/create_ticket", json=self.valid_ticket_input, headers={"X-API-Key": "wrong_key"})
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid API key"}

    @pytest.mark.asyncio
    @patch("jira_chatbot.JIRA")
    async def test_create_ticket_invalid_project(self, mock_jira):
        mock_jira_instance = MagicMock()
        mock_jira_instance.createmeta.return_value = {"projects": []}
        mock_jira.return_value = mock_jira_instance

        response = self.client.post("/create_ticket", json=self.valid_ticket_input, headers=self.headers)
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid project: TEST"}

    @pytest.mark.asyncio
    @patch("jira_chatbot.JIRA")
    async def test_create_ticket_invalid_priority(self, mock_jira):
        mock_jira_instance = MagicMock()
        mock_jira_instance.createmeta.return_value = {
            "projects": [{
                "key": "TEST",
                "issuetypes": [{
                    "name": "Task",
                    "fields": {
                        "priority": {"allowedValues": [{"name": "P2 - Medium Priority"}]},
                        "labels": {"required": False},
                        "assignee": {"required": False}
                    }
                }]
            }]
        }
        mock_jira.return_value = mock_jira_instance

        response = self.client.post("/create_ticket", json=self.valid_ticket_input, headers=self.headers)
        assert response.status_code == 400
        assert response.json() == {
            "detail": "Priority 'P1 - High Priority' not allowed for issue type 'Task' in project TEST. Valid options: ['P2 - Medium Priority']"
        }

    @pytest.mark.asyncio
    @patch("jira_chatbot.JIRA")
    @patch("jira_chatbot.AIProcessor")
    async def test_create_ticket_with_ai_enhancement(self, mock_ai_processor, mock_jira):
        ticket_input = self.valid_ticket_input.copy()
        ticket_input["enable_ai"] = True
        ticket_input["ai_configs"] = [{"provider": "ollama", "api_url": "http://localhost:11434", "model": "llama2"}]

        # Mock Jira
        mock_issue = MagicMock()
        mock_issue.key = "TEST-123"
        mock_jira_instance = MagicMock()
        mock_jira_instance.create_issue.return_value = mock_issue
        mock_jira_instance.createmeta.return_value = {
            "projects": [{
                "key": "TEST",
                "issuetypes": [{
                    "name": "Task",
                    "fields": {
                        "priority": {"allowedValues": [{"name": "P1 - High Priority"}]},
                        "labels": {"required": False},
                        "assignee": {"required": False}
                    }
                }]
            }]
        }
        mock_jira.return_value = mock_jira_instance

        # Mock AIProcessor
        mock_ai_instance = AsyncMock()
        mock_ai_instance.enhance_ticket_data.return_value = {
            "summary": "Enhanced ticket",
            "description": text_to_adf("Enhanced description\n- Item 1\n- Item 2"),
            "project": "TEST",
            "issuetype": "Task",
            "priority": "P1 - High Priority",
            "labels": ["test", "urgent"],
            "assignee": "test_user"
        }
        mock_ai_processor.return_value = mock_ai_instance

        response = self.client.post("/create_ticket", json=ticket_input, headers=self.headers)
        assert response.status_code == 200
        assert response.json()["ticket_key"] == "TEST-123"
        mock_ai_instance.enhance_ticket_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_ticket_missing_ai_configs(self):
        ticket_input = self.valid_ticket_input.copy()
        ticket_input["enable_ai"] = True
        ticket_input["ai_configs"] = None

        response = self.client.post("/create_ticket", json=ticket_input, headers=self.headers)
        assert response.status_code == 400
        assert response.json() == {"detail": "AI configs required if enable_ai is true"}

    @pytest.mark.asyncio
    @patch("jira_chatbot.JIRA")
    async def test_create_ticket_jira_error(self, mock_jira):
        mock_jira_instance = MagicMock()
        mock_jira_instance.createmeta.side_effect = JIRAError("Jira server error")
        mock_jira.return_value = mock_jira_instance

        response = self.client.post("/create_ticket", json=self.valid_ticket_input, headers=self.headers)
        assert response.status_code == 500
        assert response.json() == {"detail": "Jira connection failed: Jira server error"}

    @pytest.mark.asyncio
    @patch("jira_chatbot.JIRA")
    async def test_get_project_metadata_success(self, mock_jira):
        mock_jira_instance = MagicMock()
        mock_jira_instance.createmeta.return_value = {
            "projects": [{
                "key": "TEST",
                "issuetypes": [{
                    "name": "Task",
                    "description": "Task issue type",
                    "fields": {
                        "priority": {
                            "required": False,
                            "name": "Priority",
                            "allowedValues": [{"name": "P1 - High Priority"}]
                        },
                        "labels": {"required": False, "name": "Labels"},
                        "assignee": {"required": False, "name": "Assignee"}
                    }
                }]
            }]
        }
        mock_jira.return_value = mock_jira_instance

        response = self.client.get("/project_metadata/TEST", headers=self.headers)
        assert response.status_code == 200
        assert response.json() == {
            "project": "TEST",
            "issuetypes": {
                "Task": {
                    "description": "Task issue type",
                    "fields": {
                        "priority": {"required": False, "name": "Priority", "allowedValues": ["P1 - High Priority"]},
                        "labels": {"required": False, "name": "Labels", "allowedValues": None},
                        "assignee": {"required": False, "name": "Assignee", "allowedValues": None}
                    }
                }
            }
        }

    @pytest.mark.asyncio
    async def test_get_project_metadata_invalid_project(self):
        with patch("jira_chatbot.JIRA") as mock_jira:
            mock_jira_instance = MagicMock()
            mock_jira_instance.createmeta.return_value = {"projects": []}
            mock_jira.return_value = mock_jira_instance

            response = self.client.get("/project_metadata/INVALID", headers=self.headers)
            assert response.status_code == 400
            assert response.json() == {"detail": "Invalid project: INVALID"}

    @pytest.mark.asyncio
    @patch("jira_chatbot.httpx.AsyncClient.get")
    async def test_health_check_success(self, mock_httpx_get):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "9.0.0"}
        mock_httpx_get.return_value = mock_response

        response = self.client.get("/health", headers=self.headers)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["jira_version"] == "9.0.0"

    @pytest.mark.asyncio
    @patch("jira_chatbot.httpx.AsyncClient.get")
    async def test_health_check_failure(self, mock_httpx_get):
        mock_httpx_get.side_effect = httpx.HTTPError("Connection failed")

        response = self.client.get("/health", headers=self.headers)
        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"
        assert "Connection failed" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_help_endpoint(self):
        response = self.client.get("/help")
        assert response.status_code == 200
        assert response.json()["message"] == "Jira Ticket Creator API Help"
        assert response.json()["version"] == "1.0.0"
        assert len(response.json()["endpoints"]) == 3
        assert "priority_options" in response.json()

    def test_text_to_adf(self):
        text = "Paragraph\n- Item 1\n- Item 2\nAnother paragraph"
        adf = text_to_adf(text)
        expected = {
            "version": 1,
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Paragraph"}]},
                {"type": "bulletList", "content": [
                    {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Item 1"}]}]},
                    {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Item 2"}]}]}
                ]},
                {"type": "paragraph", "content": [{"type": "text", "text": "Another paragraph"}]}
            ]
        }
        assert adf == expected

    @pytest.mark.asyncio
    @patch("jira_chatbot.importlib.import_module")
    async def test_ai_processor_ollama(self, mock_import):
        mock_ollama = MagicMock()
        mock_ollama.AsyncClient.return_value.generate = AsyncMock(return_value={"response": json.dumps({
            "summary": "Enhanced ticket",
            "description": "Enhanced description\n- Item 1",
            "project": "TEST",
            "issuetype": "Task"
        })})
        mock_import.return_value = mock_ollama

        ai_configs = [AIConfig(provider="ollama", api_url="http://localhost:11434", model="llama2")]
        processor = AIProcessor(ai_configs)
        result = await processor.enhance_ticket_data({"summary": "Test", "description": "Test desc", "project": "TEST"}, ["P1"])
        assert result["summary"] == "Enhanced ticket"
        assert result["description"] == text_to_adf("Enhanced description\n- Item 1")

    @pytest.mark.asyncio
    async def test_teams_notifier_success(self):
        with patch("jira_chatbot.connectorcard") as mock_teams:
            mock_teams_instance = MagicMock()
            mock_teams.return_value = mock_teams_instance
            notifier = TeamsNotifier("https://test.webhook")
            await notifier.send_notification(
                template="New ticket {ticket_key}: {summary}",
                ticket_key="TEST-123",
                summary="Test ticket",
                ticket_url="https://jira.test/browse/TEST-123",
                priority="P1",
                assignee="test_user"
            )
            mock_teams_instance.send.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
