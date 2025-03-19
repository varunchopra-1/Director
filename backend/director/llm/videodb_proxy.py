import os
import json
import requests
from enum import Enum
from typing import List, Dict, Optional
import logging
from pydantic import Field, field_validator, FieldValidationInfo


from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import (
    LLMType,
)

logger = logging.getLogger(__name__)


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT4o = "gpt-4o-2024-11-20"


class VideoDBProxyConfig(BaseLLMConfig):
    """OpenAI Config"""

    llm_type: str = LLMType.VIDEODB_PROXY
    api_key: str = os.getenv("VIDEO_DB_API_KEY")
    api_base: str = os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")
    chat_model: str = Field(default=OpenAIChatModel.GPT4o)
    max_tokens: int = 4096
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:5000")

    @field_validator("api_key")
    @classmethod
    def validate_non_empty(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError("Please set VIDEO_DB_API_KEY environment variable.")
        return v


class VideoDBProxy(BaseLLM):
    def __init__(self, config: VideoDBProxyConfig = None):
        """
        :param config: OpenAI Config
        """
        if config is None:
            config = VideoDBProxyConfig()
        super().__init__(config=config)
        try:
            import openai
        except ImportError:
            raise ImportError("Please install OpenAI python library.")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=f"{self.api_base}")
        self.conversation_history = []
        self.mcp_server_url = config.mcp_server_url

    def _format_messages(self, messages: list):
        """Format the messages to the format that OpenAI expects."""
        formatted_messages = []
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "function": {
                                    "name": tool_call["tool"]["name"],
                                    "arguments": json.dumps(
                                        tool_call["tool"]["arguments"]
                                    ),
                                },
                                "type": tool_call["type"],
                            }
                            for tool_call in message["tool_calls"]
                        ],
                    }
                )
            else:
                formatted_messages.append(message)
        return formatted_messages

    def _format_tools(self, tools: list):
        """Format the tools to the format that OpenAI expects.

        **Example**::

            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_delivery_date",
                        "description": "Get the delivery date for a customer's order.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "The customer's order ID."
                                }
                            },
                            "required": ["order_id"],
                            "additionalProperties": False
                        }
                    }
                }
            ]
        """
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                    "strict": True,
                }
            )
        return formatted_tools

    def list_mcp_tools(self):
        """List available MCP Knowledge Graph tools."""
        try:
            response = requests.get(f"{self.mcp_server_url}/api/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error listing MCP tools: {e}")
            return {"tools": []}

    def call_mcp_tool(self, tool_name, arguments):
        """Call a tool on the MCP Knowledge Graph Server."""
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "callTool",
                "params": {"name": tool_name, "arguments": arguments}
            }
            response = requests.post(f"{self.mcp_server_url}/api/mcp", json=mcp_request)
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise Exception(f"MCP error: {result['error']}")
            
            for content_item in result.get("result", {}).get("content", []):
                if content_item["type"] == "text":
                    try:
                        return json.loads(content_item["text"])
                    except json.JSONDecodeError:
                        return {"message": content_item["text"]}
            return result
        except Exception as e:
            print(f"Error calling MCP tool {tool_name}: {e}")
            return None

    def extract_insights(self, messages: List[Dict[str, str]]):
        """Extract insights from conversation messages and store them in the knowledge graph"""
        try:
            prompt = """
                Analyze the following conversation and extract important entities, observations, and relationships.
                For each entity found, categorize it with an appropriate entity type.
                For each relationship found, express it in active voice (e.g. "Alice knows Bob").

                Your response MUST be valid JSON with the following structure:
                {
                    "entities": [
                        {"name": "entity_name", "entityType": "person/concept/place/etc", "observations": ["observation1", "observation2"]}
                    ],
                    "relations": [
                        {"from": "entity1_name", "to": "entity2_name", "relationType": "relationship_type"}
                    ]
                }

                Only extract factual information, not opinions or hypotheticals unless they represent user preferences.
                Return ONLY the JSON object, with no additional text before or after.
            """
            
            conv_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": conv_text}
                ]
            )
            
            insights_raw = json.loads(response.choices[0].message.content)
            insights = {
                "entities": [
                    {
                        "name": e["name"].lower(),
                        "entityType": e["entityType"].lower(),
                        "observations": [obs.lower() for obs in e["observations"]]
                    }
                    for e in insights_raw.get("entities", [])
                ],
                "relations": [
                    {
                        "from": r["from"].lower(),
                        "to": r["to"].lower(),
                        "relationType": r["relationType"].lower()
                    }
                    for r in insights_raw.get("relations", [])
                ]
            }
            
            if insights.get("entities"):
                self.call_mcp_tool("create_entities", {"entities": insights["entities"]})
            
            if insights.get("relations"):
                self.call_mcp_tool("create_relations", {"relations": insights["relations"]})
                
            return insights
        except Exception as e:
            print(f"Error extracting insights: {e}")
            return None

    def find_relevant_information(self, query: str) -> Optional[str]:
        """Find relevant information from knowledge graph based on query."""
        try:
            search_results = self.call_mcp_tool("search_nodes", {"query": query})
            
            if not search_results or not search_results.get("entities"):
                return None
                
            entities = search_results.get("entities", [])
            relations = search_results.get("relations", [])
            
            context = "Known facts from previous conversation:\n"
            
            for entity in entities:
                if entity['entityType'].lower() == 'person':
                    context += f"• {entity['name']} is a person.\n"
                    for obs in entity['observations'][:3]:
                        context += f"• {obs}\n"
                else:
                    context += f"• {entity['name']} is a {entity['entityType']}.\n"
                    for obs in entity['observations'][:3]:
                        context += f"• {obs}\n"
            
            for relation in relations:
                context += f"• {relation['from']} {relation['relationType']} {relation['to']}.\n"
                    
            return context
        except Exception as e:
            print(f"Error finding relevant information: {e}")
            return None

    def chat_completions(
        self, messages: list, tools: list = [], stop=None, response_format=None):
        
        for message in messages:
            if message.get("role") and message.get("content"):
                self.conversation_history.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        latest_messages = [msg for msg in messages if msg.get("role") and msg.get("content")]
        if latest_messages:
            self.extract_insights(latest_messages)
        
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        relevant_info = None
        if user_messages:
            last_user_message = user_messages[-1]["content"]
            # logger.info("Finding relevant information..........................................")
            # logger.info("Finding relevant informationTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            logger.info(f"Last user message: {last_user_message}")
            relevant_info = self.find_relevant_information(last_user_message)
        
        enhanced_messages = messages.copy()
        if relevant_info:
            system_message_found = False
            for i, message in enumerate(enhanced_messages):
                if message.get("role") == "system":
                    enhanced_messages[i]["content"] += f"\n\n{relevant_info}"
                    system_message_found = True
                    break
            
            if not system_message_found:
                enhanced_messages.insert(0, {
                    "role": "system", 
                    "content": f"You are an intelligent assistant that helps users with their queries. {relevant_info}"
                })

        params = {
            "model": self.chat_model,
            "messages": self._format_messages(enhanced_messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": stop,
            "timeout": self.timeout,
        }
        if tools:
            params["tools"] = self._format_tools(tools)
            params["tool_choice"] = "auto"

        if response_format:
            params["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            print(f"Error: {e}")
            return LLMResponse(content=f"Error: {e}")
        
        if response.choices[0].message.content:
            self.conversation_history.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            self.extract_insights([{
                "role": "assistant",
                "content": response.choices[0].message.content
            }])

        return LLMResponse(
            content=response.choices[0].message.content or "",
            tool_calls=[
                {
                    "id": tool_call.id,
                    "tool": {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    },
                    "type": tool_call.type,
                }
                for tool_call in response.choices[0].message.tool_calls
            ]
            if response.choices[0].message.tool_calls
            else [],
            finish_reason=response.choices[0].finish_reason,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            status=LLMResponseStatus.SUCCESS,
        )