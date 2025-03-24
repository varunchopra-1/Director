import os
import json
from enum import Enum
from typing import Optional, List, Dict, Any
from director.llm.memory_client import get_memory_client
from pydantic import Field, field_validator, FieldValidationInfo


from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import (
    LLMType,
)


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
    mem0_api_key: Optional[str] = os.getenv("MEM0_API_KEY")
    use_mem0: bool = Field(default=True)
    default_context_limit: int = Field(default=5)
    test_user_id: str = Field(default="test_user_123")

    @field_validator("api_key")
    @classmethod
    def validate_non_empty(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError("Please set VIDEO_DB_API_KEY environment variable.")
        return v
    
    @field_validator("mem0_api_key")
    @classmethod
    def validate_mem0_api_key(cls, v, info: FieldValidationInfo):
        # Since use_mem0 is hardcoded to True, this validation is important
        if not v:
            raise ValueError("Please set MEM0_API_KEY environment variable when use_mem0 is True.")
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
        
        # Initialize Mem0 (hardcoded to always try)
        self.mem0_client = None
        if self.config.use_mem0:
            try:
                self.mem0_client = get_memory_client(self.config.mem0_api_key)
            except Exception as e:
                print(f"Error initializing Mem0 client: {e}")
                raise

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
        """Format the tools to the format that OpenAI expects."""
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
    
    def add_to_memory(self, messages: List[Dict[str, Any]], user_id: str = None) -> bool:
        """Add messages to the Mem0 memory layer.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            user_id: Unique identifier for the user (defaults to test_user_id)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if user_id is None:
            user_id = self.config.test_user_id
            
        if not self.mem0_client:
            print("Warning: Mem0 client is not initialized, skipping memory storage")
            return False
            
        try:
            print(f"Adding {len(messages)} messages to Mem0 for user {user_id}")

            self.mem0_client.add(messages, user_id=user_id, output_format="v1.1")
            return True
        except Exception as e:
            print(f"Error adding to Mem0 memory: {e}")
            return False
    
    def get_memory_context(self, query: str, user_id: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant memory context.
        
        Args:
            query: The query or message to find relevant context for
            user_id: Unique identifier for the user (defaults to test_user_id)
            limit: Maximum number of context items to retrieve
            
        Returns:
            List of relevant memory items
        """
        # Always use the test user ID if none specified
        if user_id is None:
            user_id = self.config.test_user_id
            
        if not self.mem0_client:
            print("Warning: Mem0 client is not initialized, skipping memory retrieval")
            return []
            
        try:
            limit = limit or self.config.default_context_limit
            
            print(query)
            if isinstance(query, list) and len(query) > 0:
                query_text = query[0].get('text', '')
                print(query_text)
                query = query_text
            else:
                print(query)
            
            print(f"Retrieving up to {limit} memory items for user {user_id} with query: {query[:50]}...")
            memory_items = self.mem0_client.search(query, user_id=user_id, output_format="v1.1")
            print(f"Retrieved {len(memory_items)} memory items")
            print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            print(memory_items)
            print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            return memory_items
        except Exception as e:
            print(f"Error retrieving from Mem0 memory: {e}")
            return []
    
    def enhance_messages_with_memory(self, messages: List[Dict[str, Any]], user_id: str = None) -> List[Dict[str, Any]]:
        """Enhance the conversation with relevant memory context.
            
        Args:
            messages: List of message dictionaries
            user_id: Unique identifier for the user (defaults to test_user_id)
                
        Returns:
            Enhanced list of messages with memory context
        """
        # Always use the test user ID if none specified
        if user_id is None:
            user_id = self.config.test_user_id
            
        if not self.mem0_client:
            print("Warning: Mem0 client is not initialized, skipping memory enhancement")
            return messages
            
        # Extract the most recent user query to search for relevant context
        recent_queries = [msg["content"] for msg in messages if msg["role"] == "user"]
        if not recent_queries:
            return messages
            
        query = recent_queries[-1]
        memory_items = self.get_memory_context(query, user_id)
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        print(memory_items)
        print("resulting memory items")
        memories = []
        if memory_items and isinstance(memory_items, dict) and "results" in memory_items:
            memories = [item["memory"] for item in memory_items["results"]]
        elif isinstance(memory_items, list) and memory_items:
            # Handle case where it might be a list of memories directly
            memories = [item.get("memory", item) for item in memory_items]

        # Print extracted memories
        for memory in memories:
            print(memory)
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            
        if not memory_items:
            print("No relevant memory items found")
            return messages
            
        print(f"Enhancing messages with {len(memory_items)} memory items")
            
        # Create a system message with the memory context
        memory_context = "Previous relevant information:\n"
        
        for idx, memory in enumerate(memories, 1):
            memory_context += f"{idx}. {memory}\n"
            
        # Find the first system message if it exists
        system_msg_index = next((i for i, msg in enumerate(messages) if msg["role"] == "system"), -1)
            
        if system_msg_index >= 0:
            # Append memory context to existing system message
            enhanced_messages = []
            for i, msg in enumerate(messages):
                if i == system_msg_index:
                    msg["content"] = msg["content"] + "\n\n" + memory_context
                enhanced_messages.append(msg)
        else:
            # Add the memory system message at the beginning
            enhanced_messages = [{"role": "system", "content": memory_context}] + messages

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print(memory_context)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        print("Enhanced messages with memory context")
        print(enhanced_messages)
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        
        return enhanced_messages

    def chat_completions(
        self, messages: list, tools: list = [], stop=None, response_format=None, user_id: Optional[str] = None
    ):
        """Get completions for chat with memory enhancement.

        Args:
            messages: List of message dictionaries
            tools: List of tools for function calling
            stop: Optional stop sequence
            response_format: Optional response format specification
            user_id: Optional user ID for memory retrieval and storage (defaults to test_user_id)
            
        Returns:
            LLMResponse object with completion information
        """
        # Always use the test user ID if none specified
        if user_id is None:
            user_id = self.config.test_user_id
            print(f"Using default test user ID: {user_id}")
        
        # Always enhance messages with memory context in testing mode
        enhanced_messages = self.enhance_messages_with_memory(messages, user_id)
        
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
            
        # Always store the conversation in memory in testing mode
        # Prepare messages for memory storage
        memory_messages = []
        # Add the last user message if it exists
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages:
            memory_messages.append(user_messages[-1])
        
        # Add the assistant's response
        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content or "",
        }
        
        # Add tool calls if present
        if response.choices[0].message.tool_calls:
            assistant_message["tool_calls"] = [
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
        
        memory_messages.append(assistant_message)
        
        # Store in memory
        if memory_messages:
            print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
            for msg in memory_messages:
                if isinstance(msg.get('content'), list):
                    for content in msg['content']:
                        if content.get('type') == 'text':
                            print(f"Text content: {content.get('text')}")
                            msg['content'] = content.get('text')
                            print(f"Text content: {msg['content']}")
                else:
                    print(f"Text content: {msg.get('content')}")

            print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
            print("---------------------------------------------------------------------------")
            print(memory_messages)
            print("---------------------------------------------------------------------------")


            self.add_to_memory(memory_messages, user_id)
            print(f"Stored {len(memory_messages)} messages in memory for user {user_id}")            

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