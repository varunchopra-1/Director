import os
import json
from enum import Enum
from typing import List, Dict, Any, Optional
import time

from pydantic import Field, field_validator, FieldValidationInfo

from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import (
    LLMType,
)


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT4o = "gpt-4o-2024-11-20"


class VideoDBProxyConfig(BaseLLMConfig):
    """VideoDBProxy Config with memory functionality"""

    llm_type: str = LLMType.VIDEODB_PROXY
    api_key: str = os.getenv("VIDEO_DB_API_KEY")
    api_base: str = os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")
    chat_model: str = Field(default=OpenAIChatModel.GPT4o)
    max_tokens: int = 4096
    
    mongodb_uri: str = os.getenv("MONGODB_URI")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "director_memory")
    mongodb_collection: str = os.getenv("MONGODB_COLLECTION", "chat_memory")

    embedding_api_key: str = os.getenv("EMBEDDING_OPENAI_API_KEY")
    embedding_model: str = "text-embedding-ada-002"
    memory_limit: int = 5  
    similarity_threshold: float = 0.75

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError("Please set VIDEO_DB_API_KEY environment variable.")
        return v
    
    @field_validator("mongodb_uri")
    @classmethod
    def validate_mongodb_uri(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError("Please set MONGODB_URI environment variable.")
        return v


class VideoDBProxy(BaseLLM):
    def __init__(self, config: VideoDBProxyConfig = None):
        """
        Initialize the VideoDBProxy with memory capabilities
        
        :param config: VideoDBProxy Config
        """
        if config is None:
            config = VideoDBProxyConfig()
        super().__init__(config=config)
        
        try:
            import openai
            import pymongo
            from pymongo import MongoClient
        except ImportError:
            missing_libs = []
            try:
                import openai
            except ImportError:
                missing_libs.append("openai")
            try:
                import pymongo
            except ImportError:
                missing_libs.append("pymongo")
            
            raise ImportError(f"Please install required libraries: {', '.join(missing_libs)}")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=f"{self.api_base}")

        self.embeddings_client = openai.OpenAI(api_key=self.config.embedding_api_key)
        
        self.mongo_client = MongoClient(self.config.mongodb_uri)
        self.db = self.mongo_client[self.config.mongodb_db_name]
        self.memory_collection = self.db[self.config.mongodb_collection]
        
        self._setup_vector_index()

    def _setup_vector_index(self):
        """
        Set up vector index in MongoDB Atlas if it doesn't exist
        """
        try:
            from pymongo.operations import SearchIndexModel
            
            existing_indexes = list(self.memory_collection.list_search_indexes())
            index_exists = any(idx.get('name') == 'vector_index' for idx in existing_indexes)
            
            if not index_exists:
                search_index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 1536,
                                "similarity": "dotProduct",
                                "quantization": "scalar"
                            }
                        ]
                    },
                    name="vector_index",
                    type="vectorSearch"
                )
                self.memory_collection.create_search_index(model=search_index_model)
                print("Vector index created successfully")
            else:
                print("Vector index already exists")
        except Exception as e:
            print(f"Error setting up vector index: {e}")

    def _create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text using OpenAI's embedding model
        
        :param text: Text to create embedding for
        :return: Embedding vector
        """
        try:
            print(text)
            response = self.embeddings_client.embeddings.create(
                input=text,
                model=self.config.embedding_model
            )
            print(response.data[0].embedding)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []

    def _extract_insights(self, message: str) -> str:
        """
        Extract key insights from a message using LLM
        
        :param message: Original user message
        :return: Extracted insights
        """
        insights_prompt = [
            {"role": "system", "content": "Extract the key information, topics, and entities from the following message. Focus on identifying main concepts that would be useful for retrieving relevant past conversations. Be concise."},
            {"role": "user", "content": message}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=insights_prompt,
                temperature=0.3,
                max_tokens=256
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error extracting insights: {e}")
            return message

    def _store_conversation(self, user_message: str, assistant_response: str, conversation_id: str = None):
        """
        Store conversation in MongoDB with vector embeddings
        
        :param user_message: User's message
        :param assistant_response: Assistant's response
        :param conversation_id: Optional conversation ID for threading
        """
        insights = self._extract_insights(user_message)
        
        embedding = self._create_embedding(insights)
        
        conversation = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "insights": insights,
            "embedding": embedding,
            "conversation_id": conversation_id,
            "timestamp": time.time()
        }
        
        try:
            self.memory_collection.insert_one(conversation)
        except Exception as e:
            print(f"Error storing conversation: {e}")

    def _retrieve_relevant_context(self, message: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past conversations using vector similarity search
        
        :param message: Current user message
        :return: List of relevant conversation entries
        """
        insights = self._extract_insights(message)
        
        query_embedding = self._create_embedding(insights)
        
        relevant_conversations = []
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "vector_index",
                        "vectorSearch": {
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": 20,
                            "limit": self.config.memory_limit
                        }
                    }
                },
                {
                    "$project": {
                        "user_message": 1,
                        "assistant_response": 1,
                        "insights": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
            
            search_results = list(self.memory_collection.aggregate(pipeline))
            
            relevant_conversations = [
                result for result in search_results 
                if result.get("score", 0) >= self.config.similarity_threshold
            ]
        except Exception as e:
            print(f"Error retrieving relevant context: {e}")
        
        return relevant_conversations

    def _format_context_for_prompt(self, relevant_conversations: List[Dict[str, Any]]) -> str:
        """
        Format retrieved conversations into context for the prompt
        
        :param relevant_conversations: List of relevant conversation entries
        :return: Formatted context string
        """
        if not relevant_conversations:
            return ""
        
        context_parts = ["Here is some relevant context from previous conversations:"]
        
        for idx, conv in enumerate(relevant_conversations):
            context_parts.append(f"Conversation {idx+1}:")
            context_parts.append(f"User: {conv.get('user_message', '')}")
            context_parts.append(f"Assistant: {conv.get('assistant_response', '')}")
            context_parts.append("")
        
        return "\n".join(context_parts)

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

    def chat_completions(
        self, messages: list, tools: list = [], stop=None, response_format=None,
        conversation_id: str = None, enable_memory: bool = True
    ):
        """
        Get completions for chat with memory enhancement.
        
        :param messages: List of messages in the conversation
        :param tools: List of tools available to the model
        :param stop: Stop sequences
        :param response_format: Format for the response
        :param conversation_id: Optional conversation ID for threading
        :param enable_memory: Whether to use memory functionality
        :return: LLM response
        """
        user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
        print("User Message: ", user_message)
        
        if enable_memory and user_message:
            relevant_context = self._retrieve_relevant_context(user_message)
            
            if relevant_context:
                context_str = self._format_context_for_prompt(relevant_context)
                
                system_msg_idx = next((i for i, msg in enumerate(messages) if msg["role"] == "system"), None)
                
                if system_msg_idx is not None:
                    messages[system_msg_idx]["content"] += f"\n\n{context_str}"
                else:
                    messages.insert(0, {"role": "system", "content": context_str})


        params = {
            "model": self.chat_model,
            "messages": self._format_messages(messages),
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
            
            response_content = response.choices[0].message.content or ""
            
            # Store conversation in memory if enabled
            if enable_memory and user_message:
                self._store_conversation(
                    user_message=user_message,
                    assistant_response=response_content,
                    conversation_id=conversation_id
                )
            
            return LLMResponse(
                content=response_content,
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
            
        except Exception as e:
            print(f"Error: {e}")
            return LLMResponse(content=f"Error: {e}")