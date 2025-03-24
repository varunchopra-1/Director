import os
from typing import Optional

class Mem0ClientSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls, api_key: Optional[str] = None):
        if cls._instance is None:
            from mem0 import MemoryClient
            api_key = api_key or os.getenv("MEM0_API_KEY")
            if not api_key:
                raise ValueError("MEM0_API_KEY not set")
            cls._instance = MemoryClient(api_key=api_key)
            print("Mem0 client initialized once as singleton")
        return cls._instance

def get_memory_client(api_key: Optional[str] = None):
    """Get the memory client singleton instance"""
    return Mem0ClientSingleton.get_instance(api_key)