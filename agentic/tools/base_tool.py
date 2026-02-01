"""
🔧 BASE TOOL
============
Abstract base class for all tools in the agentic system.

All tools must implement:
- name: Tool identifier
- description: What the tool does
- run(): Execute the tool with given arguments
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Standard result format for tools"""
    success: bool
    data: Any = None
    error: str = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata or {}
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Each tool should:
    1. Have a unique name
    2. Provide a description for the planner
    3. Implement the run() method
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does (for planner)"""
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute the tool.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            Tool output (format depends on tool type)
        """
        pass
    
    def validate_args(self, required: list, kwargs: dict) -> Optional[str]:
        """
        Validate that required arguments are present.
        
        Returns:
            None if valid, error message if invalid
        """
        missing = [arg for arg in required if arg not in kwargs]
        if missing:
            return f"Missing required arguments: {missing}"
        return None
    
    def __str__(self) -> str:
        return f"Tool({self.name})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
