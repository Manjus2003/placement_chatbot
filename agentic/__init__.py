"""
Agentic RAG Module for Placement Companion Chatbot

This module implements a Planner-Executor-Critic architecture for
intelligent query handling with multi-step reasoning.

Components:
- Planner: Query understanding, classification, and task decomposition
- Executor: Tool orchestration and execution management
- Critic: Answer evaluation, hallucination checking, and refinement decisions
- Tools: Modular tools for various operations (search, SQL, comparison, etc.)
- Memory: Conversation memory and context resolution
"""

from .planner import Planner, Plan, Step
from .executor import Executor, ExecutionResult
from .critic import Critic, Evaluation
from .agentic_rag import AgenticRAG
from .memory_resolver import MemoryResolver
from .entity_extractor import EntityExtractor

__all__ = [
    "Planner",
    "Plan", 
    "Step",
    "Executor",
    "ExecutionResult",
    "Critic",
    "Evaluation",
    "AgenticRAG",
    "MemoryResolver",
    "EntityExtractor"
]

__version__ = "1.0.0"
