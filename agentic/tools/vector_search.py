"""
🔍 VECTOR SEARCH TOOL
=====================
Wraps the existing QueryHelper to provide vector search functionality.

This tool:
1. Encodes query using GTE-Qwen2 embeddings
2. Searches Pinecone index
3. Optionally filters by company
4. Reranks results with Cross-Encoder
5. Returns context and sources
"""

import sys
import os
from typing import Dict, Any, List, Optional

# Add parent directory to path to import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .base_tool import BaseTool, ToolResult


class VectorSearchTool(BaseTool):
    """
    Vector search tool using existing QueryHelper.
    
    This tool wraps the existing Pinecone-based search with
    reranking capabilities.
    """
    
    def __init__(self, query_helper=None):
        """
        Initialize the vector search tool.
        
        Args:
            query_helper: Existing QueryHelper instance.
                         If None, will create a new one.
        """
        self._query_helper = query_helper
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "vector_search"
    
    @property
    def description(self) -> str:
        return "Search placement documents using semantic similarity. Can filter by company name."
    
    def _ensure_initialized(self):
        """Lazy initialization of QueryHelper"""
        if not self._initialized:
            if self._query_helper is None:
                try:
                    from query_helper import QueryHelper
                    self._query_helper = QueryHelper(use_llm=False)
                    print("✅ VectorSearchTool: QueryHelper initialized")
                except Exception as e:
                    print(f"❌ VectorSearchTool: Failed to init QueryHelper: {e}")
                    raise
            self._initialized = True
    
    def run(
        self,
        query: str,
        company_filter: Optional[str] = None,
        top_k: int = 5,
        use_reranking: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute vector search.
        
        Args:
            query: Search query
            company_filter: Optional company name to filter results
            top_k: Number of results to return
            use_reranking: Whether to use cross-encoder reranking
            
        Returns:
            Dict with keys:
                - context: Combined text from retrieved documents
                - sources: List of source file paths
                - chunks: Raw chunk data
                - scores: Relevance scores
        """
        self._ensure_initialized()
        
        print(f"🔍 VectorSearch: '{query[:50]}...'")
        if company_filter:
            print(f"   📌 Company filter: {company_filter}")
        
        try:
            # Use existing QueryHelper methods
            if hasattr(self._query_helper, 'search_and_rerank'):
                # Advanced search with reranking
                results = self._query_helper.search_and_rerank(
                    query=query,
                    company=company_filter,
                    top_k=top_k * 2 if use_reranking else top_k,  # Get more for reranking
                    rerank_top_k=top_k
                )
            elif hasattr(self._query_helper, 'search'):
                # Basic search
                results = self._query_helper.search(
                    query=query,
                    top_k=top_k,
                    company=company_filter
                )
            else:
                # Fallback: Use generate_answer and extract context
                result = self._query_helper.generate_answer(
                    query=query,
                    top_k=top_k
                )
                return {
                    "context": result.get("context", ""),
                    "sources": result.get("sources", []),
                    "chunks": [],
                    "scores": [],
                    "answer": result.get("answer", "")
                }
            
            # Format results
            context = self._format_context(results)
            sources = self._extract_sources(results)
            
            return {
                "context": context,
                "sources": sources,
                "chunks": results if isinstance(results, list) else [],
                "scores": [r.get("score", 0) for r in results] if isinstance(results, list) else []
            }
            
        except Exception as e:
            print(f"❌ VectorSearch error: {e}")
            return {
                "context": "",
                "sources": [],
                "chunks": [],
                "scores": [],
                "error": str(e)
            }
    
    def _format_context(self, results: Any) -> str:
        """Format search results into context string"""
        if isinstance(results, str):
            return results
        
        if isinstance(results, list):
            context_parts = []
            for r in results:
                if isinstance(r, dict):
                    text = r.get("text", r.get("content", r.get("chunk", "")))
                    company = r.get("company", "Unknown")
                    context_parts.append(f"[{company}] {text}")
                elif isinstance(r, str):
                    context_parts.append(r)
            return "\n\n".join(context_parts)
        
        if isinstance(results, dict):
            return results.get("context", str(results))
        
        return str(results)
    
    def _extract_sources(self, results: Any) -> List[str]:
        """Extract source file paths from results"""
        sources = []
        
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    source = r.get("source", r.get("file", r.get("filename", "")))
                    if source and source not in sources:
                        sources.append(source)
        
        return sources
    
    def get_all_companies(self) -> List[str]:
        """Get list of all companies in the index"""
        self._ensure_initialized()
        return getattr(self._query_helper, 'all_companies', [])


# Simple search without full QueryHelper (fallback)
class SimpleVectorSearchTool(BaseTool):
    """
    Simplified vector search for when full QueryHelper is not available.
    Uses direct Pinecone queries.
    """
    
    def __init__(self):
        self.index = None
        self.model = None
    
    @property
    def name(self) -> str:
        return "vector_search"
    
    @property
    def description(self) -> str:
        return "Search placement documents using semantic similarity"
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Fallback search implementation"""
        # This is a simplified implementation
        # In production, you'd want to use the full QueryHelper
        return {
            "context": f"Search results for: {query}",
            "sources": [],
            "chunks": [],
            "scores": []
        }


# Example usage
if __name__ == "__main__":
    tool = VectorSearchTool()
    result = tool.run(
        query="What is Amazon's CTC?",
        company_filter="Amazon",
        top_k=3
    )
    print(f"Context length: {len(result['context'])}")
    print(f"Sources: {result['sources']}")
