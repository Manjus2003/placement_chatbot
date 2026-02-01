"""
🧠 QUERY COMPLEXITY ANALYZER
============================
Automatically determines whether a query needs Basic RAG or Agentic RAG.

Complexity Indicators:
- Multiple entities (companies) → Agentic
- Comparison requests → Agentic  
- Statistical queries → Agentic
- Eligibility checks → Agentic
- Simple single-entity queries → Basic
"""

import re
from typing import Tuple, List, Dict
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"       # Basic RAG is enough
    MODERATE = "moderate"   # Could use either, prefer Agentic
    COMPLEX = "complex"     # Needs Agentic RAG


class QueryAnalyzer:
    """
    Analyzes query complexity to auto-select RAG mode.
    
    Uses pattern matching and heuristics to determine
    if a query needs simple retrieval or multi-step reasoning.
    """
    
    def __init__(self, known_companies: List[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            known_companies: List of known company names for entity detection
        """
        self.known_companies = known_companies or []
        self.known_companies_lower = [c.lower() for c in self.known_companies]
        
        # Patterns that indicate COMPLEX queries (need Agentic RAG)
        self.complex_patterns = {
            # Comparison patterns
            "comparison": [
                r"compare\s+\w+",
                r"\w+\s+vs\.?\s+\w+",
                r"\w+\s+versus\s+\w+",
                r"difference\s+between",
                r"which\s+is\s+better",
                r"better\s+between",
            ],
            
            # Statistical/Aggregation patterns
            "statistics": [
                r"average\s+",
                r"mean\s+",
                r"total\s+",
                r"how\s+many\s+companies",
                r"count\s+of",
                r"number\s+of\s+companies",
                r"highest\s+",
                r"lowest\s+",
                r"maximum\s+",
                r"minimum\s+",
                r"top\s+\d+",
                r"best\s+\d+",
                r"statistics",
                r"trend",
                r"percentage",
            ],
            
            # Eligibility patterns
            "eligibility": [
                r"am\s+i\s+eligible",
                r"can\s+i\s+apply",
                r"eligibility\s+for",
                r"eligible\s+for",
                r"do\s+i\s+qualify",
                r"with\s+\d+\.?\d*\s*cgpa",
                r"my\s+cgpa\s+is",
                r"i\s+have\s+\d+\.?\d*\s*cgpa",
            ],
            
            # Multi-step reasoning patterns
            "multi_step": [
                r"and\s+also",
                r"additionally",
                r"along\s+with",
                r"as\s+well\s+as",
                r"step\s+by\s+step",
                r"first.*then",
            ],
            
            # Ranking patterns
            "ranking": [
                r"rank",
                r"sort\s+by",
                r"order\s+by",
                r"list\s+all",
                r"show\s+all",
                r"all\s+companies",
            ],
        }
        
        # Patterns that indicate SIMPLE queries (Basic RAG is enough)
        self.simple_patterns = [
            r"^what\s+is\s+\w+('s)?\s+\w+\??$",  # "What is Amazon's CTC?"
            r"^tell\s+me\s+about\s+\w+$",        # "Tell me about Amazon"
            r"^(what|how|when|where)\s+.{10,50}$",  # Short simple questions
        ]
    
    def analyze(self, query: str) -> Tuple[QueryComplexity, str, Dict]:
        """
        Analyze query complexity.
        
        Args:
            query: User's query string
            
        Returns:
            Tuple of (complexity, reason, details)
        """
        query_lower = query.lower().strip()
        details = {
            "query": query,
            "company_count": 0,
            "matched_patterns": [],
            "complexity_score": 0
        }
        
        complexity_score = 0
        reasons = []
        
        # ========== CHECK 1: Count company mentions ==========
        company_count = self._count_companies(query_lower)
        details["company_count"] = company_count
        
        if company_count >= 3:
            complexity_score += 3
            reasons.append(f"Multiple companies mentioned ({company_count})")
        elif company_count == 2:
            complexity_score += 2
            reasons.append("Two companies mentioned (likely comparison)")
        
        # ========== CHECK 2: Check complex patterns ==========
        for category, patterns in self.complex_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    complexity_score += 2
                    reasons.append(f"Pattern matched: {category}")
                    details["matched_patterns"].append(category)
                    break  # Only count each category once
        
        # ========== CHECK 3: Check for simple patterns ==========
        is_simple = False
        for pattern in self.simple_patterns:
            if re.match(pattern, query_lower):
                is_simple = True
                complexity_score -= 1
                break
        
        # ========== CHECK 4: Query length heuristic ==========
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 1
            reasons.append("Long query (complex intent)")
        elif word_count < 6 and company_count <= 1:
            complexity_score -= 1
        
        # ========== CHECK 5: Question complexity ==========
        if query_lower.count("?") > 1:
            complexity_score += 1
            reasons.append("Multiple questions")
        
        if " and " in query_lower and company_count >= 1:
            complexity_score += 1
            reasons.append("Compound query with 'and'")
        
        # ========== DETERMINE COMPLEXITY ==========
        details["complexity_score"] = complexity_score
        
        if complexity_score >= 3:
            complexity = QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            complexity = QueryComplexity.MODERATE
        else:
            complexity = QueryComplexity.SIMPLE
        
        reason = "; ".join(reasons) if reasons else "Simple single-topic query"
        
        return complexity, reason, details
    
    def _count_companies(self, query_lower: str) -> int:
        """Count how many known companies are mentioned"""
        count = 0
        for company in self.known_companies_lower:
            if company in query_lower:
                count += 1
        return count
    
    def should_use_agentic(self, query: str) -> Tuple[bool, str]:
        """
        Simple interface: Should this query use Agentic RAG?
        
        Args:
            query: User's query
            
        Returns:
            Tuple of (should_use_agentic, reason)
        """
        complexity, reason, _ = self.analyze(query)
        
        # Use Agentic for COMPLEX and MODERATE queries
        use_agentic = complexity in [QueryComplexity.COMPLEX, QueryComplexity.MODERATE]
        
        return use_agentic, reason
    
    def get_recommendation(self, query: str) -> Dict:
        """
        Get detailed recommendation with explanation.
        
        Returns dict with:
            - recommended_mode: "basic" or "agentic"
            - complexity: SIMPLE/MODERATE/COMPLEX
            - reason: Human-readable explanation
            - confidence: 0-1 confidence score
            - details: Additional analysis details
        """
        complexity, reason, details = self.analyze(query)
        
        score = details["complexity_score"]
        
        # Calculate confidence based on how clear-cut the decision is
        if score >= 4:
            confidence = 0.95
        elif score >= 2:
            confidence = 0.8
        elif score <= -1:
            confidence = 0.9
        else:
            confidence = 0.6  # Borderline case
        
        return {
            "recommended_mode": "agentic" if complexity != QueryComplexity.SIMPLE else "basic",
            "complexity": complexity.value,
            "reason": reason,
            "confidence": confidence,
            "details": details
        }


# ============================================================
# STANDALONE FUNCTION FOR EASY USE
# ============================================================

_analyzer = None

def get_analyzer(known_companies: List[str] = None) -> QueryAnalyzer:
    """Get or create a singleton analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryAnalyzer(known_companies)
    return _analyzer


def auto_select_mode(query: str, known_companies: List[str] = None) -> str:
    """
    Simple function to auto-select RAG mode.
    
    Usage:
        from agentic.query_analyzer import auto_select_mode
        mode = auto_select_mode("Compare Amazon vs Google")
        # Returns: "agentic"
    
    Args:
        query: User's query
        known_companies: Optional list of known company names
        
    Returns:
        "basic" or "agentic"
    """
    analyzer = QueryAnalyzer(known_companies or [])
    use_agentic, _ = analyzer.should_use_agentic(query)
    return "agentic" if use_agentic else "basic"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test with sample queries
    known = ["Amazon", "Google", "Microsoft", "Intel", "Nvidia", "Samsung"]
    analyzer = QueryAnalyzer(known)
    
    test_queries = [
        # SIMPLE - should use Basic RAG
        ("What is Amazon's CTC?", "basic"),
        ("Tell me about Intel", "basic"),
        ("Amazon eligibility criteria", "basic"),
        
        # COMPLEX - should use Agentic RAG
        ("Compare Amazon vs Google vs Microsoft", "agentic"),
        ("What is the average CTC across all companies?", "agentic"),
        ("Am I eligible for Intel with 8.5 CGPA in CSE?", "agentic"),
        ("Show top 5 highest paying companies", "agentic"),
        ("Which company is better between Amazon and Google?", "agentic"),
        ("List all companies with CTC above 20 LPA", "agentic"),
        ("How many companies visited in 2025?", "agentic"),
    ]
    
    print("=" * 70)
    print("QUERY COMPLEXITY ANALYSIS TEST")
    print("=" * 70)
    
    for query, expected in test_queries:
        result = analyzer.get_recommendation(query)
        actual = result["recommended_mode"]
        status = "✅" if actual == expected else "❌"
        
        print(f"\n{status} Query: {query}")
        print(f"   Expected: {expected} | Got: {actual}")
        print(f"   Complexity: {result['complexity']} | Confidence: {result['confidence']:.0%}")
        print(f"   Reason: {result['reason']}")
