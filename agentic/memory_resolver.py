"""
🧠 CONVERSATION MEMORY RESOLVER
================================
Resolves ambiguous queries using conversation context.

Handles:
- Pronoun resolution (their, its, it, they)
- Implicit references ("What about the salary?")
- Comparison shortcuts ("Compare with Amazon")
- Contextual query expansion
"""

import re
from typing import Dict, List, Optional, Tuple


class MemoryResolver:
    """
    Resolves ambiguous queries using conversation memory.
    """
    
    def __init__(self):
        """Initialize the resolver"""
        self.pronouns = ["their", "its", "it", "they", "them", "this", "that", "these"]
        self.comparison_phrases = [
            "compare with", "compare to", "versus", "vs",
            "difference with", "different from"
        ]
    
    def resolve_query(self, query: str, memory: Dict) -> Tuple[str, bool]:
        """
        Resolve ambiguous query using conversation context.
        
        Args:
            query: Original user query
            memory: Conversation memory dictionary
            
        Returns:
            Tuple of (resolved_query, was_modified)
        """
        original_query = query
        query_lower = query.lower().strip()
        
        # Get context
        last_company = memory.get("context", {}).get("last_company")
        companies = memory.get("entities", {}).get("companies", [])
        cgpa = memory.get("entities", {}).get("cgpa")
        branch = memory.get("entities", {}).get("branch")
        
        # Skip if no context available
        if not last_company and not companies:
            return query, False
        
        # ========== PRONOUN RESOLUTION ==========
        query = self._resolve_pronouns(query, query_lower, last_company)
        
        # ========== IMPLICIT REFERENCES ==========
        query = self._resolve_implicit_references(query, query_lower, last_company)
        
        # ========== COMPARISON SHORTCUTS ==========
        query = self._resolve_comparison_shortcuts(query, query_lower, last_company)
        
        # ========== ELIGIBILITY CONTEXT ==========
        query = self._add_eligibility_context(query, query_lower, cgpa, branch)
        
        # ========== FOLLOW-UP QUESTIONS ==========
        query = self._resolve_follow_up(query, query_lower, last_company)
        
        was_modified = (query != original_query)
        
        return query, was_modified
    
    def _resolve_pronouns(self, query: str, query_lower: str, last_company: Optional[str]) -> str:
        """Resolve pronouns to company names"""
        if not last_company:
            return query
        
        # Pattern matching for pronouns
        patterns = [
            (r'\btheir\b', f"{last_company}'s"),
            (r'\bits\b', f"{last_company}'s"),
            (r'\bthey\b', last_company),
            (r'\bthem\b', last_company),
            (r'\bthis company\b', last_company),
            (r'\bthat company\b', last_company),
        ]
        
        for pattern, replacement in patterns:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def _resolve_implicit_references(self, query: str, query_lower: str, last_company: Optional[str]) -> str:
        """Resolve implicit references like 'What about the salary?'"""
        if not last_company:
            return query
        
        # Patterns that indicate implicit reference
        implicit_patterns = [
            (r'^what about (.+)', f"What about {last_company}'s {{1}}"),
            (r'^tell me about (.+)', f"Tell me about {last_company}'s {{1}}"),
            (r'^how about (.+)', f"How about {last_company}'s {{1}}"),
            (r'^what is the (.+)', f"What is {last_company}'s {{1}}"),
        ]
        
        for pattern, replacement in implicit_patterns:
            match = re.match(pattern, query_lower)
            if match:
                # Check if company is not already mentioned
                if last_company.lower() not in query_lower:
                    query = replacement.format(match.group(1))
                break
        
        return query
    
    def _resolve_comparison_shortcuts(self, query: str, query_lower: str, last_company: Optional[str]) -> str:
        """Resolve comparison shortcuts like 'Compare with Amazon'"""
        if not last_company:
            return query
        
        # Check for comparison phrases
        for phrase in self.comparison_phrases:
            if query_lower.startswith(phrase):
                # Extract the second company
                second_company = query[len(phrase):].strip()
                if second_company and last_company.lower() not in query_lower:
                    query = f"Compare {last_company} with {second_company}"
                break
        
        return query
    
    def _add_eligibility_context(self, query: str, query_lower: str, cgpa: Optional[float], branch: Optional[str]) -> str:
        """Add user's CGPA and branch to eligibility queries"""
        # Check if it's an eligibility question
        eligibility_keywords = ["eligible", "eligibility", "can i apply", "qualify", "am i"]
        
        if not any(kw in query_lower for kw in eligibility_keywords):
            return query
        
        # Add context if available
        context_parts = []
        if cgpa and "cgpa" not in query_lower:
            context_parts.append(f"with {cgpa} CGPA")
        if branch and branch.lower() not in query_lower:
            context_parts.append(f"from {branch}")
        
        if context_parts:
            query = f"{query} {' '.join(context_parts)}"
        
        return query
    
    def _resolve_follow_up(self, query: str, query_lower: str, last_company: Optional[str]) -> str:
        """Resolve simple follow-up questions"""
        if not last_company:
            return query
        
        # Very short questions likely refer to last context
        follow_up_patterns = [
            r'^(and|also|what about)?\s*(the|their)?\s*(ctc|salary|package)\??$',
            r'^(and|also|what about)?\s*(the|their)?\s*(eligibility|criteria|requirement)\??$',
            r'^(and|also|what about)?\s*(the|their)?\s*(process|interview|rounds)\??$',
            r'^(and|also|what about)?\s*(the|their)?\s*(roles|positions)\??$',
        ]
        
        for pattern in follow_up_patterns:
            if re.match(pattern, query_lower):
                if last_company.lower() not in query_lower:
                    # Extract the main topic
                    topic = re.sub(r'^(and|also|what about)?\s*(the|their)?\s*', '', query_lower)
                    query = f"What is {last_company}'s {topic}?"
                break
        
        return query
    
    def should_use_memory(self, query: str) -> bool:
        """
        Determine if query needs memory resolution.
        
        Returns True if query contains ambiguous references.
        """
        query_lower = query.lower()
        
        # Check for pronouns
        if any(pronoun in query_lower for pronoun in self.pronouns):
            return True
        
        # Check for implicit references
        implicit_starters = ["what about", "tell me about", "how about"]
        if any(query_lower.startswith(phrase) for phrase in implicit_starters):
            return True
        
        # Check for comparison shortcuts
        if any(query_lower.startswith(phrase) for phrase in self.comparison_phrases):
            return True
        
        # Check for very short queries (likely follow-ups)
        if len(query.split()) <= 3:
            return True
        
        return False
    
    def get_context_summary(self, memory: Dict) -> str:
        """
        Get a human-readable summary of current context.
        
        Useful for debugging or showing to user.
        """
        last_company = memory.get("context", {}).get("last_company")
        companies = memory.get("entities", {}).get("companies", [])
        cgpa = memory.get("entities", {}).get("cgpa")
        branch = memory.get("entities", {}).get("branch")
        query_count = memory.get("context", {}).get("query_count", 0)
        
        summary_parts = []
        
        if last_company:
            summary_parts.append(f"Current topic: {last_company}")
        
        if len(companies) > 1:
            summary_parts.append(f"Discussed: {', '.join(companies[:5])}")
        
        if cgpa:
            summary_parts.append(f"Your CGPA: {cgpa}")
        
        if branch:
            summary_parts.append(f"Branch: {branch}")
        
        if query_count:
            summary_parts.append(f"Queries: {query_count}")
        
        return " | ".join(summary_parts) if summary_parts else "No context yet"


# Example usage
if __name__ == "__main__":
    # Test the resolver
    resolver = MemoryResolver()
    
    memory = {
        "entities": {
            "companies": ["Intel", "AMD"],
            "cgpa": 8.5,
            "branch": "ECE"
        },
        "context": {
            "last_company": "Intel",
            "query_count": 3
        }
    }
    
    test_queries = [
        "What's their CTC?",
        "Compare with AMD",
        "Am I eligible?",
        "What about the interview process?",
        "Tell me about salary"
    ]
    
    print("Testing Memory Resolver\n" + "="*50)
    for query in test_queries:
        resolved, modified = resolver.resolve_query(query, memory)
        if modified:
            print(f"Original:  {query}")
            print(f"Resolved:  {resolved}")
            print()
