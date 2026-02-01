"""
🏢 COMPANY EXTRACTOR TOOL
=========================
Extracts company names from user queries.

This tool:
1. Matches against known company names
2. Uses fuzzy matching for typos
3. Falls back to LLM extraction for unknown companies
"""

import os
import re
from typing import List, Dict, Any, Optional
from difflib import get_close_matches
from dotenv import load_dotenv

from .base_tool import BaseTool


class CompanyExtractorTool(BaseTool):
    """
    Extracts company names from queries.
    
    Uses a combination of:
    1. Exact matching against known companies
    2. Fuzzy matching for misspellings
    3. LLM extraction as fallback
    """
    
    def __init__(self, known_companies: List[str] = None):
        """
        Initialize the company extractor.
        
        Args:
            known_companies: List of known company names.
                            Will be populated from QueryHelper if not provided.
        """
        load_dotenv(dotenv_path="environment.env")
        
        self.known_companies = known_companies or []
        self.known_companies_lower = [c.lower() for c in self.known_companies]
        
        # Common aliases/variations
        self.aliases = {
            "msft": "Microsoft",
            "ms": "Microsoft",
            "amzn": "Amazon",
            "aws": "Amazon",
            "goog": "Google",
            "nvda": "Nvidia",
            "ge": "GE Healthcare",
            "hpe": "Hewlett Packard",
            "hp": "Hewlett Packard",
            "ibm": "IBM",
            "sap": "SAP Labs",
            "st": "STMicroelectronics",
            "stm": "STMicroelectronics",
        }
        
        # LLM client for fallback
        self.llm_client = None
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
            except Exception:
                pass
    
    @property
    def name(self) -> str:
        return "company_extractor"
    
    @property
    def description(self) -> str:
        return "Extract company names from a query"
    
    def set_known_companies(self, companies: List[str]):
        """Update the list of known companies"""
        self.known_companies = companies
        self.known_companies_lower = [c.lower() for c in companies]
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Extract company names from query.
        
        Args:
            query: User query to extract companies from
            
        Returns:
            Dict with:
                - companies: List of extracted company names
                - confidence: Confidence score
                - method: Extraction method used
        """
        print(f"🏢 Extracting companies from: '{query[:50]}...'")
        
        companies = []
        method = "none"
        
        # Step 1: Check aliases
        query_lower = query.lower()
        for alias, company in self.aliases.items():
            if re.search(rf'\b{alias}\b', query_lower):
                companies.append(company)
                method = "alias"
        
        # Step 2: Exact matching
        exact_matches = self._exact_match(query_lower)
        if exact_matches:
            companies.extend(exact_matches)
            method = "exact"
        
        # Step 3: Fuzzy matching for potential typos
        if not companies:
            fuzzy_matches = self._fuzzy_match(query)
            if fuzzy_matches:
                companies.extend(fuzzy_matches)
                method = "fuzzy"
        
        # Step 4: LLM extraction as fallback
        if not companies and self.llm_client:
            llm_companies = self._llm_extract(query)
            if llm_companies:
                companies.extend(llm_companies)
                method = "llm"
        
        # Deduplicate while preserving order
        seen = set()
        unique_companies = []
        for c in companies:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_companies.append(c)
        
        print(f"   Found: {unique_companies} (method: {method})")
        
        return {
            "companies": unique_companies,
            "confidence": 0.9 if method in ["exact", "alias"] else 0.7 if method == "fuzzy" else 0.5,
            "method": method
        }
    
    def _exact_match(self, query_lower: str) -> List[str]:
        """Find exact company name matches"""
        matches = []
        
        for i, company_lower in enumerate(self.known_companies_lower):
            # Check for exact word boundary match
            if re.search(rf'\b{re.escape(company_lower)}\b', query_lower):
                matches.append(self.known_companies[i])
        
        return matches
    
    def _fuzzy_match(self, query: str) -> List[str]:
        """Find fuzzy matches for potential typos"""
        # Extract potential company words (capitalized words or specific patterns)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        matches = []
        for word in words:
            close = get_close_matches(
                word.lower(),
                self.known_companies_lower,
                n=1,
                cutoff=0.8  # 80% similarity
            )
            if close:
                idx = self.known_companies_lower.index(close[0])
                matches.append(self.known_companies[idx])
        
        return matches
    
    def _llm_extract(self, query: str) -> List[str]:
        """Use LLM to extract company names"""
        if not self.llm_client:
            return []
        
        # Provide some known companies as examples
        examples = self.known_companies[:15] if self.known_companies else [
            "Amazon", "Google", "Microsoft", "Intel", "Nvidia", "Samsung"
        ]
        
        prompt = f"""Extract company names from this placement-related query.

Query: "{query}"

Known companies in our database: {', '.join(examples)}

Return ONLY a JSON array of company names mentioned or implied.
Example: ["Amazon", "Google"]
If no companies are mentioned, return: []

Return ONLY the JSON array, nothing else."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            
            # Extract JSON array
            match = re.search(r'\[.*?\]', result)
            if match:
                import json
                companies = json.loads(match.group())
                # Validate against known companies
                validated = []
                for c in companies:
                    if c.lower() in self.known_companies_lower:
                        idx = self.known_companies_lower.index(c.lower())
                        validated.append(self.known_companies[idx])
                    else:
                        # Keep it even if not in known list
                        validated.append(c)
                return validated
                
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
        
        return []


# Example usage
if __name__ == "__main__":
    known = ["Amazon", "Google", "Microsoft", "Intel", "Nvidia", "Samsung", 
             "SAP Labs", "GE Healthcare", "Honeywell", "Bosch"]
    
    tool = CompanyExtractorTool(known_companies=known)
    
    test_queries = [
        "What is Amazon's CTC?",
        "Compare Google vs Microsoft",
        "Tell me about amzn placement",  # alias
        "Amazn interview process",  # typo
        "Which companies have highest CTC?"  # no specific company
    ]
    
    for query in test_queries:
        result = tool.run(query)
        print(f"  → {result['companies']}\n")
