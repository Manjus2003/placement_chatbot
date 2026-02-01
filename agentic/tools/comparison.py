"""
📊 COMPARISON TOOL
==================
Generates comparison tables for multiple companies.

This tool:
1. Takes context from multiple company searches
2. Extracts comparable attributes (CTC, roles, eligibility, process)
3. Generates a formatted comparison table
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .base_tool import BaseTool


class ComparisonTool(BaseTool):
    """
    Generates comparison tables for multiple companies.
    
    Uses LLM to extract and compare key attributes across companies.
    """
    
    def __init__(self):
        """Initialize the comparison tool"""
        load_dotenv(dotenv_path="environment.env")
        
        self.llm_client = None
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
                print("✅ ComparisonTool initialized")
            except Exception as e:
                print(f"⚠️ ComparisonTool: LLM init failed: {e}")
    
    @property
    def name(self) -> str:
        return "comparison"
    
    @property
    def description(self) -> str:
        return "Generate a comparison table for multiple companies based on their placement data"
    
    def run(
        self,
        query: str,
        companies: List[str],
        contexts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comparison table.
        
        Args:
            query: Original user query
            companies: List of company names to compare
            contexts: List of context strings for each company
                     (one context per company, same order)
        
        Returns:
            Dict with:
                - comparison_table: Markdown table
                - structured_data: Dict with extracted data
                - answer: Natural language summary
        """
        print(f"📊 Comparing {len(companies)} companies: {companies}")
        
        if not companies or not contexts:
            return {
                "comparison_table": "",
                "structured_data": {},
                "answer": "No companies or context provided for comparison."
            }
        
        # Ensure contexts list matches companies
        while len(contexts) < len(companies):
            contexts.append("")
        
        if self.llm_client:
            return self._generate_comparison_llm(query, companies, contexts)
        else:
            return self._generate_comparison_rules(query, companies, contexts)
    
    def _generate_comparison_llm(
        self,
        query: str,
        companies: List[str],
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Use LLM to generate comparison"""
        
        # Prepare context for each company
        company_contexts = []
        for i, (company, context) in enumerate(zip(companies, contexts)):
            company_contexts.append(f"### {company}\n{context[:1500]}")
        
        combined_context = "\n\n".join(company_contexts)
        
        prompt = f"""Create a comparison table for these companies based on their placement data.

User Query: {query}

Company Information:
{combined_context}

Generate a comprehensive comparison with these columns:
1. Company Name
2. CTC (Cost to Company)
3. Roles Offered
4. Eligibility (CGPA, branches)
5. Selection Process (rounds)
6. Notable Points

Return the response in this exact JSON format:
{{
    "table": "| Company | CTC | Roles | Eligibility | Process | Notes |\\n|---------|-----|-------|-------------|---------|-------|\\n| ... |",
    "data": {{
        "Company1": {{"ctc": "X LPA", "roles": ["role1"], "eligibility": "...", "process": "...", "notes": "..."}},
        "Company2": {{...}}
    }},
    "summary": "A brief natural language comparison summary"
}}

Extract ONLY information present in the contexts. Use "N/A" for missing data.
Return ONLY valid JSON."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2
            )
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "comparison_table": result.get("table", ""),
                    "structured_data": result.get("data", {}),
                    "answer": result.get("summary", ""),
                    "response": result.get("summary", "")  # Alias for executor
                }
            
        except Exception as e:
            print(f"⚠️ LLM comparison failed: {e}")
        
        return self._generate_comparison_rules(query, companies, contexts)
    
    def _generate_comparison_rules(
        self,
        query: str,
        companies: List[str],
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Rule-based comparison generation"""
        
        data = {}
        
        for company, context in zip(companies, contexts):
            extracted = self._extract_attributes(company, context)
            data[company] = extracted
        
        # Build markdown table
        table_lines = [
            "| Company | CTC | Roles | Eligibility | Process |",
            "|---------|-----|-------|-------------|---------|"
        ]
        
        for company, attrs in data.items():
            row = f"| {company} | {attrs.get('ctc', 'N/A')} | {attrs.get('roles', 'N/A')} | {attrs.get('eligibility', 'N/A')} | {attrs.get('process', 'N/A')} |"
            table_lines.append(row)
        
        table = "\n".join(table_lines)
        
        # Build summary
        summary_parts = [f"Comparison of {', '.join(companies)}:"]
        for company, attrs in data.items():
            if attrs.get('ctc') != 'N/A':
                summary_parts.append(f"- {company}: {attrs.get('ctc', 'N/A')}")
        
        return {
            "comparison_table": table,
            "structured_data": data,
            "answer": "\n".join(summary_parts),
            "response": "\n".join(summary_parts)
        }
    
    def _extract_attributes(self, company: str, context: str) -> Dict[str, str]:
        """Extract key attributes from context using regex"""
        attrs = {
            "ctc": "N/A",
            "roles": "N/A",
            "eligibility": "N/A",
            "process": "N/A"
        }
        
        if not context:
            return attrs
        
        context_lower = context.lower()
        
        # Extract CTC
        ctc_patterns = [
            r'ctc[:\s]+(\d+(?:\.\d+)?\s*(?:lpa|lakhs?|l))',
            r'(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?)',
            r'package[:\s]+(\d+(?:\.\d+)?\s*(?:lpa|lakhs?|l))',
            r'salary[:\s]+(\d+(?:\.\d+)?\s*(?:lpa|lakhs?|l))',
        ]
        
        for pattern in ctc_patterns:
            match = re.search(pattern, context_lower)
            if match:
                attrs["ctc"] = match.group(1).upper().replace("LAKHS", "LPA").replace("L", "LPA")
                break
        
        # Extract roles
        role_patterns = [
            r'role[s]?[:\s]+([^.]+)',
            r'position[s]?[:\s]+([^.]+)',
            r'hiring\s+for[:\s]+([^.]+)',
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, context_lower)
            if match:
                attrs["roles"] = match.group(1).strip()[:50]
                break
        
        # Extract eligibility
        eligibility_patterns = [
            r'cgpa[:\s]+(\d+(?:\.\d+)?)',
            r'eligibility[:\s]+([^.]+)',
            r'minimum[:\s]+(\d+(?:\.\d+)?\s*cgpa)',
        ]
        
        for pattern in eligibility_patterns:
            match = re.search(pattern, context_lower)
            if match:
                attrs["eligibility"] = match.group(1).strip()
                break
        
        # Extract process
        process_patterns = [
            r'(\d+)\s*rounds?',
            r'selection\s+process[:\s]+([^.]+)',
            r'interview\s+process[:\s]+([^.]+)',
        ]
        
        for pattern in process_patterns:
            match = re.search(pattern, context_lower)
            if match:
                attrs["process"] = match.group(0).strip()[:50]
                break
        
        return attrs


# Example usage
if __name__ == "__main__":
    tool = ComparisonTool()
    
    result = tool.run(
        query="Compare Amazon and Google",
        companies=["Amazon", "Google"],
        contexts=[
            "Amazon offers CTC of 28 LPA. Roles: SDE-1. Eligibility: 7.5 CGPA. 4 rounds of interviews.",
            "Google offers CTC of 35 LPA. Roles: SWE. Eligibility: 8.0 CGPA. 5 rounds including onsite."
        ]
    )
    
    print(result["comparison_table"])
    print("\nSummary:", result["answer"])
