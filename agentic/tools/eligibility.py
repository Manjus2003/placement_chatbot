"""
✅ ELIGIBILITY TOOL
===================
Checks if a student meets company eligibility criteria.

This tool:
1. Parses student profile from query
2. Extracts eligibility criteria from context
3. Compares and returns eligibility status
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .base_tool import BaseTool


class EligibilityTool(BaseTool):
    """
    Checks student eligibility for company placements.
    
    Parses student profile (CGPA, branch, skills) and compares
    against company requirements.
    """
    
    def __init__(self):
        """Initialize the eligibility tool"""
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
                print("✅ EligibilityTool initialized")
            except Exception as e:
                print(f"⚠️ EligibilityTool: LLM init failed: {e}")
    
    @property
    def name(self) -> str:
        return "eligibility"
    
    @property
    def description(self) -> str:
        return "Check if a student is eligible for a company based on CGPA, branch, and other criteria"
    
    def run(
        self,
        query: str,
        criteria_context: str = "",
        company: str = None,
        student_profile: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check eligibility.
        
        Args:
            query: Original user query (may contain student info)
            criteria_context: Context containing eligibility criteria
            company: Company name
            student_profile: Optional pre-parsed student profile
            
        Returns:
            Dict with:
                - eligible: Boolean or "Maybe"
                - explanation: Detailed explanation
                - criteria: Extracted criteria
                - profile: Parsed student profile
        """
        print(f"✅ Checking eligibility for: {company or 'company'}")
        
        # Parse student profile from query if not provided
        if not student_profile:
            student_profile = self._parse_student_profile(query)
        
        # Extract criteria from context
        criteria = self._extract_criteria(criteria_context, company)
        
        # Check eligibility
        if self.llm_client:
            result = self._check_eligibility_llm(
                query, student_profile, criteria, company, criteria_context
            )
        else:
            result = self._check_eligibility_rules(student_profile, criteria)
        
        result["profile"] = student_profile
        result["criteria"] = criteria
        
        return result
    
    def _parse_student_profile(self, query: str) -> Dict[str, Any]:
        """Parse student information from query"""
        profile = {
            "cgpa": None,
            "branch": None,
            "skills": [],
            "experience": None,
            "backlogs": 0
        }
        
        query_lower = query.lower()
        
        # Extract CGPA
        cgpa_patterns = [
            r'cgpa[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*cgpa',
            r'gpa[:\s]+(\d+(?:\.\d+)?)',
            r'(?:my|i\s+have)\s+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in cgpa_patterns:
            match = re.search(pattern, query_lower)
            if match:
                cgpa = float(match.group(1))
                if cgpa <= 10:  # Valid CGPA range
                    profile["cgpa"] = cgpa
                    break
        
        # Extract branch
        branches = {
            "cs": "Computer Science",
            "cse": "Computer Science",
            "computer science": "Computer Science",
            "ece": "Electronics",
            "electronics": "Electronics",
            "ee": "Electrical",
            "electrical": "Electrical",
            "it": "Information Technology",
            "information technology": "Information Technology",
            "me": "Mechanical",
            "mechanical": "Mechanical",
            "data science": "Data Science",
            "ai": "Artificial Intelligence",
            "ml": "Machine Learning",
        }
        
        for key, value in branches.items():
            if key in query_lower:
                profile["branch"] = value
                break
        
        # Extract skills
        skill_keywords = [
            "python", "java", "c++", "javascript", "sql", "machine learning",
            "deep learning", "data science", "web development", "cloud",
            "aws", "azure", "docker", "kubernetes", "react", "angular"
        ]
        
        for skill in skill_keywords:
            if skill in query_lower:
                profile["skills"].append(skill.title())
        
        # Extract backlogs
        backlog_match = re.search(r'(\d+)\s*backlogs?', query_lower)
        if backlog_match:
            profile["backlogs"] = int(backlog_match.group(1))
        
        # Check for "no backlogs"
        if "no backlog" in query_lower or "0 backlog" in query_lower:
            profile["backlogs"] = 0
        
        return profile
    
    def _extract_criteria(self, context: str, company: str = None) -> Dict[str, Any]:
        """Extract eligibility criteria from context"""
        criteria = {
            "min_cgpa": None,
            "branches": [],
            "backlogs_allowed": None,
            "skills_required": [],
            "other": []
        }
        
        if not context:
            return criteria
        
        context_lower = context.lower()
        
        # Extract minimum CGPA
        cgpa_patterns = [
            r'(?:minimum|min)?\s*cgpa[:\s]+(\d+(?:\.\d+)?)',
            r'eligibility[:\s]+(\d+(?:\.\d+)?)\s*(?:cgpa)?',
            r'(\d+(?:\.\d+)?)\s*(?:cgpa|and above)',
            r'above\s+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in cgpa_patterns:
            match = re.search(pattern, context_lower)
            if match:
                cgpa = float(match.group(1))
                if cgpa <= 10:
                    criteria["min_cgpa"] = cgpa
                    break
        
        # Extract branches
        branch_keywords = ["cs", "cse", "ece", "ee", "it", "me", "all branches", "any branch"]
        for branch in branch_keywords:
            if branch in context_lower:
                criteria["branches"].append(branch.upper())
        
        # Check for backlogs policy
        if "no backlog" in context_lower or "zero backlog" in context_lower:
            criteria["backlogs_allowed"] = 0
        elif "backlog" in context_lower:
            backlog_match = re.search(r'(\d+)\s*(?:active\s*)?backlogs?\s*(?:allowed)?', context_lower)
            if backlog_match:
                criteria["backlogs_allowed"] = int(backlog_match.group(1))
        
        return criteria
    
    def _check_eligibility_llm(
        self,
        query: str,
        profile: Dict[str, Any],
        criteria: Dict[str, Any],
        company: str,
        context: str
    ) -> Dict[str, Any]:
        """Use LLM to check eligibility"""
        
        prompt = f"""Check if this student is eligible for {company or 'the company'}.

Student Profile:
- CGPA: {profile.get('cgpa', 'Not specified')}
- Branch: {profile.get('branch', 'Not specified')}
- Skills: {', '.join(profile.get('skills', [])) or 'Not specified'}
- Active Backlogs: {profile.get('backlogs', 'Not specified')}

Eligibility Criteria from company documents:
{context[:1500]}

Answer in this JSON format:
{{
    "eligible": true/false/"maybe",
    "explanation": "Clear explanation of eligibility status",
    "matching_criteria": ["criteria met"],
    "missing_criteria": ["criteria not met"],
    "recommendation": "What student should do"
}}

Return ONLY valid JSON."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "eligible": result.get("eligible", "maybe"),
                    "explanation": result.get("explanation", ""),
                    "matching_criteria": result.get("matching_criteria", []),
                    "missing_criteria": result.get("missing_criteria", []),
                    "recommendation": result.get("recommendation", ""),
                    "answer": result.get("explanation", ""),
                    "response": result.get("explanation", "")
                }
                
        except Exception as e:
            print(f"⚠️ LLM eligibility check failed: {e}")
        
        return self._check_eligibility_rules(profile, criteria)
    
    def _check_eligibility_rules(
        self,
        profile: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rule-based eligibility check"""
        
        eligible = True
        matching = []
        missing = []
        explanations = []
        
        # Check CGPA
        student_cgpa = profile.get("cgpa")
        min_cgpa = criteria.get("min_cgpa")
        
        if min_cgpa and student_cgpa:
            if student_cgpa >= min_cgpa:
                matching.append(f"CGPA {student_cgpa} >= {min_cgpa} ✓")
            else:
                eligible = False
                missing.append(f"CGPA {student_cgpa} < {min_cgpa} required ✗")
        elif min_cgpa and not student_cgpa:
            explanations.append(f"Minimum CGPA required: {min_cgpa}")
        
        # Check backlogs
        student_backlogs = profile.get("backlogs", 0)
        allowed_backlogs = criteria.get("backlogs_allowed")
        
        if allowed_backlogs is not None:
            if student_backlogs <= allowed_backlogs:
                matching.append(f"Backlogs: {student_backlogs} (max {allowed_backlogs}) ✓")
            else:
                eligible = False
                missing.append(f"Has {student_backlogs} backlogs, max {allowed_backlogs} allowed ✗")
        
        # Build explanation
        if not matching and not missing:
            explanation = "Could not determine eligibility - insufficient information."
            eligible_status = "maybe"
        else:
            explanation_parts = []
            if matching:
                explanation_parts.append("Meets: " + ", ".join(matching))
            if missing:
                explanation_parts.append("Does not meet: " + ", ".join(missing))
            explanation = ". ".join(explanation_parts)
            eligible_status = eligible
        
        return {
            "eligible": eligible_status,
            "explanation": explanation,
            "matching_criteria": matching,
            "missing_criteria": missing,
            "recommendation": "Apply if you meet all criteria" if eligible else "Check if you can improve CGPA or clear backlogs",
            "answer": explanation,
            "response": explanation
        }


# Example usage
if __name__ == "__main__":
    tool = EligibilityTool()
    
    result = tool.run(
        query="Am I eligible for Amazon with 8.5 CGPA in Computer Science?",
        criteria_context="Amazon eligibility: Minimum 7.5 CGPA, CSE/ECE branches, No active backlogs",
        company="Amazon"
    )
    
    print(f"Eligible: {result['eligible']}")
    print(f"Explanation: {result['explanation']}")
