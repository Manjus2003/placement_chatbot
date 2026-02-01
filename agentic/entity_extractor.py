"""
🔍 ENTITY EXTRACTOR
===================
Extracts entities from conversation Q&A pairs.

Extracts:
- Company names
- CGPA values
- Branch/Department names
- Skills mentioned
- Topics discussed
"""

import re
from typing import Dict, List, Optional, Set


class EntityExtractor:
    """
    Extracts structured entities from conversation text.
    """
    
    def __init__(self, known_companies: List[str] = None):
        """
        Initialize the entity extractor.
        
        Args:
            known_companies: List of known company names
        """
        self.known_companies = known_companies or []
        self.known_companies_lower = [c.lower() for c in self.known_companies]
        
        # Common branches in engineering
        self.branches = [
            "CSE", "Computer Science", "CS",
            "ECE", "Electronics and Communication", "EC",
            "EEE", "Electrical and Electronics", "EE", "Electrical",
            "Mechanical", "Mech", "ME",
            "Civil", "CE",
            "IT", "Information Technology",
            "IS", "Information Science",
            "AIML", "AI", "Artificial Intelligence", "Machine Learning",
            "Data Science", "DS"
        ]
        
        # Common skills
        self.skills = [
            "Python", "Java", "C++", "JavaScript", "React", "Node.js",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "DSA", "Data Structures", "Algorithms",
            "SQL", "MongoDB", "PostgreSQL",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes",
            "VLSI", "Embedded Systems", "IoT"
        ]
    
    def extract_from_conversation(self, query: str, answer: str) -> Dict:
        """
        Extract entities from Q&A pair.
        
        Args:
            query: User's question
            answer: Bot's response
            
        Returns:
            Dictionary with extracted entities
        """
        text = f"{query} {answer}"
        text_lower = text.lower()
        
        entities = {
            "companies": self._extract_companies(text, text_lower),
            "cgpa": self._extract_cgpa(text_lower),
            "branch": self._extract_branch(text, text_lower),
            "skills": self._extract_skills(text, text_lower),
            "topic": self._extract_topic(query.lower())
        }
        
        return entities
    
    def _extract_companies(self, text: str, text_lower: str) -> List[str]:
        """Extract company names from text"""
        found_companies = []
        
        # Check against known companies
        for i, company in enumerate(self.known_companies):
            company_lower = self.known_companies_lower[i]
            if company_lower in text_lower:
                # Avoid duplicates
                if company not in found_companies:
                    found_companies.append(company)
        
        return found_companies
    
    def _extract_cgpa(self, text_lower: str) -> Optional[float]:
        """Extract CGPA value from text"""
        # Pattern: "8.5 CGPA", "CGPA of 8.5", "my CGPA is 8.5"
        patterns = [
            r'(\d+\.?\d*)\s*cgpa',
            r'cgpa\s*(?:of|is|:)?\s*(\d+\.?\d*)',
            r'grade point.*?(\d+\.?\d*)',
            r'gpa.*?(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    cgpa = float(match.group(1))
                    # Validate CGPA range (0-10)
                    if 0 <= cgpa <= 10:
                        return cgpa
                except ValueError:
                    continue
        
        return None
    
    def _extract_branch(self, text: str, text_lower: str) -> Optional[str]:
        """Extract branch/department from text"""
        # Common patterns
        patterns = [
            r'(?:from|in|studying)\s+(\w+)\s+(?:branch|department|engineering)',
            r'(?:branch|department)\s*:?\s*(\w+)',
            r'\b(cse|ece|eee|mech|civil|it|is|aiml)\b'
        ]
        
        # Check patterns
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                branch_candidate = match.group(1).upper()
                # Validate against known branches
                for branch in self.branches:
                    if branch.upper() == branch_candidate or branch.lower() in text_lower:
                        return branch
        
        # Direct branch name check
        for branch in self.branches:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(branch.lower()) + r'\b', text_lower):
                return branch
        
        return None
    
    def _extract_skills(self, text: str, text_lower: str) -> List[str]:
        """Extract skills mentioned"""
        found_skills = []
        
        for skill in self.skills:
            # Case-insensitive search with word boundaries
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                if skill not in found_skills:
                    found_skills.append(skill)
        
        return found_skills
    
    def _extract_topic(self, query_lower: str) -> Optional[str]:
        """Extract the main topic from query"""
        # Topic keywords mapping
        topic_keywords = {
            "ctc": ["ctc", "salary", "package", "compensation", "pay"],
            "eligibility": ["eligible", "eligibility", "criteria", "requirement", "cutoff"],
            "interview": ["interview", "rounds", "process", "preparation", "questions"],
            "roles": ["role", "position", "job", "designation"],
            "deadline": ["deadline", "date", "timeline", "when", "schedule"],
            "process": ["process", "procedure", "how to apply", "application"],
            "comparison": ["compare", "comparison", "vs", "versus", "difference"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return topic
        
        return None
    
    def merge_entities(self, current: Dict, new: Dict) -> Dict:
        """
        Merge new entities with current entities.
        
        Args:
            current: Current entities in memory
            new: Newly extracted entities
            
        Returns:
            Merged entities dictionary
        """
        merged = current.copy()
        
        # Merge companies (add unique ones)
        for company in new.get("companies", []):
            if company not in merged.get("companies", []):
                merged.setdefault("companies", []).append(company)
        
        # Update CGPA if found
        if new.get("cgpa"):
            merged["cgpa"] = new["cgpa"]
        
        # Update branch if found
        if new.get("branch"):
            merged["branch"] = new["branch"]
        
        # Merge skills (add unique ones)
        for skill in new.get("skills", []):
            if skill not in merged.get("skills", []):
                merged.setdefault("skills", []).append(skill)
        
        return merged
    
    def get_entity_summary(self, entities: Dict) -> str:
        """Get human-readable entity summary"""
        parts = []
        
        companies = entities.get("companies", [])
        if companies:
            parts.append(f"Companies: {', '.join(companies[:5])}")
        
        cgpa = entities.get("cgpa")
        if cgpa:
            parts.append(f"CGPA: {cgpa}")
        
        branch = entities.get("branch")
        if branch:
            parts.append(f"Branch: {branch}")
        
        skills = entities.get("skills", [])
        if skills:
            parts.append(f"Skills: {', '.join(skills[:3])}")
        
        return " | ".join(parts) if parts else "No entities extracted"


# Example usage
if __name__ == "__main__":
    # Test the extractor
    known_companies = ["Intel", "AMD", "Nvidia", "Amazon", "Google"]
    extractor = EntityExtractor(known_companies)
    
    test_cases = [
        {
            "query": "What is Intel's CTC package?",
            "answer": "Intel offers a CTC of 44 LPA for M.Tech students."
        },
        {
            "query": "I have 8.5 CGPA in ECE. Am I eligible for Nvidia?",
            "answer": "Yes, Nvidia requires 8.0 CGPA for ECE students."
        },
        {
            "query": "Compare Amazon and Google",
            "answer": "Amazon offers 45 LPA while Google offers 48 LPA."
        }
    ]
    
    print("Testing Entity Extractor\n" + "="*50)
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Query: {case['query']}")
        entities = extractor.extract_from_conversation(case['query'], case['answer'])
        print(f"Extracted: {entities}")
        print(f"Summary: {extractor.get_entity_summary(entities)}")
