"""
🛠️ SKILL DEMAND ANALYZER TOOL
==============================
Analyzes skill demand across companies.

Capabilities:
- Rank skills by demand
- Identify skill combinations
- Domain-wise skill mapping
- Emerging skills detection
"""

import re
from typing import Dict, Any, List, Set
from collections import Counter, defaultdict
from .base_tool import BaseTool


class SkillDemandTool(BaseTool):
    """
    Analyzes skill demand across placement companies.
    """
    
    # Known skills database
    SKILLS_DATABASE = {
        "programming": [
            "Python", "Java", "C++", "C", "JavaScript", "TypeScript",
            "Go", "Rust", "Scala", "Kotlin", "Swift"
        ],
        "data_structures": [
            "DSA", "Data Structures", "Algorithms", "Problem Solving"
        ],
        "web": [
            "React", "Angular", "Vue", "Node.js", "Django", "Flask",
            "Spring Boot", "HTML", "CSS"
        ],
        "database": [
            "SQL", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Cassandra"
        ],
        "cloud": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins"
        ],
        "ml_ai": [
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "TensorFlow", "PyTorch", "Scikit-learn"
        ],
        "hardware": [
            "VLSI", "Embedded Systems", "Verilog", "VHDL", "RTL",
            "Circuit Design", "Microcontrollers", "IoT"
        ],
        "systems": [
            "Operating Systems", "Computer Networks", "System Design",
            "Distributed Systems", "Microservices"
        ],
        "tools": [
            "Git", "Linux", "CI/CD", "Agile", "JIRA"
        ]
    }
    
    # Flatten skills list
    ALL_SKILLS = [skill for category in SKILLS_DATABASE.values() for skill in category]
    
    def __init__(self, vector_search_tool=None):
        """
        Initialize skill demand analyzer.
        
        Args:
            vector_search_tool: VectorSearchTool for querying documents
        """
        self.vector_search_tool = vector_search_tool
    
    @property
    def name(self) -> str:
        return "skill_demand"
    
    @property
    def description(self) -> str:
        return "Analyze skill demand and requirements across companies"
    
    def run(
        self,
        skill_category: str = "all",
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze skill demand.
        
        Args:
            skill_category: "programming", "ml_ai", "hardware", or "all"
            top_k: Number of top skills to return
            
        Returns:
            Dict with skill rankings and insights
        """
        try:
            print(f"🛠️ Analyzing skill demand...")
        
        # Extract skills from documents
        skill_mentions = self._extract_skills_from_documents()
        
        # Calculate statistics
        skill_stats = self._calculate_skill_statistics(skill_mentions)
        
        # Rank skills
        rankings = self._rank_skills(skill_stats, skill_category, top_k)
        
        # Find skill combinations
        combinations = self._find_skill_combinations(skill_mentions)
        
        # Identify domain mapping
        domain_mapping = self._map_skills_to_domains()
        
        # Detect emerging skills
        emerging = self._detect_emerging_skills(skill_mentions)
        
            return {
                "response": self._format_response(rankings, combinations, emerging),
                "skill_rankings": rankings,
                "skill_combinations": combinations[:5] if combinations else [],
                "domain_mapping": domain_mapping,
                "emerging_skills": emerging,
                "visualization": "horizontal_bar_chart"
            }
        except Exception as e:
            print(f"❌ Error in skill demand analysis: {e}")
            return {
                "response": f"## 🛠️ Skill Demand Analysis Error\n\nUnable to analyze skills: {str(e)}\n\nPlease try again or contact support.",
                "skill_rankings": [],
                "skill_combinations": [],
                "domain_mapping": {},
                "emerging_skills": [],
                "error": str(e)
            }
    
    def _extract_skills_from_documents(self) -> Dict[str, List[str]]:
        """
        Extract skills mentioned in company documents.
        
        Returns:
            {"CompanyName": ["Python", "DSA", "Java"]}
        """
        skill_mentions = {}
        
        # Get all companies
        if self.vector_search_tool:
            companies = self.vector_search_tool.get_all_companies()
        else:
            companies = ["Intel", "AMD", "Amazon", "Google", "Microsoft", 
                        "Nvidia", "Qualcomm", "Bosch"]
        
        for company in companies:
            skills = self._extract_skills_for_company(company)
            if skills:
                skill_mentions[company] = skills
        
        return skill_mentions
    
    def _extract_skills_for_company(self, company: str) -> List[str]:
        """Extract skills for a specific company"""
        
        # Search for skills/requirements
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} skills required technical skills qualifications",
                    company_filter=company,
                    top_k=3
                )
                context = result.get("context", "")
            except:
                context = ""
        else:
            context = ""
        
        # Extract skills using pattern matching
        found_skills = []
        
        for skill in self.ALL_SKILLS:
            # Case-insensitive search with word boundaries
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, context, re.IGNORECASE):
                found_skills.append(skill)
        
        # If no skills found, use sample data
        if not found_skills:
            found_skills = self._get_sample_skills(company)
        
        return found_skills
    
    def _get_sample_skills(self, company: str) -> List[str]:
        """Return sample skills based on company type"""
        
        company_lower = company.lower()
        
        # Software companies
        if any(kw in company_lower for kw in ['amazon', 'google', 'microsoft']):
            return ["Python", "Java", "DSA", "System Design", "AWS", "SQL"]
        
        # Hardware companies
        elif any(kw in company_lower for kw in ['intel', 'amd', 'nvidia', 'qualcomm']):
            return ["VLSI", "Embedded Systems", "C", "C++", "Verilog"]
        
        # ML/AI companies
        elif any(kw in company_lower for kw in ['openai', 'anthropic']):
            return ["Python", "Machine Learning", "Deep Learning", "PyTorch", "NLP"]
        
        # Default
        else:
            return ["Python", "DSA", "C++", "SQL"]
    
    def _calculate_skill_statistics(
        self,
        skill_mentions: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each skill"""
        
        skill_stats = defaultdict(lambda: {"count": 0, "companies": []})
        
        for company, skills in skill_mentions.items():
            for skill in skills:
                skill_stats[skill]["count"] += 1
                skill_stats[skill]["companies"].append(company)
        
        return dict(skill_stats)
    
    def _rank_skills(
        self,
        skill_stats: Dict[str, Dict[str, Any]],
        category: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rank skills by demand"""
        
        # Filter by category if specified
        if category != "all" and category in self.SKILLS_DATABASE:
            category_skills = set(self.SKILLS_DATABASE[category])
            skill_stats = {
                skill: stats for skill, stats in skill_stats.items()
                if skill in category_skills
            }
        
        # Calculate demand score (simple count for now)
        rankings = []
        for skill, stats in skill_stats.items():
            demand_score = stats["count"] * 10  # Scale to 0-100
            
            rankings.append({
                "skill": skill,
                "demand_score": min(100, demand_score),
                "companies": len(stats["companies"]),
                "company_list": stats["companies"][:5]
            })
        
        # Sort by demand score
        rankings.sort(key=lambda x: x["demand_score"], reverse=True)
        
        return rankings[:top_k]
    
    def _find_skill_combinations(
        self,
        skill_mentions: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Find common skill combinations"""
        
        combinations = Counter()
        
        for company, skills in skill_mentions.items():
            # Find pairs of skills
            for i, skill1 in enumerate(skills):
                for skill2 in skills[i+1:]:
                    # Sort to avoid duplicates (A+B == B+A)
                    pair = tuple(sorted([skill1, skill2]))
                    combinations[pair] += 1
        
        # Format results
        result = []
        for (skill1, skill2), count in combinations.most_common(10):
            result.append({
                "combination": f"{skill1} + {skill2}",
                "frequency": count,
                "companies": count
            })
        
        return result
    
    def _map_skills_to_domains(self) -> Dict[str, List[str]]:
        """Map skills to job domains"""
        
        domain_mapping = {
            "Software Development": ["Python", "Java", "DSA", "System Design", "Git"],
            "Data Science/ML": ["Python", "Machine Learning", "SQL", "TensorFlow", "Scikit-learn"],
            "Hardware/Chip Design": ["VLSI", "Verilog", "Embedded Systems", "C"],
            "Web Development": ["React", "Node.js", "JavaScript", "HTML", "CSS"],
            "Cloud/DevOps": ["AWS", "Docker", "Kubernetes", "Jenkins", "Linux"],
            "Mobile Development": ["Swift", "Kotlin", "React Native", "Flutter"]
        }
        
        return domain_mapping
    
    def _detect_emerging_skills(
        self,
        skill_mentions: Dict[str, List[str]]
    ) -> List[str]:
        """Detect emerging/trending skills"""
        
        # In real implementation, would compare year-over-year
        # For now, return known emerging skills
        emerging = ["GenAI", "LLM", "Rust", "Kubernetes", "MLOps", "GraphQL"]
        
        # Filter to only those mentioned
        mentioned_emerging = [
            skill for skill in emerging
            if any(skill in skills for skills in skill_mentions.values())
        ]
        
        return mentioned_emerging if mentioned_emerging else emerging[:3]
    
    def _format_response(
        self,
        rankings: List[Dict[str, Any]],
        combinations: List[Dict[str, Any]],
        emerging: List[str]
    ) -> str:
        """Format response for display"""
        
        response = "## 🛠️ Skill Demand Analysis\n\n"
        
        # Top skills
        response += "### Top Skills by Demand\n\n"
        response += "| Rank | Skill | Demand Score | Companies |\n"
        response += "|------|-------|--------------|----------|\n"
        
        for i, skill_data in enumerate(rankings, 1):
            response += f"| {i} | {skill_data['skill']} | {skill_data['demand_score']}/100 | {skill_data['companies']} |\n"
        
        # Skill combinations
        if combinations:
            response += "\n### 🔗 Popular Skill Combinations\n\n"
            for combo in combinations[:5]:
                response += f"- **{combo['combination']}** ({combo['frequency']} companies)\n"
        
        # Emerging skills
        if emerging:
            response += "\n### 🚀 Emerging Skills\n\n"
            for skill in emerging:
                response += f"- {skill}\n"
        
        return response


# Example usage
if __name__ == "__main__":
    tool = SkillDemandTool()
    
    result = tool.run(skill_category="all", top_k=10)
    
    print(result["response"])
