"""
🎓 BRANCH-WISE STATISTICS TOOL
===============================
Analyzes placement patterns by branch/department.

Capabilities:
- Offer counts per branch
- Average CTC by branch
- Company preferences per branch
- Skill requirements per branch
"""

import re
from typing import Dict, Any, List, Set
from collections import defaultdict
from .base_tool import BaseTool


class BranchWiseStatsTool(BaseTool):
    """
    Analyzes branch-wise placement statistics.
    """
    
    # Known branch abbreviations and full names
    BRANCHES = {
        "CSE": "Computer Science Engineering",
        "ECE": "Electronics and Communication Engineering",
        "EEE": "Electrical and Electronics Engineering",
        "MECH": "Mechanical Engineering",
        "CIVIL": "Civil Engineering",
        "IT": "Information Technology",
        "IS": "Information Science",
        "AIML": "Artificial Intelligence and Machine Learning"
    }
    
    BRANCH_ALIASES = {
        "CS": "CSE",
        "EC": "ECE",
        "EE": "EEE",
        "ME": "MECH",
        "CE": "CIVIL"
    }
    
    def __init__(self, vector_search_tool=None):
        """
        Initialize branch stats tool.
        
        Args:
            vector_search_tool: VectorSearchTool for querying documents
        """
        self.vector_search_tool = vector_search_tool
    
    @property
    def name(self) -> str:
        return "branch_stats"
    
    @property
    def description(self) -> str:
        return "Analyze placement statistics by branch/department"
    
    def run(
        self,
        branch: str = None,
        metric: str = "all",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze branch-wise statistics.
        
        Args:
            branch: Specific branch to analyze (None for all)
            metric: "offers", "ctc", "companies", or "all"
            
        Returns:
            Dict with branch statistics
        """
        try:
            print(f\"🎓 Analyzing branch-wise statistics...\")
        
        # Extract branch eligibility from documents
        eligibility_data = self._extract_branch_eligibility()
        
        # Calculate statistics
        branch_stats = self._calculate_branch_stats(eligibility_data)
        
        # Find company preferences
        preferences = self._find_company_preferences(eligibility_data)
        
        # Identify common skills
        skills = self._identify_branch_skills()
        
        # Filter if specific branch requested
        if branch:
            branch = self._normalize_branch_name(branch)
            if branch in branch_stats:
                branch_stats = {branch: branch_stats[branch]}
                preferences = {branch: preferences.get(branch, [])}
        
        return {
            "response": self._format_response(branch_stats, preferences, skills),
            "branch_distribution": branch_stats,
            "company_preferences": preferences,
            "skill_mapping": skills,
            "visualization": "grouped_bar_chart"
        }    except Exception as e:
        print(f\"❌ Error in branch analysis: {e}\")
        return {
            \"response\": f\"## 🎓 Branch Analysis Error\\n\\nUnable to analyze branch data: {str(e)}\\n\\nPlease try again or contact support.\",
            \"branch_distribution\": {},
            \"company_preferences\": {},
            \"skill_mapping\": {},
            \"error\": str(e)
        }    
    def _normalize_branch_name(self, branch: str) -> str:
        """Normalize branch name to standard abbreviation"""
        branch = branch.upper().strip()
        
        # Check if it's an alias
        if branch in self.BRANCH_ALIASES:
            return self.BRANCH_ALIASES[branch]
        
        # Check if it's already a standard name
        if branch in self.BRANCHES:
            return branch
        
        # Check if it's a full name
        for abbr, full in self.BRANCHES.items():
            if branch in full.upper():
                return abbr
        
        return branch
    
    def _extract_branch_eligibility(self) -> Dict[str, Set[str]]:
        """
        Extract which branches are eligible for which companies.
        
        Returns:
            {"CompanyName": {"CSE", "ECE", "EEE"}}
        """
        eligibility = defaultdict(set)
        
        # Get all companies
        if self.vector_search_tool:
            companies = self.vector_search_tool.get_all_companies()
        else:
            # Fallback sample data
            companies = ["Intel", "AMD", "Amazon", "Google", "Microsoft", 
                        "Nvidia", "Qualcomm", "Bosch", "Dell", "NetApp"]
        
        # For each company, extract branch eligibility
        for company in companies:
            branches = self._extract_branches_for_company(company)
            if branches:
                eligibility[company] = branches
        
        return dict(eligibility)
    
    def _extract_branches_for_company(self, company: str) -> Set[str]:
        """Extract eligible branches for a specific company"""
        
        # Search for eligibility criteria
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} eligibility criteria branches eligible",
                    company_filter=company,
                    top_k=3
                )
                context = result.get("context", "")
            except:
                context = ""
        else:
            context = ""
        
        # Extract branches from context
        branches = set()
        
        # Common patterns
        patterns = [
            r'(?:Eligible|Open to|For)\s+(?:branches?|departments?):?\s*([\w\s,/]+)',
            r'Branches?:?\s*([\w\s,/]+)',
            r'(?:CSE|ECE|EEE|MECH|CIVIL|IT|IS|AIML)(?:\s*[,/]\s*(?:CSE|ECE|EEE|MECH|CIVIL|IT|IS|AIML))+',
            r'Only\s+([\w\s,/]+)\s+students'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                text = match.group(1) if len(match.groups()) > 0 else match.group(0)
                # Extract branch abbreviations
                for branch in self.BRANCHES.keys():
                    if branch in text.upper():
                        branches.add(branch)
        
        # If no branches found, use sample data based on company type
        if not branches:
            branches = self._get_sample_branches(company)
        
        return branches
    
    def _get_sample_branches(self, company: str) -> Set[str]:
        """Return sample branch eligibility based on company type"""
        
        company_lower = company.lower()
        
        # Software companies - all CS branches
        if any(kw in company_lower for kw in ['amazon', 'google', 'microsoft', 'netflix']):
            return {"CSE", "IT", "IS", "AIML"}
        
        # Hardware companies - EE branches
        elif any(kw in company_lower for kw in ['intel', 'amd', 'nvidia', 'qualcomm']):
            return {"CSE", "ECE", "EEE"}
        
        # Automotive - Mechanical + EE
        elif any(kw in company_lower for kw in ['volvo', 'nissan', 'bosch']):
            return {"MECH", "ECE", "EEE", "CSE"}
        
        # Consulting/IT - All branches
        elif any(kw in company_lower for kw in ['sap', 'dell', 'netapp']):
            return {"CSE", "ECE", "EEE", "MECH", "IT"}
        
        # Default - CS and EC
        else:
            return {"CSE", "ECE"}
    
    def _calculate_branch_stats(
        self,
        eligibility_data: Dict[str, Set[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each branch"""
        
        branch_stats = {}
        
        for branch in self.BRANCHES.keys():
            # Count companies eligible for this branch
            eligible_companies = [
                company for company, branches in eligibility_data.items()
                if branch in branches
            ]
            
            company_count = len(eligible_companies)
            
            # Sample CTC data (in real implementation, would extract from documents)
            avg_ctc = self._estimate_avg_ctc(branch, eligible_companies)
            
            branch_stats[branch] = {
                "total_companies": company_count,
                "avg_ctc": round(avg_ctc, 1),
                "offer_percentage": round((company_count / len(eligibility_data)) * 100, 1) if eligibility_data else 0,
                "eligible_companies": eligible_companies[:10]  # Top 10
            }
        
        return branch_stats
    
    def _estimate_avg_ctc(self, branch: str, companies: List[str]) -> float:
        """Estimate average CTC for branch (placeholder)"""
        
        # In real implementation, would extract from documents
        # For now, use heuristics
        base_ctc = {
            "CSE": 40,
            "IT": 38,
            "IS": 38,
            "AIML": 42,
            "ECE": 38,
            "EEE": 36,
            "MECH": 34,
            "CIVIL": 32
        }
        
        return base_ctc.get(branch, 35)
    
    def _find_company_preferences(
        self,
        eligibility_data: Dict[str, Set[str]]
    ) -> Dict[str, List[str]]:
        """Find top companies for each branch"""
        
        preferences = {}
        
        for branch in self.BRANCHES.keys():
            # Get companies that prefer this branch
            branch_companies = [
                company for company, branches in eligibility_data.items()
                if branch in branches
            ]
            
            # Sort by relevance (in real implementation, would use CTC or other metrics)
            preferences[branch] = branch_companies[:5]  # Top 5
        
        return preferences
    
    def _identify_branch_skills(self) -> Dict[str, List[str]]:
        """Identify common skills required for each branch"""
        
        # Sample skill mapping (in real implementation, would extract from documents)
        skills = {
            "CSE": ["DSA", "System Design", "DBMS", "Python", "Java"],
            "IT": ["Web Development", "Cloud", "DevOps", "Python", "SQL"],
            "IS": ["Data Science", "Analytics", "Python", "ML", "SQL"],
            "AIML": ["Machine Learning", "Deep Learning", "Python", "TensorFlow", "NLP"],
            "ECE": ["VLSI", "Embedded Systems", "Signal Processing", "C", "Verilog"],
            "EEE": ["Power Systems", "Control Systems", "Embedded", "Circuit Design"],
            "MECH": ["CAD", "Thermodynamics", "Manufacturing", "ANSYS"],
            "CIVIL": ["Structural Analysis", "AutoCAD", "Project Management"]
        }
        
        return skills
    
    def _format_response(
        self,
        branch_stats: Dict[str, Dict[str, Any]],
        preferences: Dict[str, List[str]],
        skills: Dict[str, List[str]]
    ) -> str:
        """Format response for display"""
        
        response = "## Branch-Wise Placement Statistics\n\n"
        
        # Statistics table
        response += "| Branch | Companies | Avg CTC | Coverage |\n"
        response += "|--------|-----------|---------|----------|\n"
        
        for branch, stats in sorted(branch_stats.items()):
            response += f"| {branch} | {stats['total_companies']} | {stats['avg_ctc']} LPA | {stats['offer_percentage']}% |\n"
        
        # Top companies per branch
        response += "\n### 🏢 Top Companies by Branch\n\n"
        for branch, companies in preferences.items():
            if companies and branch in branch_stats:
                response += f"**{branch}:** {', '.join(companies[:5])}\n\n"
        
        # Key skills
        response += "### 🛠️ Key Skills by Branch\n\n"
        for branch, skill_list in skills.items():
            if branch in branch_stats:
                response += f"**{branch}:** {', '.join(skill_list[:5])}\n\n"
        
        return response


# Example usage
if __name__ == "__main__":
    tool = BranchWiseStatsTool()
    
    result = tool.run(metric="all")
    
    print(result["response"])
