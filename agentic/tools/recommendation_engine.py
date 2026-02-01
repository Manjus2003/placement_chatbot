"""
🎯 PERSONALIZED RECOMMENDATION ENGINE
======================================
Generates personalized company recommendations.

Capabilities:
- Multi-factor scoring (eligibility, skills, interests, CTC, location)
- Personalized recommendations
- Action items and guidance
- Career path suggestions
"""

import re
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from .base_tool import BaseTool


class RecommendationEngineTool(BaseTool):
    """
    Generates personalized company recommendations based on student profile.
    """
    
    # Scoring weights
    WEIGHTS = {
        "eligibility": 30,
        "skills": 25,
        "interests": 20,
        "ctc": 15,
        "location": 10
    }
    
    # Location preferences
    LOCATIONS = {
        "Bangalore": ["bangalore", "bengaluru", "blr"],
        "Hyderabad": ["hyderabad", "hyd"],
        "Pune": ["pune"],
        "Mumbai": ["mumbai", "navi mumbai"],
        "Chennai": ["chennai"],
        "Delhi NCR": ["delhi", "gurgaon", "noida", "ncr"],
        "Remote": ["remote", "work from home", "wfh"]
    }
    
    def __init__(self, vector_search_tool=None, skill_demand_tool=None):
        """
        Initialize recommendation engine.
        
        Args:
            vector_search_tool: VectorSearchTool for querying documents
            skill_demand_tool: SkillDemandTool for skill analysis
        """
        self.vector_search_tool = vector_search_tool
        self.skill_demand_tool = skill_demand_tool
    
    @property
    def name(self) -> str:
        return "recommendation_engine"
    
    @property
    def description(self) -> str:
        return "Generate personalized company recommendations based on student profile"
    
    def run(
        self,
        cgpa: float = None,
        branch: str = None,
        skills: List[str] = None,
        interests: List[str] = None,
        location_preference: str = None,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate recommendations.
        
        Args:
            cgpa: Student CGPA (0-10 scale)
            branch: Branch/major (CSE, ECE, etc.)
            skills: List of skills ["Python", "DSA", "ML"]
            interests: List of interests ["AI", "Hardware", "Web Dev"]
            location_preference: Preferred location
            top_k: Number of recommendations
            
        Returns:
            Dict with recommendations and action items
        """
        try:
            print(f"🎯 Generating personalized recommendations...")
        
        # Get all companies
        companies = self._get_all_companies()
        
        # Score each company
        scored_companies = []
        for company in companies:
            score_breakdown = self._score_company(
                company, cgpa, branch, skills, interests, location_preference
            )
            
            if score_breakdown["total_score"] > 30:  # Minimum threshold
                scored_companies.append({
                    "company": company,
                    "total_score": score_breakdown["total_score"],
                    "breakdown": score_breakdown,
                    "reasoning": self._generate_reasoning(company, score_breakdown)
                })
        
        # Sort by score
        scored_companies.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Generate action items
        action_items = self._generate_action_items(
            scored_companies[:top_k], cgpa, skills
        )
        
        # Generate career path
        career_path = self._generate_career_path(scored_companies[:top_k])
        
            return {
                "response": self._format_response(
                    scored_companies[:top_k], action_items, career_path
                ),
                "recommendations": scored_companies[:top_k],
                "action_items": action_items,
                "career_path": career_path,
                "visualization": "radar_chart"
            }
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return {
                "response": f"## 🎯 Recommendation Error\n\nUnable to generate recommendations: {str(e)}\n\nPlease ensure your profile information is correct and try again.",
                "recommendations": [],
                "action_items": [],
                "career_path": {},
                "error": str(e)
            }
    
    def _get_all_companies(self) -> List[str]:
        """Get list of all companies"""
        
        if self.vector_search_tool:
            try:
                return self.vector_search_tool.get_all_companies()
            except:
                pass
        
        # Default sample companies
        return [
            "Amazon", "Google", "Microsoft", "Intel", "AMD", "Nvidia",
            "Qualcomm", "Bosch", "Samsung", "IBM", "Oracle", "SAP",
            "Dell", "HP", "Cisco", "Juniper", "Ericsson", "Accenture"
        ]
    
    def _score_company(
        self,
        company: str,
        cgpa: float,
        branch: str,
        skills: List[str],
        interests: List[str],
        location_preference: str
    ) -> Dict[str, Any]:
        """Score a company based on student profile"""
        
        scores = {}
        
        # 1. Eligibility Score (30 points)
        scores["eligibility"] = self._score_eligibility(company, cgpa, branch)
        
        # 2. Skills Match Score (25 points)
        scores["skills"] = self._score_skills(company, skills) if skills else 0
        
        # 3. Interests Match Score (20 points)
        scores["interests"] = self._score_interests(company, interests) if interests else 0
        
        # 4. CTC Score (15 points)
        scores["ctc"] = self._score_ctc(company)
        
        # 5. Location Score (10 points)
        scores["location"] = self._score_location(company, location_preference) if location_preference else 0
        
        # Calculate total
        total = sum(scores.values())
        
        return {
            "eligibility_score": scores["eligibility"],
            "skills_score": scores["skills"],
            "interests_score": scores["interests"],
            "ctc_score": scores["ctc"],
            "location_score": scores["location"],
            "total_score": round(total, 1)
        }
    
    def _score_eligibility(self, company: str, cgpa: float, branch: str) -> float:
        """Score eligibility (30 points max)"""
        
        score = 0
        
        # Get eligibility criteria
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} eligibility criteria CGPA branch",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "")
            except:
                context = ""
        else:
            context = ""
        
        # Check CGPA requirement
        if cgpa:
            min_cgpa = self._extract_min_cgpa(context)
            if min_cgpa and cgpa >= min_cgpa:
                score += 15  # Half points for CGPA
            elif not min_cgpa:  # No requirement found
                score += 10  # Partial credit
        
        # Check branch eligibility
        if branch:
            eligible_branches = self._extract_eligible_branches(context)
            if branch.upper() in eligible_branches or not eligible_branches:
                score += 15  # Half points for branch
        
        return score
    
    def _extract_min_cgpa(self, context: str) -> float:
        """Extract minimum CGPA requirement"""
        
        patterns = [
            r'CGPA[:\s]+(\d+(?:\.\d+)?)',
            r'minimum[:\s]+(\d+(?:\.\d+)?)[:\s]+CGPA',
            r'(\d+(?:\.\d+)?)[:\s]+CGPA'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None
    
    def _extract_eligible_branches(self, context: str) -> Set[str]:
        """Extract eligible branches"""
        
        branches = {"CSE", "ECE", "EEE", "MECH", "CIVIL", "IT", "IS", "AIML"}
        found_branches = set()
        
        context_upper = context.upper()
        for branch in branches:
            if branch in context_upper:
                found_branches.add(branch)
        
        return found_branches
    
    def _score_skills(self, company: str, skills: List[str]) -> float:
        """Score skills match (25 points max)"""
        
        if not skills:
            return 0
        
        # Get required skills for company
        required_skills = self._get_required_skills(company)
        
        if not required_skills:
            return 10  # Partial credit if no data
        
        # Calculate match percentage
        skills_set = set(s.lower() for s in skills)
        required_set = set(s.lower() for s in required_skills)
        
        match_count = len(skills_set & required_set)
        match_ratio = match_count / len(required_set) if required_set else 0
        
        return round(match_ratio * 25, 1)
    
    def _get_required_skills(self, company: str) -> List[str]:
        """Get required skills for company"""
        
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} required skills technical skills qualifications",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "")
                
                # Extract common skills
                skills = []
                skill_keywords = [
                    "Python", "Java", "C++", "DSA", "Machine Learning",
                    "VLSI", "Embedded", "SQL", "AWS", "React"
                ]
                
                for skill in skill_keywords:
                    if skill.lower() in context.lower():
                        skills.append(skill)
                
                return skills
            except:
                pass
        
        return []
    
    def _score_interests(self, company: str, interests: List[str]) -> float:
        """Score interests match (20 points max)"""
        
        if not interests:
            return 0
        
        # Get company domain
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} company profile domain work areas",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "").lower()
            except:
                context = company.lower()
        else:
            context = company.lower()
        
        # Check interests match
        match_count = 0
        for interest in interests:
            if interest.lower() in context:
                match_count += 1
        
        match_ratio = match_count / len(interests) if interests else 0
        return round(match_ratio * 20, 1)
    
    def _score_ctc(self, company: str) -> float:
        """Score CTC (15 points max)"""
        
        # Get CTC
        ctc = self._extract_ctc(company)
        
        # Score based on CTC range
        if ctc >= 40:
            return 15  # Top tier
        elif ctc >= 25:
            return 12  # High tier
        elif ctc >= 15:
            return 9   # Mid tier
        elif ctc >= 10:
            return 6   # Entry level
        else:
            return 3
    
    def _extract_ctc(self, company: str) -> float:
        """Extract CTC for company"""
        
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} CTC package salary",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "")
                
                patterns = [
                    r'(\d+(?:\.\d+)?)\s*LPA',
                    r'Rs\.?\s*(\d+(?:\.\d+)?)\s*lakhs'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            except:
                pass
        
        # Sample CTC
        company_lower = company.lower()
        if any(kw in company_lower for kw in ['google', 'amazon', 'microsoft']):
            return 45.0
        elif any(kw in company_lower for kw in ['intel', 'nvidia', 'qualcomm']):
            return 30.0
        else:
            return 18.0
    
    def _score_location(self, company: str, location_preference: str) -> float:
        """Score location match (10 points max)"""
        
        if not location_preference:
            return 0
        
        # Get company locations
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} office location work location",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "").lower()
            except:
                context = ""
        else:
            context = ""
        
        # Check if preferred location matches
        pref_lower = location_preference.lower()
        
        for location, keywords in self.LOCATIONS.items():
            if pref_lower in location.lower():
                if any(kw in context for kw in keywords):
                    return 10  # Full match
        
        return 0
    
    def _generate_reasoning(
        self,
        company: str,
        score_breakdown: Dict[str, Any]
    ) -> str:
        """Generate reasoning for recommendation"""
        
        reasons = []
        
        if score_breakdown["eligibility_score"] >= 20:
            reasons.append("You meet the eligibility criteria")
        
        if score_breakdown["skills_score"] >= 15:
            reasons.append("Strong skills match")
        
        if score_breakdown["interests_score"] >= 10:
            reasons.append("Aligns with your interests")
        
        if score_breakdown["ctc_score"] >= 12:
            reasons.append("Competitive compensation")
        
        if score_breakdown["location_score"] >= 8:
            reasons.append("Preferred location available")
        
        return "; ".join(reasons) if reasons else "Good overall fit"
    
    def _generate_action_items(
        self,
        recommendations: List[Dict[str, Any]],
        cgpa: float,
        skills: List[str]
    ) -> List[str]:
        """Generate action items for student"""
        
        action_items = []
        
        # CGPA improvement
        if cgpa and cgpa < 7.5:
            action_items.append("📚 Focus on improving CGPA to 7.5+ for more opportunities")
        
        # Skills gap
        if skills:
            # Get top required skills from recommendations
            all_required = []
            for rec in recommendations[:3]:
                company = rec["company"]
                required = self._get_required_skills(company)
                all_required.extend(required)
            
            # Find missing skills
            skills_set = set(s.lower() for s in skills)
            required_set = set(s.lower() for s in all_required)
            missing = required_set - skills_set
            
            if missing:
                missing_list = list(missing)[:3]
                action_items.append(f"💻 Learn these in-demand skills: {', '.join(missing_list)}")
        
        # General tips
        action_items.append("📝 Prepare for coding interviews (DSA, System Design)")
        action_items.append("🎯 Customize resume for each company's requirements")
        
        return action_items
    
    def _generate_career_path(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate career path suggestion"""
        
        if not recommendations:
            return {}
        
        # Identify career track based on companies
        companies = [rec["company"] for rec in recommendations[:3]]
        companies_str = ", ".join(companies).lower()
        
        if any(kw in companies_str for kw in ['intel', 'nvidia', 'amd', 'qualcomm']):
            track = "Hardware/Chip Design"
            progression = ["Junior Engineer", "Senior Engineer", "Architect", "Principal Engineer"]
        elif any(kw in companies_str for kw in ['google', 'amazon', 'microsoft']):
            track = "Software Engineering"
            progression = ["SDE-1", "SDE-2", "Senior SDE", "Staff Engineer", "Principal Engineer"]
        else:
            track = "Technology"
            progression = ["Associate", "Engineer", "Senior Engineer", "Lead Engineer"]
        
        return {
            "track": track,
            "progression": progression,
            "timeline": "2-3 years per level (average)"
        }
    
    def _format_response(
        self,
        recommendations: List[Dict[str, Any]],
        action_items: List[str],
        career_path: Dict[str, Any]
    ) -> str:
        """Format response for display"""
        
        response = "## 🎯 Personalized Company Recommendations\n\n"
        
        # Top recommendations
        for i, rec in enumerate(recommendations, 1):
            company = rec["company"]
            score = rec["total_score"]
            reasoning = rec["reasoning"]
            breakdown = rec["breakdown"]
            
            response += f"### {i}. {company} (Score: {score}/100)\n\n"
            response += f"**Why this match?** {reasoning}\n\n"
            response += "**Score Breakdown:**\n"
            response += f"- Eligibility: {breakdown['eligibility_score']}/30\n"
            response += f"- Skills: {breakdown['skills_score']}/25\n"
            response += f"- Interests: {breakdown['interests_score']}/20\n"
            response += f"- CTC: {breakdown['ctc_score']}/15\n"
            response += f"- Location: {breakdown['location_score']}/10\n\n"
        
        # Action items
        if action_items:
            response += "### 📋 Action Items\n\n"
            for item in action_items:
                response += f"{item}\n\n"
        
        # Career path
        if career_path:
            response += "### 🚀 Suggested Career Path\n\n"
            response += f"**Track:** {career_path['track']}\n\n"
            response += "**Progression:**\n"
            for level in career_path["progression"]:
                response += f"- {level}\n"
            response += f"\n*{career_path['timeline']}*\n"
        
        return response


# Example usage
if __name__ == "__main__":
    tool = RecommendationEngineTool()
    
    result = tool.run(
        cgpa=8.5,
        branch="CSE",
        skills=["Python", "DSA", "Machine Learning"],
        interests=["AI", "Software Development"],
        location_preference="Bangalore",
        top_k=5
    )
    
    print(result["response"])
