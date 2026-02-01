"""
🧠 PLANNER MODULE
=================
Responsible for:
1. Query Classification - Understanding what type of question is being asked
2. Task Decomposition - Breaking complex queries into sub-tasks
3. Tool Selection - Deciding which tools are needed
4. Execution Strategy - Creating the execution plan

Query Types:
- SINGLE_COMPANY_INFO: Questions about one specific company
- MULTI_COMPANY_COMPARISON: Compare 2+ companies
- STATISTICAL_ANALYSIS: Averages, counts, trends
- ELIGIBILITY_CHECK: Student eligibility verification
- INTERVIEW_PREP: Interview questions and preparation
- TIMELINE_INFO: Dates, deadlines, schedules
- OPEN_ENDED: General placement questions
"""

import os
import sys
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for input_validator
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from input_validator import validate_input, ValidationResult


class QueryType(Enum):
    """Types of queries the system can handle"""
    SINGLE_COMPANY_INFO = "single_company_info"
    MULTI_COMPANY_COMPARISON = "multi_company_comparison"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ELIGIBILITY_CHECK = "eligibility_check"
    INTERVIEW_PREP = "interview_prep"
    TIMELINE_INFO = "timeline_info"
    SALARY_NEGOTIATION = "salary_negotiation"
    OFFER_COMPARISON = "offer_comparison"
    CAREER_PATH = "career_path"
    LOCATION_BASED = "location_based"
    TREND_ANALYSIS = "trend_analysis"
    BRANCH_ANALYSIS = "branch_analysis"
    SKILL_ANALYSIS = "skill_analysis"
    COMPANY_CLUSTERING = "company_clustering"
    PERSONALIZED_RECOMMENDATION = "personalized_recommendation"
    OPEN_ENDED = "open_ended"


@dataclass
class Step:
    """
    A single step in the execution plan.
    
    Attributes:
        tool: Name of the tool to execute
        args: Arguments to pass to the tool
        depends_on: List of step indices this step depends on
        description: Human-readable description of what this step does
    """
    tool: str
    args: Dict[str, Any]
    depends_on: List[int] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "depends_on": self.depends_on,
            "description": self.description
        }


@dataclass
class Plan:
    """
    Execution plan containing multiple steps.
    
    Attributes:
        query: Original user query
        query_type: Classified query type
        steps: List of execution steps
        execution_mode: 'sequential' or 'parallel' where possible
        metadata: Additional planning metadata
    """
    query: str
    query_type: QueryType
    steps: List[Step]
    execution_mode: str = "sequential"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "steps": [s.to_dict() for s in self.steps],
            "execution_mode": self.execution_mode,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        lines = [
            f"📋 Plan for: {self.query[:50]}...",
            f"   Type: {self.query_type.value}",
            f"   Mode: {self.execution_mode}",
            f"   Steps ({len(self.steps)}):"
        ]
        for i, step in enumerate(self.steps):
            lines.append(f"     {i+1}. [{step.tool}] {step.description}")
        return "\n".join(lines)


class Planner:
    """
    Query understanding and task decomposition engine.
    
    The Planner analyzes incoming queries and creates execution plans
    that the Executor will follow.
    """
    
    def __init__(self, use_llm_classification: bool = True):
        """
        Initialize the Planner.
        
        Args:
            use_llm_classification: If True, use LLM for query classification.
                                   If False, use rule-based classification.
        """
        load_dotenv(dotenv_path="environment.env")
        self.use_llm_classification = use_llm_classification
        
        # Initialize LLM client for classification
        hf_token = os.getenv("HF_TOKEN")
        if hf_token and use_llm_classification:
            self.llm_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
            )
            print("✅ Planner initialized with LLM classification")
        else:
            self.llm_client = None
            print("✅ Planner initialized with rule-based classification")
        
        # Feedback from Critic for re-planning
        self.feedback_history: List[Dict] = []
        
        # Known company names (loaded later from data)
        self.known_companies: List[str] = []
    
    def set_known_companies(self, companies: List[str]):
        """Set list of known company names for extraction"""
        self.known_companies = companies
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the query into a QueryType.
        
        Uses LLM if available, otherwise falls back to rule-based.
        """
        if self.llm_client and self.use_llm_classification:
            return self._classify_with_llm(query)
        return self._classify_with_rules(query)
    
    def _classify_with_llm(self, query: str) -> QueryType:
        """Use LLM for query classification"""
        classification_prompt = f"""Classify this placement-related query into exactly ONE category.

Query: "{query}"

Categories (return ONLY the category name, nothing else):
- SINGLE_COMPANY_INFO: Info about one specific company (CTC, process, eligibility, roles)
- MULTI_COMPANY_COMPARISON: Compare multiple companies side by side
- STATISTICAL_ANALYSIS: Averages, counts, percentages, trends over time
- ELIGIBILITY_CHECK: Check if student is eligible based on CGPA, branch, skills
- INTERVIEW_PREP: Interview questions, preparation tips, selection process
- TIMELINE_INFO: Dates, deadlines, schedules, when companies visit
- OPEN_ENDED: General questions that don't fit above categories

Return ONLY the category name in uppercase with underscores (e.g., SINGLE_COMPANY_INFO)."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=50,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip().upper()
            
            # Parse result to QueryType
            result = result.replace(" ", "_")
            for qt in QueryType:
                if qt.name in result:
                    return qt
            
            return QueryType.OPEN_ENDED
            
        except Exception as e:
            print(f"⚠️ LLM classification failed: {e}, using rules")
            return self._classify_with_rules(query)
    
    def _classify_with_rules(self, query: str) -> QueryType:
        """Rule-based query classification"""
        query_lower = query.lower()
        
        # Check for comparison keywords
        comparison_patterns = [
            r"compare\s+\w+\s+(and|vs|versus|with)\s+\w+",
            r"difference\s+between",
            r"which\s+(is\s+)?better",
            r"\w+\s+vs\.?\s+\w+",
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, query_lower):
                return QueryType.MULTI_COMPANY_COMPARISON
        
        # Check for statistical keywords
        statistical_keywords = [
            "average", "mean", "total", "count", "how many",
            "percentage", "trend", "statistics", "highest", "lowest",
            "maximum", "minimum", "top 10", "top 5", "ranking"
        ]
        if any(kw in query_lower for kw in statistical_keywords):
            return QueryType.STATISTICAL_ANALYSIS
        
        # Check for eligibility keywords
        eligibility_keywords = [
            "eligible", "eligibility", "can i apply", "qualify",
            "cgpa requirement", "criteria", "am i eligible",
            "minimum cgpa", "required cgpa", "cutoff"
        ]
        if any(kw in query_lower for kw in eligibility_keywords):
            return QueryType.ELIGIBILITY_CHECK
        
        # Check for interview prep keywords
        interview_keywords = [
            "interview", "prepare", "preparation", "questions",
            "rounds", "selection process", "how to crack",
            "tips", "experience", "asked in"
        ]
        if any(kw in query_lower for kw in interview_keywords):
            return QueryType.INTERVIEW_PREP
        
        # Check for timeline keywords
        timeline_keywords = [
            "when", "date", "deadline", "schedule", "timeline",
            "visiting", "next", "upcoming", "last date"
        ]
        if any(kw in query_lower for kw in timeline_keywords):
            return QueryType.TIMELINE_INFO
        
        # Check for salary negotiation keywords
        negotiation_keywords = [
            "negotiate", "negotiation", "bargain", "leverage",
            "counter offer", "how to negotiate", "negotiating"
        ]
        if any(kw in query_lower for kw in negotiation_keywords):
            return QueryType.SALARY_NEGOTIATION
        
        # Check for offer comparison keywords
        offer_keywords = [
            "have offers", "which offer", "should i choose",
            "accept offer", "decide between offers"
        ]
        if any(kw in query_lower for kw in offer_keywords):
            return QueryType.OFFER_COMPARISON
        
        # Check for career path keywords
        career_keywords = [
            "career", "growth", "path", "progression",
            "promotion", "future", "roles can i get"
        ]
        if any(kw in query_lower for kw in career_keywords):
            return QueryType.CAREER_PATH
        
        # Check for location keywords
        location_keywords = [
            "bangalore", "hyderabad", "pune", "chennai", "mumbai",
            "location", "city", "remote", "work from home", "wfh"
        ]
        if any(kw in query_lower for kw in location_keywords):
            return QueryType.LOCATION_BASED
        
        # Check for trend analysis keywords
        trend_keywords = [
            "trend", "over time", "growth", "year over year",
            "increasing", "decreasing", "changes", "historical"
        ]
        if any(kw in query_lower for kw in trend_keywords):
            return QueryType.TREND_ANALYSIS
        
        # Check for branch analysis keywords
        branch_keywords = [
            "branch", "department", "cse", "ece", "eee", "mech",
            "which branch", "branch wise", "per branch"
        ]
        if any(kw in query_lower for kw in branch_keywords):
            return QueryType.BRANCH_ANALYSIS
        
        # Check for skill analysis keywords
        skill_keywords = [
            "skill", "skills required", "skills needed", "in demand",
            "skill demand", "learn", "technologies"
        ]
        if any(kw in query_lower for kw in skill_keywords):
            return QueryType.SKILL_ANALYSIS
        
        # Check for company clustering keywords
        cluster_keywords = [
            "similar", "group", "cluster", "type", "category",
            "like", "companies similar to", "same domain"
        ]
        if any(kw in query_lower for kw in cluster_keywords):
            return QueryType.COMPANY_CLUSTERING
        
        # Check for recommendation keywords
        recommendation_keywords = [
            "recommend", "suggestion", "should i apply",
            "best for me", "suited for", "personalized",
            "my profile", "for my cgpa"
        ]
        if any(kw in query_lower for kw in recommendation_keywords):
            return QueryType.PERSONALIZED_RECOMMENDATION
        
        # Check for single company mention
        company_patterns = [
            r"(about|for|at|in)\s+\w+\s*(company|technologies|labs|india)?$",
            r"^(what|tell|explain|describe).*(about|is)\s+\w+"
        ]
        
        # Count company mentions using known companies
        company_mentions = sum(1 for c in self.known_companies if c.lower() in query_lower)
        
        if company_mentions == 1:
            return QueryType.SINGLE_COMPANY_INFO
        elif company_mentions > 1:
            return QueryType.MULTI_COMPANY_COMPARISON
        
        return QueryType.OPEN_ENDED
    
    def extract_companies(self, query: str) -> List[str]:
        """Extract company names mentioned in the query"""
        query_lower = query.lower()
        found_companies = []
        
        for company in self.known_companies:
            if company.lower() in query_lower:
                found_companies.append(company)
        
        # If no known companies found, try LLM extraction
        if not found_companies and self.llm_client:
            try:
                prompt = f"""Extract company names from this query. Return as JSON array.
Query: "{query}"
Known companies: {', '.join(self.known_companies[:20])}...

Return ONLY a JSON array like: ["Company1", "Company2"] or [] if none found."""

                response = self.llm_client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.1
                )
                result = response.choices[0].message.content.strip()
                # Extract JSON array from response
                match = re.search(r'\[.*?\]', result)
                if match:
                    found_companies = json.loads(match.group())
            except Exception as e:
                print(f"⚠️ Company extraction failed: {e}")
        
        return found_companies
    
    def plan(self, query: str) -> Plan:
        """
        Create an execution plan for the given query.
        
        This is the main entry point for the Planner.
        """
        # Comprehensive input validation using input_validator module
        validation_result = validate_input(query, min_length=10, max_length=1500)
        
        if not validation_result.is_valid:
            # Raise error with all validation errors
            error_msg = "; ".join(validation_result.errors)
            raise ValueError(f"Invalid input: {error_msg}")
        
        # Show warnings if any
        if validation_result.warnings:
            for warning in validation_result.warnings:
                print(f"⚠️ {warning}")
        
        # Use cleaned and sanitized query
        query = validation_result.cleaned_query
        
        # Log validation metadata in debug mode
        if os.getenv("DEBUG_MODE") == "true":
            print(f"🔍 Validation: {len(validation_result.checks_passed)} checks passed")
            if 'relevance_confidence' in validation_result.metadata:
                conf = validation_result.metadata['relevance_confidence']
                print(f"📊 Relevance confidence: {conf:.2%}")
        
        try:
            # Step 1: Classify the query
            query_type = self.classify_query(query)
            print(f"🎯 Query classified as: {query_type.value}")
        
        # Step 2: Extract entities (companies, etc.)
        companies = self.extract_companies(query)
        print(f"🏢 Companies detected: {companies}")
        
        # Step 3: Build plan based on query type
        if query_type == QueryType.SINGLE_COMPANY_INFO:
            plan = self._plan_single_company(query, companies)
        
        elif query_type == QueryType.MULTI_COMPANY_COMPARISON:
            plan = self._plan_comparison(query, companies)
        
        elif query_type == QueryType.STATISTICAL_ANALYSIS:
            plan = self._plan_statistics(query)
        
        elif query_type == QueryType.ELIGIBILITY_CHECK:
            plan = self._plan_eligibility(query, companies)
        
        elif query_type == QueryType.INTERVIEW_PREP:
            plan = self._plan_interview_prep(query, companies)
        
        elif query_type == QueryType.TIMELINE_INFO:
            plan = self._plan_timeline(query, companies)
        
        elif query_type == QueryType.SALARY_NEGOTIATION:
            plan = self._plan_salary_negotiation(query, companies)
        
        elif query_type == QueryType.OFFER_COMPARISON:
            plan = self._plan_offer_comparison(query, companies)
        
        elif query_type == QueryType.CAREER_PATH:
            plan = self._plan_career_path(query, companies)
        
        elif query_type == QueryType.LOCATION_BASED:
            plan = self._plan_location_based(query)
        
        elif query_type == QueryType.TREND_ANALYSIS:
            plan = self._plan_trend_analysis(query)
        
        elif query_type == QueryType.BRANCH_ANALYSIS:
            plan = self._plan_branch_analysis(query)
        
        elif query_type == QueryType.SKILL_ANALYSIS:
            plan = self._plan_skill_analysis(query)
        
        elif query_type == QueryType.COMPANY_CLUSTERING:
            plan = self._plan_company_clustering(query, companies)
        
        elif query_type == QueryType.PERSONALIZED_RECOMMENDATION:
            plan = self._plan_personalized_recommendation(query)
        
        else:  # OPEN_ENDED
            plan = self._plan_open_ended(query)
        
            # Step 4: Apply feedback if any (for re-planning)
            if self.feedback_history:
                plan = self._apply_feedback(plan)
            
            print(plan)
            return plan
        
        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            print(f"❌ Error in planning: {e}")
            # Return a fallback plan for open-ended query
            return self._plan_open_ended(query)
    
    def _plan_single_company(self, query: str, companies: List[str]) -> Plan:
        """Plan for single company information queries"""
        company = companies[0] if companies else None
        
        steps = [
            Step(
                tool="vector_search",
                args={
                    "query": query,
                    "company_filter": company,
                    "top_k": 5
                },
                description=f"Search placement docs for {company or 'relevant company'}"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.context}",
                    "sources": "{step_0.sources}",
                    "format_type": "prose"
                },
                depends_on=[0],
                description="Generate detailed answer from retrieved context"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.SINGLE_COMPANY_INFO,
            steps=steps,
            metadata={"companies": companies}
        )
    
    def _plan_comparison(self, query: str, companies: List[str]) -> Plan:
        """Plan for multi-company comparison queries"""
        steps = []
        
        # Step 0: Extract companies if not already found
        if not companies:
            steps.append(Step(
                tool="company_extractor",
                args={"query": query},
                description="Extract company names from query"
            ))
            search_start_idx = 1
        else:
            search_start_idx = 0
        
        # Search for each company (can run in parallel)
        for i, company in enumerate(companies or ["company_1", "company_2"]):
            steps.append(Step(
                tool="vector_search",
                args={
                    "query": f"CTC salary eligibility roles selection process for {company}",
                    "company_filter": company if companies else f"{{step_0.companies[{i}]}}",
                    "top_k": 3
                },
                depends_on=[0] if not companies else [],
                description=f"Search docs for {company}"
            ))
        
        # Comparison step
        search_end_idx = search_start_idx + len(companies or [2])
        steps.append(Step(
            tool="comparison",
            args={
                "query": query,
                "companies": companies or "{step_0.companies}",
                "contexts": [f"{{step_{i}.context}}" for i in range(search_start_idx, search_end_idx)]
            },
            depends_on=list(range(search_start_idx, search_end_idx)),
            description="Generate comparison table"
        ))
        
        return Plan(
            query=query,
            query_type=QueryType.MULTI_COMPANY_COMPARISON,
            steps=steps,
            execution_mode="parallel",  # Search steps can run in parallel
            metadata={"companies": companies}
        )
    
    def _plan_statistics(self, query: str) -> Plan:
        """Plan for statistical analysis queries"""
        steps = [
            Step(
                tool="sql_query",
                args={
                    "query": query,
                    "operation": "aggregate"
                },
                description="Query structured placement data"
            ),
            Step(
                tool="vector_search",
                args={
                    "query": query,
                    "top_k": 3
                },
                description="Search for supporting context"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "structured_data": "{step_0.result}",
                    "context": "{step_1.context}",
                    "sources": "{step_1.sources}",
                    "format_type": "table"
                },
                depends_on=[0, 1],
                description="Generate statistical answer with data"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.STATISTICAL_ANALYSIS,
            steps=steps,
            execution_mode="parallel",  # SQL and vector search can run in parallel
            metadata={}
        )
    
    def _plan_eligibility(self, query: str, companies: List[str]) -> Plan:
        """Plan for eligibility check queries"""
        company = companies[0] if companies else None
        
        steps = [
            Step(
                tool="vector_search",
                args={
                    "query": f"eligibility criteria CGPA requirements {company or ''}",
                    "company_filter": company,
                    "top_k": 5
                },
                description="Search for eligibility criteria"
            ),
            Step(
                tool="eligibility",
                args={
                    "query": query,
                    "criteria_context": "{step_0.context}",
                    "company": company
                },
                depends_on=[0],
                description="Check eligibility against criteria"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.ELIGIBILITY_CHECK,
            steps=steps,
            metadata={"companies": companies}
        )
    
    def _plan_interview_prep(self, query: str, companies: List[str]) -> Plan:
        """Plan for interview preparation queries"""
        company = companies[0] if companies else None
        
        steps = [
            Step(
                tool="vector_search",
                args={
                    "query": f"interview process rounds questions {company or ''}",
                    "company_filter": company,
                    "top_k": 5
                },
                description="Search for interview information"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.context}",
                    "sources": "{step_0.sources}",
                    "format_type": "bullet_points"
                },
                depends_on=[0],
                description="Generate interview prep guide"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.INTERVIEW_PREP,
            steps=steps,
            metadata={"companies": companies}
        )
    
    def _plan_timeline(self, query: str, companies: List[str]) -> Plan:
        """Plan for timeline/schedule queries"""
        steps = [
            Step(
                tool="vector_search",
                args={
                    "query": f"date schedule deadline timeline {query}",
                    "top_k": 5
                },
                description="Search for timeline information"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.context}",
                    "sources": "{step_0.sources}",
                    "format_type": "table"
                },
                depends_on=[0],
                description="Format timeline response"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.TIMELINE_INFO,
            steps=steps,
            metadata={"companies": companies}
        )
    
    def _plan_open_ended(self, query: str) -> Plan:
        """Plan for general open-ended queries"""
        steps = [
            Step(
                tool="vector_search",
                args={
                    "query": query,
                    "top_k": 7
                },
                description="Broad search for relevant information"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.context}",
                    "sources": "{step_0.sources}",
                    "format_type": "prose"
                },
                depends_on=[0],
                description="Generate comprehensive answer"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.OPEN_ENDED,
            steps=steps,
            metadata={}
        )
    
    def add_feedback(self, evaluation: 'Evaluation'):
        """Add feedback from Critic for potential re-planning"""
        self.feedback_history.append({
            "reason": evaluation.reason,
            "checks": evaluation.checks,
            "decision": evaluation.decision
        })
    
    def _apply_feedback(self, plan: Plan) -> Plan:
        """Modify plan based on feedback from Critic"""
        if not self.feedback_history:
            return plan
        
        last_feedback = self.feedback_history[-1]
        
        # If hallucination detected, increase top_k for more context
        if "hallucination" in str(last_feedback.get("checks", {})):
            for step in plan.steps:
                if step.tool == "vector_search":
                    step.args["top_k"] = step.args.get("top_k", 5) + 3
        
        # If incomplete, add more search steps
        if "incomplete" in last_feedback.get("reason", "").lower():
            # Add a broader search step
            plan.steps.insert(0, Step(
                tool="vector_search",
                args={
                    "query": plan.query,
                    "top_k": 10
                },
                description="Broader search for more context"
            ))
        
        return plan
    
    def clear_feedback(self):
        """Clear feedback history after successful completion"""
        self.feedback_history = []
    
    def _plan_salary_negotiation(self, query: str, companies: List[str]) -> Plan:
        """Plan for salary negotiation queries"""
        company = companies[0] if companies else None
        
        steps = [
            Step(
                tool="vector_search",
                args={
                    "query": f"salary negotiation tips strategies {company or ''}",
                    "company_filter": company,
                    "top_k": 5
                },
                description="Search for negotiation strategies"
            ),
            Step(
                tool="sql_query",
                args={
                    "query": f"average salaries market rates {company or ''}",
                    "operation": "aggregate"
                },
                description="Get market salary data"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.context}",
                    "structured_data": "{step_1.result}",
                    "sources": "{step_0.sources}",
                    "format_type": "advice"
                },
                depends_on=[0, 1],
                description="Generate negotiation advice"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.SALARY_NEGOTIATION,
            steps=steps,
            execution_mode="parallel",
            metadata={"companies": companies}
        )
    
    def _plan_offer_comparison(self, query: str, companies: List[str]) -> Plan:
        """Plan for comparing multiple job offers"""
        steps = []
        
        # Extract company names if not found
        if not companies:
            steps.append(Step(
                tool="company_extractor",
                args={"query": query},
                description="Extract companies from offers"
            ))
            search_start_idx = 1
        else:
            search_start_idx = 0
        
        # Search for each company
        for i, company in enumerate(companies or ["company_1", "company_2"]):
            steps.append(Step(
                tool="vector_search",
                args={
                    "query": f"work culture growth opportunities reviews {company}",
                    "company_filter": company if companies else f"{{step_0.companies[{i}]}}",
                    "top_k": 3
                },
                depends_on=[0] if not companies else [],
                description=f"Search culture and growth data for {company}"
            ))
        
        # Comparison with pros/cons
        search_end_idx = search_start_idx + len(companies or [2])
        steps.append(Step(
            tool="comparison",
            args={
                "query": query,
                "companies": companies or "{step_0.companies}",
                "contexts": [f"{{step_{i}.context}}" for i in range(search_start_idx, search_end_idx)],
                "format_type": "pros_cons"
            },
            depends_on=list(range(search_start_idx, search_end_idx)),
            description="Generate offer comparison with recommendation"
        ))
        
        return Plan(
            query=query,
            query_type=QueryType.OFFER_COMPARISON,
            steps=steps,
            execution_mode="parallel",
            metadata={"companies": companies}
        )
    
    def _plan_career_path(self, query: str, companies: List[str]) -> Plan:
        """Plan for career path and growth queries"""
        company = companies[0] if companies else None
        
        steps = [
            Step(
                tool="vector_search",
                args={
                    "query": f"career growth path progression roles {company or ''}",
                    "company_filter": company,
                    "top_k": 5
                },
                description="Search career progression info"
            ),
            Step(
                tool="vector_search",
                args={
                    "query": "skill requirements advancement learning opportunities",
                    "company_filter": company,
                    "top_k": 3
                },
                description="Search skill development info"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.context} {step_1.context}",
                    "sources": "{step_0.sources}",
                    "format_type": "timeline"
                },
                depends_on=[0, 1],
                description="Generate career path timeline"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.CAREER_PATH,
            steps=steps,
            execution_mode="parallel",
            metadata={"companies": companies}
        )
    
    def _plan_location_based(self, query: str) -> Plan:
        """Plan for location-based queries"""
        steps = [
            Step(
                tool="sql_query",
                args={
                    "query": query,
                    "operation": "filter"
                },
                description="Filter companies by location"
            ),
            Step(
                tool="vector_search",
                args={
                    "query": f"{query} location work environment",
                    "top_k": 5
                },
                description="Search location-specific info"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "structured_data": "{step_0.result}",
                    "context": "{step_1.context}",
                    "sources": "{step_1.sources}",
                    "format_type": "table"
                },
                depends_on=[0, 1],
                description="Generate location-wise listing"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.LOCATION_BASED,
            steps=steps,
            execution_mode="parallel",
            metadata={}
        )
    
    def _plan_trend_analysis(self, query: str) -> Plan:
        """Plan for trend analysis queries"""
        steps = [
            Step(
                tool="trend_analyzer",
                args={
                    "query": query
                },
                description="Analyze trends over years"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.response}",
                    "structured_data": "{step_0.skill_rankings}",
                    "sources": ["Placement Data Analysis"],
                    "format_type": "table"
                },
                depends_on=[0],
                description="Format trend analysis results"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.TREND_ANALYSIS,
            steps=steps,
            execution_mode="sequential",
            metadata={}
        )
    
    def _plan_branch_analysis(self, query: str) -> Plan:
        """Plan for branch-wise analysis queries"""
        steps = [
            Step(
                tool="branch_stats",
                args={
                    "query": query
                },
                description="Analyze branch-wise statistics"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.response}",
                    "structured_data": "{step_0.branch_stats}",
                    "sources": ["Branch-wise Analysis"],
                    "format_type": "table"
                },
                depends_on=[0],
                description="Format branch analysis results"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.BRANCH_ANALYSIS,
            steps=steps,
            execution_mode="sequential",
            metadata={}
        )
    
    def _plan_skill_analysis(self, query: str) -> Plan:
        """Plan for skill demand analysis queries"""
        steps = [
            Step(
                tool="skill_demand",
                args={
                    "query": query,
                    "skill_category": "all",
                    "top_k": 10
                },
                description="Analyze skill demand"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.response}",
                    "structured_data": "{step_0.skill_rankings}",
                    "sources": ["Skill Demand Analysis"],
                    "format_type": "table"
                },
                depends_on=[0],
                description="Format skill analysis results"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.SKILL_ANALYSIS,
            steps=steps,
            execution_mode="sequential",
            metadata={}
        )
    
    def _plan_company_clustering(self, query: str, companies: List[str]) -> Plan:
        """Plan for company clustering queries"""
        # Determine clustering type from query
        query_lower = query.lower()
        if "similar" in query_lower and companies:
            cluster_by = "similar"
            company_name = companies[0] if companies else None
        elif "ctc" in query_lower or "salary" in query_lower:
            cluster_by = "ctc"
            company_name = None
        elif "culture" in query_lower or "work" in query_lower:
            cluster_by = "culture"
            company_name = None
        else:
            cluster_by = "domain"
            company_name = None
        
        args = {"cluster_by": cluster_by}
        if company_name:
            args["company_name"] = company_name
        
        steps = [
            Step(
                tool="company_cluster",
                args=args,
                description=f"Cluster companies by {cluster_by}"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.response}",
                    "structured_data": "{step_0.clusters}",
                    "sources": ["Company Clustering"],
                    "format_type": "table"
                },
                depends_on=[0],
                description="Format clustering results"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.COMPANY_CLUSTERING,
            steps=steps,
            execution_mode="sequential",
            metadata={"cluster_by": cluster_by}
        )
    
    def _plan_personalized_recommendation(self, query: str) -> Plan:
        """Plan for personalized recommendation queries"""
        # Extract student profile from query (simplified)
        query_lower = query.lower()
        
        # Extract CGPA
        cgpa_match = re.search(r'(\d+(?:\.\d+)?)\s*cgpa', query_lower)
        cgpa = float(cgpa_match.group(1)) if cgpa_match else None
        
        # Extract branch
        branches = ["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT", "IS", "AIML"]
        branch = None
        for b in branches:
            if b.lower() in query_lower:
                branch = b
                break
        
        # Build args
        args = {"top_k": 5}
        if cgpa:
            args["cgpa"] = cgpa
        if branch:
            args["branch"] = branch
        
        steps = [
            Step(
                tool="recommendation_engine",
                args=args,
                description="Generate personalized recommendations"
            ),
            Step(
                tool="answer_generator",
                args={
                    "query": query,
                    "context": "{step_0.response}",
                    "structured_data": "{step_0.recommendations}",
                    "sources": ["Personalized Recommendation"],
                    "format_type": "table"
                },
                depends_on=[0],
                description="Format recommendation results"
            )
        ]
        
        return Plan(
            query=query,
            query_type=QueryType.PERSONALIZED_RECOMMENDATION,
            steps=steps,
            execution_mode="sequential",
            metadata={"cgpa": cgpa, "branch": branch}
        )


# Example usage
if __name__ == "__main__":
    planner = Planner(use_llm_classification=False)
    
    # Test queries
    test_queries = [
        "What is Amazon's CTC?",
        "Compare Google vs Microsoft vs Amazon",
        "What is the average CTC of all companies?",
        "Am I eligible for Intel with 8.5 CGPA?",
        "How to prepare for Nvidia interview?",
        "When is the next company visiting?",
        "Tell me about placement process"
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        plan = planner.plan(query)
