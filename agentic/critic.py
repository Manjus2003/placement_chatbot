"""
🔍 CRITIC MODULE
================
Responsible for:
1. Completeness Check - Does answer address all parts of the query?
2. Factuality Check - Are facts grounded in sources?
3. Hallucination Detection - Is anything made up?
4. Citation Verification - Are sources properly referenced?
5. Relevance Check - Is the answer on-topic?
6. Decision Making - Accept, Refine, or Re-plan?

The Critic evaluates answers and decides whether to:
- ACCEPT: Answer is good, return to user
- REFINE: Answer needs minor improvements
- REPLAN: Answer is fundamentally wrong, need different approach
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI


class Decision(Enum):
    """Critic's decision on answer quality"""
    ACCEPT = "accept"      # Answer is good, return to user
    REFINE = "refine"      # Minor improvements needed
    REPLAN = "replan"      # Need to search again with different strategy


@dataclass
class CheckResult:
    """Result of a single quality check"""
    name: str
    passed: bool
    score: float = 0.0
    details: str = ""
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "issues": self.issues
        }


@dataclass
class Evaluation:
    """Complete evaluation of an answer"""
    decision: Decision
    checks: Dict[str, CheckResult]
    reason: str = ""
    confidence: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
            "reason": self.reason,
            "confidence": self.confidence,
            "suggestions": self.suggestions
        }
    
    def __str__(self) -> str:
        lines = [
            f"🔍 Evaluation Result: {self.decision.value.upper()}",
            f"   Confidence: {self.confidence:.1%}",
            f"   Reason: {self.reason}",
            "   Checks:"
        ]
        for name, check in self.checks.items():
            status = "✅" if check.passed else "❌"
            lines.append(f"     {status} {name}: {check.score:.0%}")
        return "\n".join(lines)


class Critic:
    """
    Evaluates answer quality and decides on next action.
    
    The Critic performs multiple quality checks and makes a decision
    about whether to accept, refine, or re-plan.
    """
    
    def __init__(
        self,
        completeness_threshold: float = 0.7,
        factuality_threshold: float = 0.8,
        hallucination_threshold: float = 0.9
    ):
        """
        Initialize the Critic.
        
        Args:
            completeness_threshold: Minimum completeness score to pass
            factuality_threshold: Minimum factuality score to pass
            hallucination_threshold: Minimum hallucination-free score to pass
        """
        load_dotenv(dotenv_path="environment.env")
        
        self.completeness_threshold = completeness_threshold
        self.factuality_threshold = factuality_threshold
        self.hallucination_threshold = hallucination_threshold
        
        # Initialize LLM client
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            self.llm_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
            )
            print("✅ Critic initialized with LLM evaluation")
        else:
            self.llm_client = None
            print("✅ Critic initialized with rule-based evaluation")
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        sources: List[str],
        plan: Optional['Plan'] = None
    ) -> Evaluation:
        """
        Evaluate an answer's quality.
        
        This is the main entry point for the Critic.
        
        Args:
            query: Original user query
            answer: Generated answer to evaluate
            context: Retrieved context used to generate answer
            sources: List of source documents
            plan: Optional plan used (for context)
            
        Returns:
            Evaluation with checks and decision
        """
        print("\n🔍 Evaluating answer quality...")
        
        # Run all checks
        checks = {
            "completeness": self._check_completeness(query, answer),
            "factuality": self._check_factuality(answer, context, sources),
            "hallucination": self._check_hallucination(answer, context),
            "relevance": self._check_relevance(query, answer),
            "citations": self._check_citations(answer, sources)
        }
        
        # Print check results
        for name, check in checks.items():
            status = "✅" if check.passed else "❌"
            print(f"   {status} {name}: {check.score:.0%}")
        
        # Make decision based on checks
        decision, reason, confidence = self._make_decision(checks)
        
        # Generate suggestions for improvement
        suggestions = self._generate_suggestions(checks, answer)
        
        evaluation = Evaluation(
            decision=decision,
            checks=checks,
            reason=reason,
            confidence=confidence,
            suggestions=suggestions
        )
        
        print(f"\n   📋 Decision: {decision.value.upper()}")
        print(f"   💭 Reason: {reason}")
        
        return evaluation
    
    def _check_completeness(self, query: str, answer: str) -> CheckResult:
        """
        Check if the answer fully addresses the query.
        
        Evaluates whether all aspects of the question are answered.
        """
        if self.llm_client:
            return self._check_completeness_llm(query, answer)
        return self._check_completeness_rules(query, answer)
    
    def _check_completeness_llm(self, query: str, answer: str) -> CheckResult:
        """Use LLM to check completeness"""
        prompt = f"""Evaluate if this answer completely addresses the query.

Query: "{query}"

Answer: "{answer}"

Check:
1. Does the answer address the main question?
2. Are all sub-questions (if any) answered?
3. Is sufficient detail provided?

Return a JSON object:
{{
    "score": <0-100, where 100 is fully complete>,
    "addressed_aspects": ["aspect1", "aspect2"],
    "missing_aspects": ["missing1", "missing2"],
    "details": "brief explanation"
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = result.get("score", 0) / 100
                return CheckResult(
                    name="completeness",
                    passed=score >= self.completeness_threshold,
                    score=score,
                    details=result.get("details", ""),
                    issues=result.get("missing_aspects", [])
                )
        except Exception as e:
            print(f"⚠️ LLM completeness check failed: {e}")
        
        return self._check_completeness_rules(query, answer)
    
    def _check_completeness_rules(self, query: str, answer: str) -> CheckResult:
        """Rule-based completeness check"""
        issues = []
        score = 1.0
        
        # Check 1: Answer length (too short is incomplete)
        if len(answer) < 50:
            score -= 0.3
            issues.append("Answer is too short")
        
        # Check 2: Question words should be addressed
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        query_lower = query.lower()
        
        for qw in question_words:
            if qw in query_lower:
                # Check if answer seems to address this
                if qw == "what" and not any(w in answer.lower() for w in ["is", "are", "means", ":"]):
                    score -= 0.1
                    issues.append(f"'{qw}' question may not be addressed")
                break
        
        # Check 3: If query mentions specific entities, they should appear in answer
        # Simple check: numbers mentioned in query should appear in answer
        query_numbers = re.findall(r'\d+', query)
        for num in query_numbers:
            if num not in answer:
                score -= 0.1
                issues.append(f"Number '{num}' from query not in answer")
        
        # Check 4: "I don't know" type responses
        uncertain_phrases = ["i don't know", "i'm not sure", "no information", "cannot find"]
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            score -= 0.4
            issues.append("Answer expresses uncertainty")
        
        score = max(0, score)
        
        return CheckResult(
            name="completeness",
            passed=score >= self.completeness_threshold,
            score=score,
            details=f"Rule-based check found {len(issues)} issues",
            issues=issues
        )
    
    def _check_factuality(self, answer: str, context: str, sources: List[str]) -> CheckResult:
        """
        Check if facts in the answer are grounded in context.
        """
        if self.llm_client:
            return self._check_factuality_llm(answer, context)
        return self._check_factuality_rules(answer, context)
    
    def _check_factuality_llm(self, answer: str, context: str) -> CheckResult:
        """Use LLM to check factuality"""
        prompt = f"""Verify if ALL facts in the answer are supported by the context.

Context:
\"\"\"
{context[:2000]}
\"\"\"

Answer to verify:
\"\"\"
{answer}
\"\"\"

For each factual claim in the answer, check if it's supported by the context.

Return a JSON object:
{{
    "score": <0-100, where 100 means all facts are grounded>,
    "supported_facts": ["fact1", "fact2"],
    "unsupported_facts": ["fact1", "fact2"],
    "details": "brief explanation"
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = result.get("score", 0) / 100
                return CheckResult(
                    name="factuality",
                    passed=score >= self.factuality_threshold,
                    score=score,
                    details=result.get("details", ""),
                    issues=result.get("unsupported_facts", [])
                )
        except Exception as e:
            print(f"⚠️ LLM factuality check failed: {e}")
        
        return self._check_factuality_rules(answer, context)
    
    def _check_factuality_rules(self, answer: str, context: str) -> CheckResult:
        """Rule-based factuality check"""
        issues = []
        
        # Extract numbers from answer and check if they appear in context
        answer_numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:LPA|lpa|lakhs?|crores?))?', answer)
        context_lower = context.lower()
        
        matched = 0
        for num in answer_numbers:
            # Check if number (or close variant) appears in context
            num_value = re.search(r'\d+(?:\.\d+)?', num)
            if num_value:
                if num_value.group() in context_lower:
                    matched += 1
                else:
                    issues.append(f"Number '{num}' not found in context")
        
        # Score based on number matching
        if answer_numbers:
            score = matched / len(answer_numbers)
        else:
            score = 0.8  # No numbers to check, assume somewhat factual
        
        return CheckResult(
            name="factuality",
            passed=score >= self.factuality_threshold,
            score=score,
            details=f"Checked {len(answer_numbers)} numbers, {matched} found in context",
            issues=issues
        )
    
    def _check_hallucination(self, answer: str, context: str) -> CheckResult:
        """
        Detect hallucinated content not present in context.
        """
        if self.llm_client:
            return self._check_hallucination_llm(answer, context)
        return self._check_hallucination_rules(answer, context)
    
    def _check_hallucination_llm(self, answer: str, context: str) -> CheckResult:
        """Use LLM to detect hallucinations"""
        prompt = f"""Detect any HALLUCINATIONS in the answer - information that is made up and NOT in the context.

Context (source of truth):
\"\"\"
{context[:2000]}
\"\"\"

Answer to check:
\"\"\"
{answer}
\"\"\"

A hallucination is:
- Specific facts (names, numbers, dates) not in the context
- Claims that contradict the context
- Made-up details not supported by context

Return a JSON object:
{{
    "hallucination_free_score": <0-100, where 100 means NO hallucinations>,
    "hallucinated_items": ["item1", "item2"],
    "details": "brief explanation"
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = result.get("hallucination_free_score", 0) / 100
                return CheckResult(
                    name="hallucination",
                    passed=score >= self.hallucination_threshold,
                    score=score,
                    details=result.get("details", ""),
                    issues=result.get("hallucinated_items", [])
                )
        except Exception as e:
            print(f"⚠️ LLM hallucination check failed: {e}")
        
        return self._check_hallucination_rules(answer, context)
    
    def _check_hallucination_rules(self, answer: str, context: str) -> CheckResult:
        """Rule-based hallucination detection"""
        issues = []
        score = 1.0
        
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        # Check for specific company names in answer not in context
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Ltd|Corp|Technologies|Labs))?\b'
        answer_companies = set(re.findall(company_pattern, answer))
        
        for company in answer_companies:
            if company.lower() not in context_lower and len(company) > 3:
                # Could be hallucinated company name
                issues.append(f"Company '{company}' not found in context")
                score -= 0.15
        
        # Check for specific dates/years
        answer_years = re.findall(r'\b20\d{2}\b', answer)
        for year in answer_years:
            if year not in context:
                issues.append(f"Year '{year}' not in context")
                score -= 0.1
        
        score = max(0, score)
        
        return CheckResult(
            name="hallucination",
            passed=score >= self.hallucination_threshold,
            score=score,
            details=f"Checked for hallucinated entities",
            issues=issues
        )
    
    def _check_relevance(self, query: str, answer: str) -> CheckResult:
        """Check if the answer is relevant to the query"""
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Extract key terms from query
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 
                      'when', 'where', 'who', 'which', 'do', 'does', 'did', 'for', 'to', 
                      'in', 'on', 'at', 'of', 'and', 'or', 'me', 'tell', 'about', 'can', 'i'}
        
        query_terms = set(query_lower.split()) - stop_words
        
        # Check how many query terms appear in answer
        matches = sum(1 for term in query_terms if term in answer_lower)
        
        if query_terms:
            score = min(1.0, matches / len(query_terms) + 0.3)  # Base relevance boost
        else:
            score = 0.8
        
        return CheckResult(
            name="relevance",
            passed=score >= 0.5,
            score=score,
            details=f"Matched {matches}/{len(query_terms)} query terms",
            issues=[]
        )
    
    def _check_citations(self, answer: str, sources: List[str]) -> CheckResult:
        """Check if answer properly cites sources"""
        # Look for citation patterns
        citation_patterns = [
            r'\[.*?\]',           # [source]
            r'Source:',           # Source: 
            r'According to',      # According to...
            r'\(.*?\.pdf\)',      # (file.pdf)
            r'from\s+\w+\.pdf',   # from file.pdf
        ]
        
        has_citations = any(re.search(p, answer, re.IGNORECASE) for p in citation_patterns)
        
        # Check if source filenames are mentioned
        sources_mentioned = 0
        for source in sources:
            # Extract filename from path
            filename = source.split('/')[-1].split('\\')[-1]
            name_parts = filename.replace('.pdf', '').replace('_', ' ').split()
            if any(part.lower() in answer.lower() for part in name_parts if len(part) > 3):
                sources_mentioned += 1
        
        if sources:
            citation_score = sources_mentioned / len(sources)
        else:
            citation_score = 0.5  # No sources to cite
        
        # Boost if has citation markers
        if has_citations:
            citation_score = min(1.0, citation_score + 0.3)
        
        return CheckResult(
            name="citations",
            passed=citation_score >= 0.3 or has_citations,
            score=citation_score,
            details=f"Found {sources_mentioned}/{len(sources)} source references",
            issues=[] if has_citations else ["No citation markers found"]
        )
    
    def _make_decision(self, checks: Dict[str, CheckResult]) -> tuple:
        """
        Make final decision based on all checks.
        
        Returns:
            (Decision, reason, confidence)
        """
        # Calculate overall score
        weights = {
            "completeness": 0.25,
            "factuality": 0.25,
            "hallucination": 0.30,
            "relevance": 0.15,
            "citations": 0.05
        }
        
        overall_score = sum(
            checks[name].score * weight 
            for name, weight in weights.items()
        )
        
        # Critical failures
        if not checks["hallucination"].passed:
            return (
                Decision.REPLAN,
                "Hallucinations detected - need better sources",
                overall_score
            )
        
        if checks["relevance"].score < 0.3:
            return (
                Decision.REPLAN,
                "Answer not relevant to query",
                overall_score
            )
        
        # Minor issues - refine
        if not checks["completeness"].passed:
            return (
                Decision.REFINE,
                "Answer incomplete - needs more details",
                overall_score
            )
        
        if not checks["factuality"].passed:
            return (
                Decision.REFINE,
                "Some facts may not be grounded",
                overall_score
            )
        
        if not checks["citations"].passed and overall_score > 0.6:
            return (
                Decision.REFINE,
                "Could improve source citations",
                overall_score
            )
        
        # All good - accept
        if overall_score >= 0.7:
            return (
                Decision.ACCEPT,
                "Answer meets quality standards",
                overall_score
            )
        
        # Default to refine
        return (
            Decision.REFINE,
            "Answer could be improved",
            overall_score
        )
    
    def _generate_suggestions(
        self,
        checks: Dict[str, CheckResult],
        answer: str
    ) -> List[str]:
        """Generate improvement suggestions based on failed checks"""
        suggestions = []
        
        if not checks["completeness"].passed:
            suggestions.append("Add more details to fully answer the query")
            for issue in checks["completeness"].issues:
                suggestions.append(f"Address: {issue}")
        
        if not checks["factuality"].passed:
            suggestions.append("Verify facts against source documents")
        
        if not checks["hallucination"].passed:
            suggestions.append("Remove information not present in sources")
            for item in checks["hallucination"].issues[:3]:
                suggestions.append(f"Check: {item}")
        
        if not checks["citations"].passed:
            suggestions.append("Add source citations [Company_Name]")
        
        return suggestions
    
    def quick_check(self, answer: str, context: str) -> bool:
        """
        Quick pass/fail check without full evaluation.
        
        Useful for fast rejection of obviously bad answers.
        """
        # Check 1: Not empty
        if not answer or len(answer.strip()) < 20:
            return False
        
        # Check 2: Not just "I don't know"
        negative_patterns = [
            r"^i don'?t know",
            r"^no information",
            r"^i cannot",
            r"^sorry,? i",
        ]
        for pattern in negative_patterns:
            if re.match(pattern, answer.lower().strip()):
                return False
        
        # Check 3: Has some overlap with context
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(context_words & answer_words)
        
        if overlap < 5:
            return False
        
        return True


# Example usage
if __name__ == "__main__":
    critic = Critic()
    
    # Test evaluation
    query = "What is Amazon's CTC?"
    answer = "Amazon offers a CTC of 28 LPA for MTech students."
    context = "Amazon visited campus in 2025. The CTC offered was 28 LPA. Eligibility: 7.5 CGPA minimum."
    sources = ["Amazon_MTech_2026/placement_details.pdf"]
    
    evaluation = critic.evaluate(query, answer, context, sources)
    print(evaluation)
