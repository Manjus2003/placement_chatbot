"""
🤖 AGENTIC RAG ORCHESTRATOR
===========================
Main orchestrator that combines Planner, Executor, and Critic
into a complete agentic RAG system.

This is the main entry point for the agentic system.

Flow:
1. User Query → Planner (creates execution plan)
2. Plan → Executor (runs tools step by step)
3. Result → Critic (evaluates quality)
4. Decision → Accept / Refine / Replan
5. Final Answer → User
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .planner import Planner, Plan, QueryType
from .executor import Executor, ExecutionResult
from .critic import Critic, Evaluation, Decision


@dataclass
class AgenticResponse:
    """Response from the agentic system"""
    answer: str
    success: bool
    query_type: str
    iterations: int
    total_time: float
    plan: Optional[Plan] = None
    evaluation: Optional[Evaluation] = None
    sources: List[str] = field(default_factory=list)
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "success": self.success,
            "query_type": self.query_type,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "sources": self.sources,
            "context": self.context[:500] if self.context else "",
            "metadata": self.metadata
        }


class AgenticRAG:
    """
    Main orchestrator for the agentic RAG system.
    
    Coordinates between Planner, Executor, and Critic to provide
    high-quality, validated answers.
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        use_llm_classification: bool = True,
        query_helper=None,
        llm_reasoner=None
    ):
        """
        Initialize the Agentic RAG system.
        
        Args:
            max_iterations: Maximum number of plan/execute/evaluate cycles
            use_llm_classification: Whether to use LLM for query classification
            query_helper: Existing QueryHelper instance (reuse for efficiency)
            llm_reasoner: Existing LLMReasoner instance (reuse for efficiency)
        """
        print("\n" + "="*60)
        print("🤖 Initializing Agentic RAG System")
        print("="*60)
        
        self.max_iterations = max_iterations
        
        # Initialize components
        print("\n📋 Initializing Planner...")
        self.planner = Planner(use_llm_classification=use_llm_classification)
        
        print("\n⚙️ Initializing Executor...")
        self.executor = Executor()
        
        print("\n🔍 Initializing Critic...")
        self.critic = Critic()
        
        # Initialize and register tools
        print("\n🔧 Loading Tools...")
        self._initialize_tools(query_helper, llm_reasoner)
        
        # Store known companies for the planner
        self._sync_companies()
        
        print("\n" + "="*60)
        print("✅ Agentic RAG System Ready!")
        print("="*60 + "\n")
    
    def _initialize_tools(self, query_helper=None, llm_reasoner=None):
        """Initialize and register all tools"""
        
        from .tools import (
            VectorSearchTool,
            CompanyExtractorTool,
            ComparisonTool,
            SQLQueryTool,
            EligibilityTool,
            AnswerGeneratorTool,
            TrendAnalyzerTool,
            BranchWiseStatsTool,
            SkillDemandTool,
            CompanyClusterTool,
            RecommendationEngineTool
        )
        
        # Vector Search Tool (wraps existing QueryHelper)
        self.vector_search_tool = VectorSearchTool(query_helper)
        self.executor.register_tool("vector_search", self.vector_search_tool)
        
        # Company Extractor Tool
        self.company_extractor_tool = CompanyExtractorTool()
        self.executor.register_tool("company_extractor", self.company_extractor_tool)
        
        # Comparison Tool
        self.comparison_tool = ComparisonTool()
        self.executor.register_tool("comparison", self.comparison_tool)
        
        # SQL Query Tool
        self.sql_tool = SQLQueryTool()
        self.executor.register_tool("sql_query", self.sql_tool)
        
        # Eligibility Tool
        self.eligibility_tool = EligibilityTool()
        self.executor.register_tool("eligibility", self.eligibility_tool)
        
        # Answer Generator Tool
        self.answer_generator_tool = AnswerGeneratorTool(llm_reasoner)
        self.executor.register_tool("answer_generator", self.answer_generator_tool)
        
        # Advanced Analytics Tools
        self.trend_analyzer_tool = TrendAnalyzerTool(self.vector_search_tool)
        self.executor.register_tool("trend_analyzer", self.trend_analyzer_tool)
        
        self.branch_stats_tool = BranchWiseStatsTool(self.vector_search_tool)
        self.executor.register_tool("branch_stats", self.branch_stats_tool)
        
        self.skill_demand_tool = SkillDemandTool(self.vector_search_tool)
        self.executor.register_tool("skill_demand", self.skill_demand_tool)
        
        self.company_cluster_tool = CompanyClusterTool(self.vector_search_tool)
        self.executor.register_tool("company_cluster", self.company_cluster_tool)
        
        self.recommendation_engine_tool = RecommendationEngineTool(
            self.vector_search_tool, 
            self.skill_demand_tool
        )
        self.executor.register_tool("recommendation_engine", self.recommendation_engine_tool)
    
    def _sync_companies(self):
        """Sync known companies from vector search to other tools"""
        try:
            companies = self.vector_search_tool.get_all_companies()
            if companies:
                self.planner.set_known_companies(companies)
                self.company_extractor_tool.set_known_companies(companies)
                print(f"📊 Synced {len(companies)} companies across tools")
        except Exception as e:
            print(f"⚠️ Could not sync companies: {e}")
    
    def answer_query(self, query: str, verbose: bool = True) -> AgenticResponse:
        """
        Answer a user query using the agentic system.
        
        This is the main entry point for queries.
        
        Args:
            query: User's question
            verbose: Whether to print detailed progress
            
        Returns:
            AgenticResponse with answer and metadata
        """
        start_time = time.time()
        iteration = 0
        
        if verbose:
            print("\n" + "="*60)
            print(f"🎯 Processing Query: {query[:60]}...")
            print("="*60)
        
        best_answer = ""
        best_evaluation = None
        last_plan = None
        final_sources = []
        final_context = ""
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n🔄 Iteration {iteration}/{self.max_iterations}")
                print("-" * 40)
            
            # ========== STEP 1: PLAN ==========
            if verbose:
                print("\n📋 PLANNING...")
            
            try:
                plan = self.planner.plan(query)
                last_plan = plan
            except Exception as e:
                print(f"❌ Planning failed: {e}")
                return AgenticResponse(
                    query=query,
                    answer=f"I apologize, but I encountered an error while planning your query: {str(e)}. Please try rephrasing your question or simplifying it.",
                    sources=[],
                    metadata={"error": str(e), "stage": "planning"},
                    execution_time=time.time() - start_time,
                    iterations=iteration
                )
            
            # ========== STEP 2: EXECUTE ==========
            if verbose:
                print("\n⚙️ EXECUTING...")
            
            try:
                execution_result = self.executor.execute(plan)
            except Exception as e:
                print(f"❌ Execution failed: {e}")
                if best_answer:  # Return best answer from previous iteration
                    print("↩️ Returning best answer from previous iteration")
                    break
                return AgenticResponse(
                    query=query,
                    answer=f"I apologize, but I encountered an error while gathering information: {str(e)}. Please try again.",
                    sources=[],
                    metadata={"error": str(e), "stage": "execution"},
                    execution_time=time.time() - start_time,
                    iterations=iteration
                )
            
            if not execution_result.success:
                print(f"❌ Execution failed: {execution_result.error}")
                # Try again with modified plan
                self.planner.add_feedback(Evaluation(
                    decision=Decision.REPLAN,
                    checks={},
                    reason=f"Execution failed: {execution_result.error}"
                ))
                continue
            
            answer = execution_result.answer
            final_sources = execution_result.sources
            final_context = execution_result.context
            
            if not answer:
                # Try to extract answer from final output
                final_output = execution_result.final_output
                if isinstance(final_output, dict):
                    answer = final_output.get("answer", final_output.get("response", ""))
                elif isinstance(final_output, str):
                    answer = final_output
            
            if verbose:
                print(f"\n📝 Draft Answer: {answer[:100]}...")
            
            # ========== STEP 3: EVALUATE ==========
            if verbose:
                print("\n🔍 EVALUATING...")
            
            try:
                evaluation = self.critic.evaluate(
                    query=query,
                    answer=answer,
                    context=final_context,
                    sources=final_sources,
                    plan=plan
                )
                best_evaluation = evaluation
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
                # Proceed with answer anyway
                evaluation = Evaluation(
                    decision=Decision.ACCEPT,
                    checks={},
                    reason="Evaluation skipped"
                )
            
            # ========== STEP 4: DECIDE ==========
            if evaluation.decision == Decision.ACCEPT:
                if verbose:
                    print("\n✅ Answer ACCEPTED!")
                best_answer = answer
                break
            
            elif evaluation.decision == Decision.REFINE:
                if verbose:
                    print(f"\n✨ REFINING: {evaluation.reason}")
                
                # Apply refinement
                refined = self._refine_answer(
                    query, answer, final_context, final_sources, evaluation
                )
                best_answer = refined
                
                # Check if refinement is acceptable
                if self.critic.quick_check(refined, final_context):
                    if verbose:
                        print("✅ Refined answer accepted!")
                    break
            
            elif evaluation.decision == Decision.REPLAN:
                if verbose:
                    print(f"\n🔄 RE-PLANNING: {evaluation.reason}")
                
                # Add feedback for next iteration
                self.planner.add_feedback(evaluation)
                best_answer = answer  # Keep as fallback
        
        # ========== FINAL RESPONSE ==========
        total_time = time.time() - start_time
        
        if not best_answer:
            best_answer = "I couldn't find a satisfactory answer to your question. Please try rephrasing."
        
        # Clear feedback for next query
        self.planner.clear_feedback()
        
        response = AgenticResponse(
            answer=best_answer,
            success=bool(best_answer and best_answer != ""),
            query_type=last_plan.query_type.value if last_plan else "unknown",
            iterations=iteration,
            total_time=total_time,
            plan=last_plan,
            evaluation=best_evaluation,
            sources=final_sources,
            context=final_context,
            metadata={
                "plan_steps": len(last_plan.steps) if last_plan else 0,
                "companies_detected": last_plan.metadata.get("companies", []) if last_plan else []
            }
        )
        
        if verbose:
            print("\n" + "="*60)
            print("📊 RESULT SUMMARY")
            print("-" * 40)
            print(f"   Query Type: {response.query_type}")
            print(f"   Iterations: {response.iterations}")
            print(f"   Time: {response.total_time:.2f}s")
            print(f"   Sources: {len(response.sources)}")
            print("="*60 + "\n")
        
        return response
    
    def _refine_answer(
        self,
        query: str,
        answer: str,
        context: str,
        sources: List[str],
        evaluation: Evaluation
    ) -> str:
        """Refine an answer based on evaluation feedback"""
        
        # Get suggestions from evaluation
        suggestions = evaluation.suggestions
        
        # Use answer generator with additional instructions
        try:
            refinement_query = f"""
Original question: {query}

Previous answer that needs improvement:
{answer}

Issues identified:
{chr(10).join(f'- {s}' for s in suggestions)}

Please provide an improved answer that addresses these issues.
"""
            
            result = self.answer_generator_tool.run(
                query=refinement_query,
                context=context,
                sources=sources,
                format_type="prose"
            )
            
            return result.get("answer", answer)
            
        except Exception as e:
            print(f"⚠️ Refinement failed: {e}")
            return answer
    
    def quick_answer(self, query: str) -> str:
        """
        Get a quick answer without full evaluation loop.
        
        Useful for simple queries where speed is prioritized.
        """
        # Simple plan + execute
        plan = self.planner.plan(query)
        result = self.executor.execute(plan)
        
        if result.success:
            return result.answer
        return "Sorry, I couldn't answer that question."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "registered_tools": list(self.executor.tools.keys()),
            "max_iterations": self.max_iterations,
            "sql_stats": self.sql_tool.get_stats_summary() if self.sql_tool else {}
        }


# ============================================================
# SIMPLE INTERFACE FOR BACKWARD COMPATIBILITY
# ============================================================

_agentic_instance = None

def get_agentic_rag(query_helper=None, llm_reasoner=None) -> AgenticRAG:
    """Get or create a singleton AgenticRAG instance"""
    global _agentic_instance
    
    if _agentic_instance is None:
        _agentic_instance = AgenticRAG(
            query_helper=query_helper,
            llm_reasoner=llm_reasoner
        )
    
    return _agentic_instance


def agentic_answer(query: str, verbose: bool = False) -> str:
    """
    Simple function to get an agentic answer.
    
    Usage:
        from agentic import agentic_answer
        answer = agentic_answer("What is Amazon's CTC?")
    """
    agent = get_agentic_rag()
    response = agent.answer_query(query, verbose=verbose)
    return response.answer


# ============================================================
# MAIN - TEST THE SYSTEM
# ============================================================

if __name__ == "__main__":
    # Test the agentic system
    agent = AgenticRAG(max_iterations=2, use_llm_classification=False)
    
    test_queries = [
        "What is Amazon's CTC?",
        "Compare Google and Microsoft",
        "What is the average CTC?",
        "Am I eligible for Intel with 8.5 CGPA?",
    ]
    
    for query in test_queries:
        print("\n" + "="*70)
        response = agent.answer_query(query, verbose=True)
        print(f"\n📌 FINAL ANSWER:\n{response.answer}")
        print("="*70)
