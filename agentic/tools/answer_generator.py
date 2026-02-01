"""
💬 ANSWER GENERATOR TOOL
========================
Generates natural language answers from context.

This tool:
1. Takes retrieved context and sources
2. Formats answer based on query type
3. Adds citations and structure
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .base_tool import BaseTool


class AnswerGeneratorTool(BaseTool):
    """
    Generates formatted answers from context.
    
    Supports multiple output formats:
    - prose: Natural language paragraphs
    - bullet_points: Structured bullet list
    - table: Markdown table format
    """
    
    def __init__(self, llm_reasoner=None):
        """
        Initialize the answer generator.
        
        Args:
            llm_reasoner: Optional existing LLMReasoner instance
        """
        load_dotenv(dotenv_path="environment.env")
        
        self._llm_reasoner = llm_reasoner
        self.llm_client = None
        
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
                print("✅ AnswerGeneratorTool initialized")
            except Exception as e:
                print(f"⚠️ AnswerGeneratorTool: LLM init failed: {e}")
    
    @property
    def name(self) -> str:
        return "answer_generator"
    
    @property
    def description(self) -> str:
        return "Generate formatted answers from retrieved context"
    
    def run(
        self,
        query: str,
        context: str = "",
        sources: List[str] = None,
        format_type: str = "prose",
        structured_data: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer.
        
        Args:
            query: User's question
            context: Retrieved document context
            sources: List of source files
            format_type: "prose", "bullet_points", or "table"
            structured_data: Optional data from SQL tool
            
        Returns:
            Dict with:
                - answer: Generated answer
                - format: Format used
                - sources: Cited sources
        """
        print(f"💬 Generating answer ({format_type} format)")
        
        sources = sources or []
        
        # Use existing LLMReasoner if available
        if self._llm_reasoner:
            try:
                result = self._llm_reasoner.refine_answer(
                    query=query,
                    draft_answer="",
                    context=context,
                    sources=sources,
                    format_type=format_type
                )
                return {
                    "answer": result.get("refined_answer", ""),
                    "response": result.get("refined_answer", ""),
                    "format": format_type,
                    "sources": sources
                }
            except Exception as e:
                print(f"⚠️ LLMReasoner failed: {e}")
        
        # Generate with LLM
        if self.llm_client:
            return self._generate_llm(query, context, sources, format_type, structured_data)
        
        # Fallback
        return self._generate_fallback(query, context, sources, format_type)
    
    def _generate_llm(
        self,
        query: str,
        context: str,
        sources: List[str],
        format_type: str,
        structured_data: Any
    ) -> Dict[str, Any]:
        """Generate answer using LLM"""
        
        # Route to specialized formatters
        if format_type == "checklist":
            return self._format_checklist(query, context, sources, structured_data)
        elif format_type == "timeline":
            return self._format_timeline(query, context, sources, structured_data)
        elif format_type == "eligibility_card":
            return self._format_eligibility_card(query, context, sources, structured_data)
        elif format_type == "pros_cons":
            return self._format_pros_cons(query, context, sources, structured_data)
        elif format_type == "advice":
            return self._format_advice(query, context, sources, structured_data)
        
        # Build format instructions for standard types
        if format_type == "table":
            format_instructions = """Format your answer as a markdown table where appropriate.
Use | Column1 | Column2 | format with headers."""
        elif format_type == "bullet_points":
            format_instructions = """Format your answer as organized bullet points:
- Main point
  - Sub-point
- Another main point"""
        else:
            format_instructions = """Write a clear, well-structured prose response.
Use paragraphs to organize information."""
        
        # Include structured data if available
        data_section = ""
        if structured_data:
            data_section = f"\n\nStructured Data:\n{json.dumps(structured_data, indent=2)}"
        
        prompt = f"""Answer this placement-related question based on the provided context.

Question: {query}

Context from placement documents:
{context[:3000]}
{data_section}

Sources: {', '.join(sources[:5]) if sources else 'Not specified'}

{format_instructions}

Guidelines:
1. Answer ONLY based on the provided context
2. If information is not in context, say "Based on available information..."
3. Cite sources when mentioning specific facts: [Company Name]
4. Be concise but comprehensive
5. Include specific numbers (CTC, CGPA) when available

Provide your answer:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system", "content": "You are the MSIS Placement Bot, helping students with placement information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "response": answer,
                "format": format_type,
                "sources": sources
            }
            
        except Exception as e:
            print(f"⚠️ LLM answer generation failed: {e}")
            return self._generate_fallback(query, context, sources, format_type)
    
    def _format_checklist(self, query: str, context: str, sources: List[str], data: Any) -> Dict:
        """Format as interactive checklist"""
        prompt = f"""Create an action checklist for: {query}

Context: {context[:2000]}

Format as:
## [Title]

### Section 1
- [ ] Item 1 ✅ (if completed) or ⚠️ (if needs attention) or ❌ (if not done)
- [ ] Item 2
- [x] Item 3 (if applicable)

Use checkboxes and status emojis. Be specific and actionable."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.5
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "response": response.choices[0].message.content.strip(),
                "format": "checklist",
                "sources": sources
            }
        except Exception as e:
            return self._generate_fallback(query, context, sources, "checklist")
    
    def _format_timeline(self, query: str, context: str, sources: List[str], data: Any) -> Dict:
        """Format as timeline visualization"""
        prompt = f"""Create an ASCII timeline for: {query}

Context: {context[:2000]}

Format as:
[Company] Selection Process Timeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Date 1  ────●  Event 1 (Description)
             │
Date 2  ────●  Event 2
             │  ↓ Percentage/Note
Date 3  ────●  Event 3

Use │ for connections and ● for milestones. Include dates and descriptions."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.5
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "response": response.choices[0].message.content.strip(),
                "format": "timeline",
                "sources": sources
            }
        except Exception as e:
            return self._generate_fallback(query, context, sources, "timeline")
    
    def _format_eligibility_card(self, query: str, context: str, sources: List[str], data: Any) -> Dict:
        """Format as eligibility status card"""
        prompt = f"""Create an eligibility status card for: {query}

Context: {context[:2000]}

Format as:
╔═══════════════════════════════════════╗
║   [Company] Eligibility Check         ║
╠═══════════════════════════════════════╣
║  Your Profile:                        ║
║  • CGPA: [value]                      ║
║  • Branch: [branch]                   ║
║                                       ║
║  Requirements:                        ║
║  • CGPA: [req]     ✅/❌ PASS/FAIL   ║
║  • Branch: [req]   ✅/❌ PASS/FAIL   ║
║                                       ║
║  🎉 VERDICT: [ELIGIBLE/NOT ELIGIBLE] ║
║  Match Score: [X/100]                 ║
╚═══════════════════════════════════════╝

Use box drawing characters and status emojis."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "response": response.choices[0].message.content.strip(),
                "format": "eligibility_card",
                "sources": sources
            }
        except Exception as e:
            return self._generate_fallback(query, context, sources, "eligibility_card")
    
    def _format_pros_cons(self, query: str, context: str, sources: List[str], data: Any) -> Dict:
        """Format as pros/cons comparison"""
        prompt = f"""Create a pros/cons comparison for: {query}

Context: {context[:2000]}

Format as:
## [Title]

### [Option 1]
**Pros:**
✅ Benefit 1
✅ Benefit 2

**Cons:**
❌ Drawback 1
❌ Drawback 2

### [Option 2]
**Pros:**
✅ Benefit 1

**Cons:**
❌ Drawback 1

### 🎯 Recommendation
[Give clear recommendation with reasoning]

Use ✅ for pros and ❌ for cons."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.5
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "response": response.choices[0].message.content.strip(),
                "format": "pros_cons",
                "sources": sources
            }
        except Exception as e:
            return self._generate_fallback(query, context, sources, "pros_cons")
    
    def _format_advice(self, query: str, context: str, sources: List[str], data: Any) -> Dict:
        """Format as actionable advice"""
        prompt = f"""Provide actionable advice for: {query}

Context: {context[:2000]}
{f"Data: {json.dumps(data, indent=2)}" if data else ""}

Format as:
## [Title]

### Key Points
• Main point 1
• Main point 2

### Recommended Actions
1. Step 1 with rationale
2. Step 2 with rationale

### Pro Tips
💡 Tip 1
💡 Tip 2

Be specific, actionable, and encouraging."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.6
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "response": response.choices[0].message.content.strip(),
                "format": "advice",
                "sources": sources
            }
        except Exception as e:
            return self._generate_fallback(query, context, sources, "advice")
    
    def _generate_fallback(
        self,
        query: str,
        context: str,
        sources: List[str],
        format_type: str
    ) -> Dict[str, Any]:
        """Fallback answer generation without LLM"""
        
        if not context:
            answer = f"I couldn't find specific information about: {query}"
        else:
            # Extract relevant sentences from context
            sentences = context.split('.')
            relevant = []
            
            query_words = set(query.lower().split())
            for sentence in sentences[:10]:
                sentence_words = set(sentence.lower().split())
                if len(query_words & sentence_words) >= 2:
                    relevant.append(sentence.strip())
            
            if relevant:
                if format_type == "bullet_points":
                    answer = "Based on the available information:\n\n"
                    for sent in relevant[:5]:
                        answer += f"• {sent}.\n"
                else:
                    answer = " ".join(relevant[:5]) + "."
            else:
                answer = f"Here's what I found:\n\n{context[:500]}..."
        
        # Add sources
        if sources:
            answer += f"\n\nSources: {', '.join(sources[:3])}"
        
        return {
            "answer": answer,
            "response": answer,
            "format": format_type,
            "sources": sources
        }


# Example usage
if __name__ == "__main__":
    tool = AnswerGeneratorTool()
    
    result = tool.run(
        query="What is Amazon's CTC?",
        context="Amazon visited campus in 2025. The CTC offered was 28 LPA for SDE-1 roles. Eligibility: 7.5 CGPA minimum from CS/ECE branches.",
        sources=["Amazon_MTech_2026/placement.pdf"],
        format_type="prose"
    )
    
    print(result["answer"])
