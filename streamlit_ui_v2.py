"""
🎓 MSIS Placement Assistant - Streamlit UI
==========================================
Unified UI with both Basic RAG and Agentic RAG modes.

Features:
- Toggle between Basic RAG and Agentic RAG modes
- Chat interface with history
- Detailed metadata and debugging info
- Query type classification (Agentic mode)
- Feedback collection
"""

import streamlit as st
import time
import uuid
from typing import Dict, Any

# Import input validator
from input_validator import validate_input, RateLimiter

# Import query analyzer for auto mode
try:
    from agentic.query_analyzer import QueryAnalyzer, auto_select_mode
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# Import conversation memory components
try:
    from agentic.memory_resolver import MemoryResolver
    from agentic.entity_extractor import EntityExtractor
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Import feedback collector
try:
    from feedback_collector import FeedbackCollector
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="MSIS Placement Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# SYSTEM INITIALIZATION
# ============================================================

@st.cache_resource
def load_basic_system():
    """Load the basic RAG system (QueryHelper + LLMReasoner)"""
    try:
        from query_helper import QueryHelper
        from llm_reasoner import LLMReasoner
        helper = QueryHelper(use_llm=True)
        reasoner = LLMReasoner()
        return helper, reasoner
    except ImportError as e:
        st.error(f"❌ Import Error: {e}. Please ensure all required modules are installed.")
        return None, None
    except Exception as e:
        st.error(f"❌ Failed to load basic system: {e}")
        return None, None
        return helper, reasoner, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_agentic_system():
    """Load the Agentic RAG system"""
    try:
        from agentic import AgenticRAG
        agent = AgenticRAG(
            max_iterations=3,
            use_llm_classification=True
        )
        return agent, None
    except Exception as e:
        return None, str(e)


# ============================================================
# BASIC RAG HANDLER
# ============================================================

def handle_basic_rag(
    prompt: str,
    helper,
    reasoner,
    top_k: int,
    show_draft: bool,
    show_context: bool
) -> Dict[str, Any]:
    """Handle query using basic RAG pipeline"""
    
    with st.spinner("🔍 Searching placement records..."):
        start_time = time.time()
        
        # Get draft answer
        draft_result = helper.generate_answer(prompt, top_k=top_k)
        
        # Show draft if enabled
        if show_draft:
            with st.expander("📝 Draft Answer"):
                st.markdown(draft_result.get('answer', 'No draft generated'))
        
        # Show context if enabled
        if show_context:
            with st.expander("📄 Retrieved Context"):
                context = draft_result.get('context', '')
                st.text(context[:2000] + ("..." if len(context) > 2000 else ""))
    
    with st.spinner("✨ Refining answer..."):
        # Refine answer
        refined = reasoner.refine_answer(
            query=prompt,
            draft_answer=draft_result.get('answer', ''),
            context=draft_result.get('context', ''),
            sources=draft_result.get('sources', [])
        )
        
        elapsed = time.time() - start_time
    
    return {
        "answer": refined.get('refined_answer', ''),
        "format_type": refined.get('format_type', 'prose'),
        "sources": refined.get('sources', []),
        "validation": refined.get('validation', {}),
        "elapsed_time": elapsed,
        "mode": "basic"
    }


# ============================================================
# AGENTIC RAG HANDLER
# ============================================================

def handle_agentic_rag(
    prompt: str,
    agent,
    show_plan: bool,
    show_evaluation: bool
) -> Dict[str, Any]:
    """Handle query using Agentic RAG pipeline"""
    
    with st.spinner("🤖 Agentic processing..."):
        start_time = time.time()
        
        # Process with agentic system
        response = agent.answer_query(prompt, verbose=False)
        
        elapsed = time.time() - start_time
    
    # Show plan if enabled
    if show_plan and response.plan:
        with st.expander("📋 Execution Plan"):
            st.write(f"**Query Type:** {response.query_type}")
            st.write(f"**Steps:** {len(response.plan.steps)}")
            for i, step in enumerate(response.plan.steps):
                st.write(f"  {i+1}. `{step.tool}` - {step.description}")
    
    # Show evaluation if enabled
    if show_evaluation and response.evaluation:
        with st.expander("🔍 Quality Evaluation"):
            eval_data = response.evaluation.to_dict()
            st.write(f"**Decision:** {eval_data['decision'].upper()}")
            st.write(f"**Confidence:** {eval_data['confidence']:.0%}")
            st.write("**Checks:**")
            for name, check in eval_data['checks'].items():
                status = "✅" if check['passed'] else "❌"
                st.write(f"  {status} {name}: {check['score']:.0%}")
    
    return {
        "answer": response.answer,
        "query_type": response.query_type,
        "sources": response.sources,
        "iterations": response.iterations,
        "elapsed_time": elapsed,
        "mode": "agentic",
        "success": response.success
    }


# ============================================================
# FEEDBACK COLLECTION
# ============================================================

def collect_feedback(message_id: str, query: str, answer: str, metadata: Dict):
    """Display feedback collection UI"""
    
    if not FEEDBACK_AVAILABLE:
        return
    
    feedback_key = f"feedback_{message_id}"
    
    # Check if feedback already submitted
    if feedback_key in st.session_state:
        st.success("✅ Thank you for your feedback!")
        return
    
    with st.expander("💬 Was this helpful?", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👍 Helpful", key=f"helpful_yes_{message_id}", use_container_width=True):
                collector = FeedbackCollector()
                collector.save_feedback(query, answer, helpful=True, metadata=metadata)
                st.session_state[feedback_key] = True
                st.success("Thanks for your feedback!")
                st.rerun()
        
        with col2:
            if st.button("👎 Not helpful", key=f"helpful_no_{message_id}", use_container_width=True):
                st.session_state[f"show_detailed_{message_id}"] = True
        
        # Show detailed feedback form if thumbs down clicked
        if st.session_state.get(f"show_detailed_{message_id}", False):
            st.markdown("---")
            st.write("**Help us improve:**")
            
            # Star rating
            rating = st.slider(
                "Rate this answer (1-5 stars)", 
                1, 5, 3, 
                key=f"rating_{message_id}"
            )
            
            # Issues checkboxes
            st.write("**What went wrong?**")
            issues = []
            if st.checkbox("❌ Incomplete information", key=f"incomplete_{message_id}"):
                issues.append("incomplete")
            if st.checkbox("❌ Wrong information", key=f"wrong_{message_id}"):
                issues.append("wrong_info")
            if st.checkbox("❌ Unclear explanation", key=f"unclear_{message_id}"):
                issues.append("unclear")
            if st.checkbox("❌ Too slow", key=f"slow_{message_id}"):
                issues.append("slow")
            
            # Suggestion text
            suggestion = st.text_area(
                "How can we improve? (optional)", 
                key=f"suggestion_{message_id}",
                placeholder="Tell us what's missing or what you expected..."
            )
            
            if st.button("Submit Feedback", key=f"submit_{message_id}", use_container_width=True):
                collector = FeedbackCollector()
                collector.save_feedback(
                    query=query,
                    answer=answer,
                    helpful=False,
                    rating=rating,
                    issues=issues,
                    suggestion=suggestion,
                    metadata=metadata
                )
                st.session_state[feedback_key] = True
                st.success("✅ Feedback submitted! Thank you!")
                st.rerun()


# ============================================================
# MAIN UI
# ============================================================

def main():
    # Title
    st.title("🎓 MSIS Placement Assistant")
    st.markdown("Ask questions about MSIS placement records (2025-2026)")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection - now with Auto option!
        mode_options = ["🔄 Auto", "📚 Basic RAG", "🤖 Agentic RAG"]
        mode = st.radio(
            "🤖 RAG Mode",
            mode_options,
            help="Auto: System decides based on query complexity\nBasic: Traditional RAG pipeline\nAgentic: Multi-step reasoning with planning"
        )
        
        # Show auto-mode info
        if mode == "🔄 Auto":
            st.info("🧠 System will analyze your query and choose the best mode automatically")
        
        st.markdown("---")
        
        # Mode-specific settings
        if mode == "📚 Basic RAG":
            top_k = st.slider("Documents to retrieve", 3, 15, 5)
            show_draft = st.checkbox("Show draft answer", value=False)
            show_context = st.checkbox("Show retrieved context", value=False)
            show_plan = False
            show_evaluation = False
        elif mode == "🤖 Agentic RAG":
            show_plan = st.checkbox("Show execution plan", value=True)
            show_evaluation = st.checkbox("Show quality evaluation", value=True)
            top_k = 5
            show_draft = False
            show_context = False
        else:  # Auto mode
            show_plan = st.checkbox("Show execution plan (if Agentic)", value=True)
            show_evaluation = st.checkbox("Show quality evaluation (if Agentic)", value=True)
            top_k = 5
            show_draft = False
            show_context = False
        
        st.markdown("---")
        
        # Stats
        st.header("📊 Session Stats")
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        if 'agentic_count' not in st.session_state:
            st.session_state.agentic_count = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", st.session_state.query_count)
        with col2:
            st.metric("Agentic", st.session_state.agentic_count)
        
        st.markdown("---")
        
        # Example queries
        st.header("💡 Example Queries")
        
        if mode == "📚 Basic RAG":
            examples = [
                "Which companies visited MSIS?",
                "What is Amazon eligibility?",
                "Tell me about Intel selection process"
            ]
        else:  # Agentic or Auto
            examples = [
                "Compare Amazon vs Google vs Microsoft",
                "What is the average CTC across all companies?",
                "Am I eligible for Intel with 8.5 CGPA in CSE?",
                "Show companies with CTC above 20 LPA",
                "What is Amazon's CTC?"  # Simple query for Auto mode demo
            ]
        
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.current_query = ex
        
        st.markdown("---")
        
        # Conversation Context Display
        if MEMORY_AVAILABLE and 'conversation_memory' in st.session_state:
            memory = st.session_state.conversation_memory
            
            # Only show if there's context
            if memory["context"]["last_company"] or memory["entities"]["companies"]:
                st.header("🧠 Conversation Context")
                
                if memory["context"]["last_company"]:
                    st.write(f"**Current topic:** {memory['context']['last_company']}")
                
                if memory["entities"]["companies"]:
                    st.write(f"**Companies discussed:** {', '.join(memory['entities']['companies'][:5])}")
                
                if memory["entities"]["cgpa"]:
                    st.write(f"**Your CGPA:** {memory['entities']['cgpa']}")
                
                if memory["entities"]["branch"]:
                    st.write(f"**Your Branch:** {memory['entities']['branch']}")
                
                st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.agentic_count = 0
            if 'conversation_memory' in st.session_state:
                st.session_state.conversation_memory = {
                    "entities": {
                        "companies": [],
                        "cgpa": None,
                        "branch": None,
                        "skills": []
                    },
                    "context": {
                        "last_company": None,
                        "last_topic": None,
                        "query_count": 0
                    }
                }
            st.rerun()
    
    # ========== LOAD SYSTEMS ==========
    # Load based on mode
    helper, reasoner, basic_error = None, None, None
    agent, agentic_error = None, None
    query_analyzer = None
    
    if mode == "📚 Basic RAG":
        helper, reasoner, basic_error = load_basic_system()
        if basic_error:
            st.error(f"❌ Failed to load Basic RAG: {basic_error}")
            return
    elif mode == "🤖 Agentic RAG":
        agent, agentic_error = load_agentic_system()
        if agentic_error:
            st.error(f"❌ Failed to load Agentic RAG: {agentic_error}")
            st.info("Falling back to Basic RAG mode")
            helper, reasoner, basic_error = load_basic_system()
            if basic_error:
                st.error(f"❌ Failed to load Basic RAG: {basic_error}")
                return
            mode = "📚 Basic RAG"
    else:  # Auto mode - load both systems
        helper, reasoner, basic_error = load_basic_system()
        agent, agentic_error = load_agentic_system()
        
        if basic_error and agentic_error:
            st.error("❌ Failed to load any RAG system")
            return
        
        # Initialize query analyzer
        if ANALYZER_AVAILABLE:
            known_companies = []
            if agent:
                try:
                    known_companies = agent.vector_search_tool.get_all_companies()
                except:
                    pass
            query_analyzer = QueryAnalyzer(known_companies)
    
    # ========== CHAT INTERFACE ==========
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize conversation memory
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = {
            "entities": {
                "companies": [],
                "cgpa": None,
                "branch": None,
                "skills": []
            },
            "context": {
                "last_company": None,
                "last_topic": None,
                "query_count": 0
            }
        }
    
    # Initialize memory components
    if MEMORY_AVAILABLE:
        memory_resolver = MemoryResolver()
        # Get known companies for entity extraction
        known_companies = []
        if agent:
            try:
                known_companies = agent.vector_search_tool.get_all_companies()
            except:
                pass
        entity_extractor = EntityExtractor(known_companies)
    else:
        memory_resolver = None
        entity_extractor = None
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata if available
            if "metadata" in message and message["metadata"]:
                with st.expander("📋 Details"):
                    st.json(message["metadata"])
    
    # Handle example query from sidebar
    if 'current_query' in st.session_state:
        prompt = st.session_state.current_query
        del st.session_state.current_query
    else:
        prompt = st.chat_input("Ask about MSIS placements...")
    
    # Process query
    if prompt:
        # ============================================================
        # INPUT VALIDATION
        # ============================================================
        
        # Validate and sanitize input
        validation_result = validate_input(prompt, min_length=10, max_length=1500)
        
        if not validation_result.is_valid:
            # Show all errors
            for error in validation_result.errors:
                st.error(error)
            st.stop()
        
        # Show warnings if any
        for warning in validation_result.warnings:
            st.warning(warning)
        
        # Use cleaned query
        prompt = validation_result.cleaned_query
        
        # Show validation details in debug mode
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Input Validation Details"):
                st.json(validation_result.to_dict())
        
        # ============================================================
        # RATE LIMITING
        # ============================================================
        
        # Initialize rate limiter if not exists
        if 'rate_limiter' not in st.session_state:
            st.session_state.rate_limiter = RateLimiter(max_requests=10, time_window=60)
        
        # Check rate limit
        session_id = st.session_state.get('session_id', str(uuid.uuid4()))
        st.session_state.session_id = session_id
        
        is_allowed, rate_message = st.session_state.rate_limiter.is_allowed(session_id)
        if not is_allowed:
            st.error(rate_message)
            remaining = st.session_state.rate_limiter.get_remaining_requests(session_id)
            st.info(f"Remaining requests: {remaining}")
            st.stop()
        
        # ============================================================
        # CONTINUE WITH NORMAL PROCESSING
        # ============================================================
        
        # Resolve query using conversation context
        original_prompt = prompt
        query_modified = False
        
        if memory_resolver and memory_resolver.should_use_memory(prompt):
            prompt, query_modified = memory_resolver.resolve_query(
                prompt, 
                st.session_state.conversation_memory
            )
            
            # Show context resolution if query was modified
            if query_modified:
                st.info(f"💭 **Context-aware query:** {prompt}")
        
        # Add user message (original)
        st.session_state.messages.append({"role": "user", "content": original_prompt})
        with st.chat_message("user"):
            st.markdown(original_prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Determine which mode to use
                use_agentic = False
                auto_reason = ""
                
                if mode == "🤖 Agentic RAG":
                    use_agentic = True
                elif mode == "🔄 Auto":
                    # Auto-select based on query complexity
                    if query_analyzer:
                        recommendation = query_analyzer.get_recommendation(prompt)
                        use_agentic = recommendation["recommended_mode"] == "agentic"
                        auto_reason = recommendation["reason"]
                        
                        # Show auto-selection result
                        with st.expander("🧠 Auto Mode Selection", expanded=False):
                            selected = "🤖 Agentic RAG" if use_agentic else "📚 Basic RAG"
                            st.write(f"**Selected:** {selected}")
                            st.write(f"**Complexity:** {recommendation['complexity'].upper()}")
                            st.write(f"**Confidence:** {recommendation['confidence']:.0%}")
                            st.write(f"**Reason:** {auto_reason}")
                    else:
                        # Fallback: simple pattern check
                        complex_keywords = ["compare", "average", "eligible", "top ", "statistics"]
                        use_agentic = any(kw in prompt.lower() for kw in complex_keywords)
                
                # Execute with chosen mode
                if use_agentic and agent:
                    result = handle_agentic_rag(
                        prompt, agent, show_plan, show_evaluation
                    )
                    st.session_state.agentic_count += 1
                    if mode == "🔄 Auto":
                        result["auto_selected"] = True
                        result["auto_reason"] = auto_reason
                else:
                    result = handle_basic_rag(
                        prompt, helper, reasoner, top_k, show_draft, show_context
                    )
                    if mode == "🔄 Auto":
                        result["auto_selected"] = True
                        result["auto_reason"] = auto_reason
                
                # Display answer
                st.markdown(result["answer"])
                
                # Show metadata bar
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    mode_icon = "🤖" if result["mode"] == "agentic" else "📚"
                    st.caption(f"{mode_icon} {result['mode'].title()}")
                
                with col2:
                    if result["mode"] == "agentic":
                        st.caption(f"🎯 Type: {result.get('query_type', 'N/A')}")
                    else:
                        st.caption(f"📊 Format: {result.get('format_type', 'N/A')}")
                
                with col3:
                    sources = result.get('sources', [])
                    st.caption(f"📚 Sources: {len(sources)}")
                
                with col4:
                    st.caption(f"⏱️ {result['elapsed_time']:.2f}s")
                
                # Build metadata for history
                metadata = {
                    "mode": result["mode"],
                    "elapsed_time": f"{result['elapsed_time']:.2f}s",
                    "sources": result.get("sources", [])[:5]
                }
                
                if result["mode"] == "agentic":
                    metadata["query_type"] = result.get("query_type")
                    metadata["iterations"] = result.get("iterations")
                else:
                    metadata["format_type"] = result.get("format_type")
                    metadata["validation"] = result.get("validation")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": metadata
                })
                
                st.session_state.query_count += 1
                
                # Update conversation memory
                if entity_extractor:
                    # Extract entities from this Q&A
                    new_entities = entity_extractor.extract_from_conversation(
                        original_prompt,
                        result["answer"]
                    )
                    
                    # Merge with existing entities
                    st.session_state.conversation_memory["entities"] = entity_extractor.merge_entities(
                        st.session_state.conversation_memory["entities"],
                        new_entities
                    )
                    
                    # Update context
                    if new_entities.get("companies"):
                        st.session_state.conversation_memory["context"]["last_company"] = new_entities["companies"][-1]
                    
                    if new_entities.get("topic"):
                        st.session_state.conversation_memory["context"]["last_topic"] = new_entities["topic"]
                    
                    st.session_state.conversation_memory["context"]["query_count"] += 1
                
                # Collect feedback
                message_id = str(uuid.uuid4())[:8]
                collect_feedback(message_id, original_prompt, result["answer"], metadata)
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {e}",
                    "metadata": {"error": str(e)}
                })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
