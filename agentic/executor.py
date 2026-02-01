"""
⚙️ EXECUTOR MODULE
==================
Responsible for:
1. Tool Registry - Managing available tools
2. Plan Execution - Running steps in order
3. State Management - Storing intermediate results
4. Dependency Resolution - Resolving step dependencies
5. Error Handling - Graceful failure management

The Executor takes a Plan from the Planner and orchestrates
the execution of each step, managing data flow between tools.
"""

import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class StepResult:
    """Result of executing a single step"""
    step_index: int
    tool: str
    success: bool
    output: Any = None
    error: str = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "step_index": self.step_index,
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time
        }


@dataclass
class ExecutionResult:
    """Result of executing an entire plan"""
    success: bool
    results: List[StepResult] = field(default_factory=list)
    final_output: Any = None
    error: str = None
    total_time: float = 0.0
    
    # Extracted data for convenience
    answer: str = ""
    context: str = ""
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "results": [r.to_dict() for r in self.results],
            "final_output": self.final_output,
            "error": self.error,
            "total_time": self.total_time,
            "answer": self.answer,
            "context": self.context,
            "sources": self.sources
        }


class Executor:
    """
    Executes plans by orchestrating tools.
    
    The Executor manages:
    - Tool registration and lookup
    - Step execution with dependency resolution
    - State management for passing data between steps
    - Parallel execution where possible
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the Executor.
        
        Args:
            max_workers: Maximum parallel workers for tool execution
        """
        self.tools: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.max_workers = max_workers
        
        print("✅ Executor initialized")
    
    def register_tool(self, name: str, tool: Any):
        """
        Register a tool with the executor.
        
        Args:
            name: Tool name (used in Plan steps)
            tool: Tool instance with a `run()` method
        """
        self.tools[name] = tool
        print(f"🔧 Registered tool: {name}")
    
    def register_tools(self, tools: Dict[str, Any]):
        """Register multiple tools at once"""
        for name, tool in tools.items():
            self.register_tool(name, tool)
    
    def execute(self, plan: 'Plan') -> ExecutionResult:
        """
        Execute a plan step by step.
        
        This is the main entry point for the Executor.
        
        Args:
            plan: Plan object from the Planner
            
        Returns:
            ExecutionResult with all step results and final output
        """
        import time
        start_time = time.time()
        
        # Input validation
        if not plan:
            return ExecutionResult(
                success=False,
                final_output="No plan provided",
                error="Plan is None or empty"
            )
        
        if not hasattr(plan, 'steps') or not plan.steps:
            return ExecutionResult(
                success=False,
                final_output="Plan has no steps",
                error="Invalid plan structure"
            )
        
        try:
            # Clear state for new execution
            self.state = {}
            results: List[StepResult] = []
            
            print(f"\\n⚙️ Executing plan with {len(plan.steps)} steps...")
        print(f"   Mode: {plan.execution_mode}")
        
        try:
            if plan.execution_mode == "parallel":
                results = self._execute_parallel(plan)
            else:
                results = self._execute_sequential(plan)
            
            # Check if all steps succeeded
            all_success = all(r.success for r in results)
            
            if not all_success:
                failed_steps = [r for r in results if not r.success]
                error_msg = f"Steps failed: {[r.step_index for r in failed_steps]}"
                return ExecutionResult(
                    success=False,
                    results=results,
                    error=error_msg,
                    total_time=time.time() - start_time
                )
            
            # Extract final output and answer
            final_output = results[-1].output if results else None
            
            # Extract answer, context, sources from final output
            answer = ""
            context = ""
            sources = []
            
            if isinstance(final_output, dict):
                answer = final_output.get("answer", final_output.get("response", ""))
                context = final_output.get("context", "")
                sources = final_output.get("sources", [])
            elif isinstance(final_output, str):
                answer = final_output
            
            # Also gather context from search steps
            for result in results:
                if result.tool == "vector_search" and result.output:
                    if not context and isinstance(result.output, dict):
                        context = result.output.get("context", "")
                        sources.extend(result.output.get("sources", []))
            
            return ExecutionResult(
                success=True,
                results=results,
                final_output=final_output,
                total_time=time.time() - start_time,
                answer=answer,
                context=context,
                sources=list(set(sources))  # Deduplicate
            )
            
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            return ExecutionResult(
                success=False,
                results=results,
                error=str(e),
                total_time=time.time() - start_time
            )
    
    def _execute_sequential(self, plan: 'Plan') -> List[StepResult]:
        """Execute steps one by one in order"""
        results = []
        
        for i, step in enumerate(plan.steps):
            print(f"\n📍 Step {i+1}/{len(plan.steps)}: {step.tool}")
            print(f"   {step.description}")
            
            result = self._execute_step(i, step)
            results.append(result)
            
            if not result.success:
                print(f"❌ Step {i+1} failed: {result.error}")
                break
            else:
                print(f"✅ Step {i+1} completed ({result.execution_time:.2f}s)")
        
        return results
    
    def _execute_parallel(self, plan: 'Plan') -> List[StepResult]:
        """
        Execute independent steps in parallel.
        
        Steps with dependencies run after their dependencies complete.
        """
        results: List[StepResult] = [None] * len(plan.steps)
        completed = set()
        
        while len(completed) < len(plan.steps):
            # Find steps that can run (dependencies satisfied)
            ready_steps = []
            for i, step in enumerate(plan.steps):
                if i in completed:
                    continue
                
                # Check if all dependencies are completed
                deps_satisfied = all(d in completed for d in step.depends_on)
                if deps_satisfied:
                    ready_steps.append((i, step))
            
            if not ready_steps:
                # No steps ready but not all completed - circular dependency
                remaining = set(range(len(plan.steps))) - completed
                raise RuntimeError(f"Circular dependency detected for steps: {remaining}")
            
            # Execute ready steps in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._execute_step, i, step): i
                    for i, step in ready_steps
                }
                
                for future in as_completed(futures):
                    step_idx = futures[future]
                    try:
                        result = future.result()
                        results[step_idx] = result
                        completed.add(step_idx)
                        
                        if result.success:
                            print(f"✅ Step {step_idx+1} completed ({result.execution_time:.2f}s)")
                        else:
                            print(f"❌ Step {step_idx+1} failed: {result.error}")
                            # Stop execution on failure
                            return [r for r in results if r is not None]
                    except Exception as e:
                        results[step_idx] = StepResult(
                            step_index=step_idx,
                            tool=plan.steps[step_idx].tool,
                            success=False,
                            error=str(e)
                        )
                        return [r for r in results if r is not None]
        
        return results
    
    def _execute_step(self, index: int, step: 'Step') -> StepResult:
        """
        Execute a single step.
        
        Args:
            index: Step index in the plan
            step: Step object to execute
            
        Returns:
            StepResult with execution outcome
        """
        import time
        start_time = time.time()
        
        # Get the tool
        tool = self.tools.get(step.tool)
        if not tool:
            return StepResult(
                step_index=index,
                tool=step.tool,
                success=False,
                error=f"Tool not found: {step.tool}"
            )
        
        try:
            # Resolve arguments (replace placeholders with actual values)
            resolved_args = self._resolve_args(step.args)
            
            # Execute the tool
            output = tool.run(**resolved_args)
            
            # Store result in state for future steps
            self.state[f"step_{index}"] = output
            
            return StepResult(
                step_index=index,
                tool=step.tool,
                success=True,
                output=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return StepResult(
                step_index=index,
                tool=step.tool,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _resolve_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve argument placeholders.
        
        Placeholders like {step_0.context} are replaced with actual
        values from the state.
        
        Args:
            args: Dictionary of arguments with potential placeholders
            
        Returns:
            Dictionary with resolved values
        """
        resolved = {}
        
        for key, value in args.items():
            resolved[key] = self._resolve_value(value)
        
        return resolved
    
    def _resolve_value(self, value: Any) -> Any:
        """Recursively resolve a value"""
        
        if isinstance(value, str):
            # Check for placeholder pattern: {step_X.field} or {step_X.field[index]}
            placeholder_pattern = r'\{step_(\d+)\.([^}]+)\}'
            
            match = re.search(placeholder_pattern, value)
            if match:
                step_idx = int(match.group(1))
                field_path = match.group(2)
                
                # Get step result from state
                step_key = f"step_{step_idx}"
                if step_key not in self.state:
                    return value  # Keep placeholder if step not executed yet
                
                step_data = self.state[step_key]
                
                # Navigate the field path
                resolved_value = self._get_nested(step_data, field_path)
                
                # If the placeholder is the entire string, return the value directly
                if value == f"{{step_{step_idx}.{field_path}}}":
                    return resolved_value
                
                # Otherwise, replace the placeholder in the string
                return value.replace(match.group(0), str(resolved_value))
            
            return value
        
        elif isinstance(value, list):
            return [self._resolve_value(v) for v in value]
        
        elif isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        
        return value
    
    def _get_nested(self, data: Any, path: str) -> Any:
        """
        Get a nested value from data using dot/bracket notation.
        
        Examples:
            path="context" -> data["context"]
            path="companies[0]" -> data["companies"][0]
            path="result.value" -> data["result"]["value"]
        """
        if data is None:
            return None
        
        # Handle array indexing: field[0]
        array_pattern = r'(\w+)\[(\d+)\]'
        match = re.match(array_pattern, path)
        if match:
            field = match.group(1)
            index = int(match.group(2))
            if isinstance(data, dict) and field in data:
                arr = data[field]
                if isinstance(arr, list) and index < len(arr):
                    return arr[index]
            return None
        
        # Handle dot notation: field.subfield
        if '.' in path:
            parts = path.split('.', 1)
            if isinstance(data, dict) and parts[0] in data:
                return self._get_nested(data[parts[0]], parts[1])
            return None
        
        # Simple field access
        if isinstance(data, dict):
            return data.get(path)
        
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current execution state (for debugging)"""
        return self.state.copy()
    
    def clear_state(self):
        """Clear execution state"""
        self.state = {}


# Example usage
if __name__ == "__main__":
    # Create executor
    executor = Executor()
    
    # Create a mock tool for testing
    class MockTool:
        def run(self, **kwargs):
            return {"result": f"Processed: {kwargs}"}
    
    # Register mock tools
    executor.register_tool("vector_search", MockTool())
    executor.register_tool("answer_generator", MockTool())
    
    # Test state resolution
    executor.state["step_0"] = {
        "context": "Test context from search",
        "sources": ["source1.pdf", "source2.pdf"],
        "companies": ["Amazon", "Google"]
    }
    
    # Test argument resolution
    args = {
        "context": "{step_0.context}",
        "sources": "{step_0.sources}",
        "first_company": "{step_0.companies[0]}"
    }
    
    resolved = executor._resolve_args(args)
    print("Resolved args:", json.dumps(resolved, indent=2))
