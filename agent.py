import argparse
import sys
import asyncio
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph

import code_executor as executor
import github_search as gh
from llm_utils import GITHUB_TOKEN, MAX_RETRIES, TOP_K_RESULTS, EXECUTION_TIMEOUT, get_llm

# State definition
class AgentState(TypedDict):
    query: str
    snippets: list[dict]
    generated_code: str
    exec_stdout: str
    exec_stderr: str
    iteration: int
    max_iterations: int
    status: str
    top_k: int
    model: str


# Node 1: Search GitHub
def search_github(state: AgentState) -> AgentState:
    """Step 2: Retrieve relevant code snippets from GitHub."""
    print(f"\n{'-'*60}")
    print(f"STEP 2 Searching GitHub for: '{state['query']}'")
    print(f"{'-'*60}")
    
    snippets = gh.search_github_code(
        query=state["query"], 
        top_k=state["top_k"],
        token=GITHUB_TOKEN
    )
    
    print(f"Retrieved {len(snippets)} snippet(s) from GitHub.")
    for s in snippets:
        print(f"  {s['repo']} / {s['path']}")
        
    return {**state, "snippets": snippets}


# Node 2: Generate Code
# Node 2: Generate Code
def generate_code(state: AgentState) -> AgentState:
    """Steps 3 & 4: Analyze GitHub context and generate new code."""
    print(f"\n{'-'*60}")
    print("STEP 3 & 4 - Analyzing context and generating code...")
    print(f"{'-'*60}")
    
    llm = get_llm(model_name=state["model"], temperature=0)
    context = gh.format_snippets_for_prompt(state["snippets"])
    
    prompt = f"""You are an expert Python developer. A user has submitted the following request:

USER REQUEST:
{state['query']}

RELEVANT GITHUB EXAMPLES (for reference and inspiration):
{context}

INSTRUCTIONS:
1. Study the GitHub examples to understand patterns, best practices, and relevant APIs.
2. Write complete, working Python code that fulfills the user's request.
3. The code must be self-contained and runnable immediately (no missing imports, no placeholders).
4. Include a brief example or test at the bottom so execution produces visible output.

OUTPUT FORMAT:
You must return ONLY raw, valid Python script text.
CRITICAL RULES:
1. DO NOT wrap the code in markdown blocks (no ```python or ```).
2. DO NOT include a single word of explanation, greeting, or conclusion.
3. If you include any non-Python text, the system will crash.
"""
    response = llm.invoke(prompt)
    raw_code = response.content
    
    # Safely extract the code in case the LLM ignored the prompt and used markdown anyway
    clean_code = executor.extract_python_code(raw_code)
    
    print("Code generated successfully.") 
    print(f" Lines: {len(clean_code.splitlines())}")
    
    # Pass the clean code to Node 3 for safe execution
    return {**state, "generated_code": clean_code, "iteration": 0, "status": "running"}

# Node 3: Execute Code
def execute_code(state: AgentState) -> AgentState:
    """Step 5: Compile and run the generated code."""
    attempt_label = "initial run" if state["iteration"] == 0 else f"retry #{state['iteration']}"
    
    print(f"\n{'-'*60}") 
    print(f"STEP 5 Executing code ({attempt_label})...") 
    print(f"{'-'*60}")
    
    result = executor.execute_python(
        code=state["generated_code"], 
        timeout=EXECUTION_TIMEOUT
    )
    
    if result["success"]:
        print("Execution SUCCESS.")
        if result["stdout"]:
            print(f"Output:\n{result['stdout']}")
        return {**state, "exec_stdout": result["stdout"], "exec_stderr": "", "status": "success"}
    else:
        print("Execution FAILED.")
        print(f"Error: \n{result['stderr']}")
        return {
            **state,
            "exec_stdout": result["stdout"],
            "exec_stderr": result["stderr"],
            "status": "running",
        }


# Node 4: Fix Code
def fix_code(state: AgentState) -> AgentState:
    """Step 6: Use LLM to diagnose and fix execution errors."""
    new_iteration = state["iteration"] + 1
    
    print(f"\n{'-'*60}")
    print(f"STEP 6 - Fixing errors (attempt {new_iteration}/{state['max_iterations']})...")
    print(f"{'-'*60}")
    
    llm = get_llm(model_name=state["model"], temperature=0)
    prompt = f"""You are an expert Python debugger.

ORIGINAL USER REQUEST:
{state['query']}

CURRENT CODE (attempt {new_iteration}):
```python
{state['generated_code']}
"""
    response = llm.invoke(prompt)
    raw_code = response.content
    fixed = executor.extract_python_code("```python\n" + raw_code)
    
    print(f"Fixed code generated ({len(fixed.splitlines())} lines).")
    return {**state, "generated_code": fixed, "iteration": new_iteration}


# Routing logic
def route_after_execution(state: AgentState) -> str:
    """Decide next step after execute_code runs."""
    if state["status"] == "success":
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        print(f"\n[Agent] Reached max retries ({state['max_iterations']}). Stopping.")
        return "end"
    return "fix"


# Build the LangGraph
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("search_github", search_github)
    graph.add_node("generate_code", generate_code)
    graph.add_node("execute_code", execute_code)
    graph.add_node("fix_code", fix_code)
    
    graph.add_edge(START, "search_github") 
    graph.add_edge("search_github", "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_conditional_edges(
        "execute_code", 
        route_after_execution, 
        {"end": END, "fix": "fix_code"}
    )
    graph.add_edge("fix_code", "execute_code")  # Loop back after fix
    
    return graph.compile()


# Display final result
def print_final_result(state: AgentState):
    print(f"\n{'-'*60}")
    if state["status"] == "success": 
        print("STEP 7 - RESULT: Working code generated successfully!") 
        print(f"{'-'*60}") 
        print("\n- Final Code -")
        print("```python")
        print(state["generated_code"])
        print("```")
        
        if state["exec_stdout"]:
            print("\n- Execution Output -")
            print(state["exec_stdout"])
            
        iterations_used = state["iteration"]
        print(f"\nCompleted in {iterations_used} fix iteration(s).")
    else:
        print("STEP 7 - RESULT: Could not produce working code after max retries.")
        print(f"{'-'*60}")
        print("\n- Last Attempted Code -")
        print("```python")
        print(state["generated_code"])
        print("```")
        print(f"\n- Last Error -\n{state['exec_stderr']}")
        print(f"{'-'*60}")


# Main
async def main():
    parser = argparse.ArgumentParser(description="Autonomous Code Synthesis System")
    parser.add_argument("--query", type=str, default=None,
                        help="what code to generate (leave blank for interactive prompt)")
    parser.add_argument("--top_k", type=int, default=TOP_K_RESULTS,
                        help=f"GitHub results to retrieve (default: {TOP_K_RESULTS})") 
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES, 
                        help=f"Max error-fix attempts (default: {MAX_RETRIES})")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model name (default: gemini-2.5-flash)")
    
    args = parser.parse_args()
    
    # Step 1: Get user query
    query = args.query
    if not query:
        print("\n" + "-" * 60)
        print(" Autonomous Code Synthesis System")
        print(" GitHub-powered AI Code Generator with Auto-Fix")
        print("-" * 60)
        query = input("\nEnter your code request:\n> ").strip()
        
    if not query:
        print("No query provided. Exiting.")
        sys.exit(0)
        
    print(f"\n{'-'*60}")
    print(f"STEP 1 - Query received: '{query}'")
    print(f"{'-'*60}")
    print(f"Settings: top_k={args.top_k}, max_retries={args.max_retries}, model={args.model}")
    
    # Confirm before executing LLM-generated code
    print("\n[WARNING] This system will execute AI-generated Python code on your machine.")
    confirm = input("Proceed? (y/n): ").strip().lower()
    
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)
        
    # Build and run the graph
    agent = build_graph()
    
    initial_state: AgentState = {
        "query": query,
        "snippets": [],
        "generated_code": "",
        "exec_stdout": "",
        "exec_stderr": "",
        "iteration": 0,
        "max_iterations": args.max_retries,
        "status": "running",
        "top_k": args.top_k,
        "model": args.model,
    }
    
    final_state = await agent.ainvoke(initial_state)
    print_final_result(final_state)


if __name__ == "__main__":
    asyncio.run(main())