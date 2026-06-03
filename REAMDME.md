# Autonomous Code Synthesis System

A LangGraph-powered AI coding agent that autonomously searches GitHub for reference code, writes Python scripts to fulfill user requests, safely executes the generated code, and automatically debugs and fixes its own errors until the code runs successfully.

## 🚀 Features

* **GitHub Context Retrieval:** Searches GitHub for real-world Python examples using the GitHub REST API to understand best practices and APIs before writing code.
* **Autonomous Generation:** Uses Google's Gemini models (via LangChain) to generate complete, self-contained Python scripts.
* **Safe Execution:** Compiles and runs the generated code in an isolated subprocess with built-in timeouts to prevent system hangs.
* **Auto-Fix Loop:** If the code throws an error during execution, the system feeds the `stderr` traceback back to the LLM to diagnose and patch the bug, looping until the code succeeds or hits a maximum retry limit.


## 📁 Project Structure

* `agent.py`: The core LangGraph state machine containing the nodes (`search_github`, `generate_code`, `execute_code`, `fix_code`) and routing logic.
* `github_search.py`: Handles API requests to GitHub, rate limits, and base64 decoding of file contents.
* `code_executor.py`: Manages safe extraction of raw Python code from LLM responses and executes it in temporary files.
* `llm_utils.py`: Configures the LangChain Google Generative AI client and loads environment variables.

## 🛠️ Prerequisites

* Python 3.9+
* A [Google Gemini API Key](https://aistudio.google.com/app/apikey)
* A [GitHub Personal Access Token](https://github.com/settings/tokens) (Classic, no specific scopes required for public repos)

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Autonomous-code-synthesis

## 🧠 System Architecture

The following diagram illustrates the LangGraph state machine and the auto-fix routing logic:

```mermaid
flowchart TD
    START([Start: User Query]) --> N1[Node 1: search_github]
    
    N1 -->|Retrieves Top-K Snippets| N2[Node 2: generate_code]
    N2 -->|Generates Initial Code| N3[Node 3: execute_code]
    
    N3 -->|Executes in Subprocess| ROUTE{route_after_execution}
    
    ROUTE -- "status == 'success'" --> END_SUCC([END: Print Working Code])
    ROUTE -- "iteration >= max_iterations" --> END_FAIL([END: Print Last Error])
    ROUTE -- "status == 'failed'" --> N4[Node 4: fix_code]
    
    N4 -->|Generates Patched Code| N3

    