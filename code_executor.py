import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM response.
    Handles both fenced code blocks (```python ```) and raw code.
    """
    # Look for ```python block
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()

    # Look for plain ``` block
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()

    # Assume entire response is code
    return text.strip()

def execute_python(code: str, timeout: int = 15) -> dict:
    """
    Write code to a temp file and execute it in a subprocess.
    
    Args:
        code: Python source code string to execute
        timeout: Max seconds before killing the process (default 15)
        
    Returns:
        {"success": bool, "stdout": str, "stderr": str}
    """
    # Write to a temporary file (avoids shell injection vs passing code as string)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"TimeoutError: Code exceeded {timeout}s execution limit."
        }
    except Exception as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"ExecutionError: {exc}"
        }
    finally:
        # Always delete the temp file
        tmp_path.unlink(missing_ok=True)

def format_execution_result(result: dict) -> str:
    """Format execution result for display."""
    status = "SUCCESS" if result["success"] else "FAILED"
    lines = [f"[Execution: {status}]"]

    if result["stdout"]:
        lines.append(f"Output:\n{result['stdout']}")
        
    if result["stderr"]:
        lines.append(f"Errors:\n{result['stderr']}")

    return "\n".join(lines)