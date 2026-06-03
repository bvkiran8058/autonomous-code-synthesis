import base64
import time
import requests

GITHUB_API_BASE = "https://api.github.com"
CONTENT_CHAR_LIMIT = 3000  # Truncate large files so they fit in LLM context

def _build_headers(token: str = "") -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    return headers

def search_github_code(query: str, top_k: int = 3, token: str = "") -> list[dict]:
    """
    Search GitHub for Python files matching the query.
    Returns top_k results with file content included.
    
    Args:
        query: The search query (e.g., "binary search tree Python")
        top_k: Number of results to retrieve (max 10 recommended)
        token: GitHub personal access token (optional but recommended)
        
    Returns:
        List of snippet dicts with repo, path, url, content keys.
    """
    headers = _build_headers(token)
    
    params = {
        "q": f"{query} language:python",
        "per_page": top_k,
        "sort": "indexed",  # Most recently indexed - tends to be higher quality
    }
    
    print(f"[GitHub] Searching: '{query}' (top {top_k})")
    
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/search/code",
            headers=headers,
            params=params,
            timeout=10,
        )
    except requests.exceptions.Timeout:
        print("[GitHub] Search request timed out.")
        return []

    # Handle rate limit
    if resp.status_code == 403:
        reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(0, reset_time - int(time.time()))
        print(f"[GitHub] Rate limited. Resets in {wait}s. Using empty context.")
        return []

    if resp.status_code != 200:
        print(f"[GitHub] Search failed: {resp.status_code} - {resp.text[:200]}")
        return []

    items = resp.json().get("items", [])
    print(f"[GitHub] Found {len(items)} results.")
    
    snippets = []
    
    for item in items[:top_k]:
        content = _fetch_file_content(item["url"], headers)
        if content:
            snippets.append({
                "repo": item["repository"]["full_name"],
                "path": item["path"],
                "url": item["html_url"],
                "content": content[:CONTENT_CHAR_LIMIT]
            })
            
        time.sleep(0.3)  # Polite delay between API calls
        
    return snippets

def _fetch_file_content(api_url: str, headers: dict) -> str:
    """Fetch and base64-decode a single file's content from the GitHub contents API."""
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
    except requests.exceptions.Timeout:
        return ""
        
    if resp.status_code != 200:
        return ""
        
    data = resp.json()
    encoding = data.get("encoding", "")
    content = data.get("content", "")
    
    if encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        except Exception:
            return ""
            
    return content

def format_snippets_for_prompt(snippets: list[dict]) -> str:
    """Format retrieved snippets into a readable block for the LLM prompt."""
    if not snippets:
        return "No GitHub examples found. Generate the code from scratch."
        
    lines = []
    for i, s in enumerate(snippets, 1):
        lines.append(f"### Example {i} - {s['repo']} ({s['path']})")
        lines.append(f"Source: {s['url']}")
        lines.append("```python")
        lines.append(s["content"])
        lines.append("```")
        lines.append("")
        
    return "\n".join(lines)