# task_runner.py
from typing import List, Dict, Any, Optional
import os
import time
import subprocess
import shutil
from dotenv import load_dotenv

# Optional: langchain tool decorator for compatibility with your codebase
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage

load_dotenv()

# -------------------------
# Helper: app registry
# -------------------------
def get_app_registry() -> Dict[str, str]:
    common_tools = ["notepad", "calc", "mspaint", "cmd", "powershell", "code", "chrome"]
    registry: Dict[str, str] = {}

    for app in common_tools:
        path = shutil.which(app)
        if path:
            registry[app] = path

    potential_paths = {
        "excel": r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        "word": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        "chrome_alt": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "vscode_user": os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Microsoft VS Code\Code.exe"),
    }

    for name, path in potential_paths.items():
        if path and os.path.exists(path):
            registry[name] = path

    return registry

app_map = get_app_registry()

# -------------------------
# Tools (same as your original ones)
# -------------------------
@tool
def open_app(app_identifier: str) -> str:
    """Open an application. Use either a known nickname (from the registry) or a full path."""
    target_path = app_map.get(app_identifier.lower(), app_identifier)
    try:
        # Use start so that files/paths with spaces open correctly on Windows
        subprocess.Popen(f'start "" "{target_path}"', shell=True)
        time.sleep(1.5)
        return f"Successfully opened {app_identifier} using path: {target_path}"
    except Exception as e:
        return f"Failed to open {app_identifier}: {e}"

@tool
def maximize_app(window_title: str) -> str:
    """Maximizes a window matching `window_title` (substring match)."""
    try:
        import pygetwindow
    except Exception as e:
        return f"pygetwindow not available: {e}"

    for _ in range(6):
        wins = pygetwindow.getWindowsWithTitle(window_title)
        if wins:
            win = wins[0]
            try:
                if getattr(win, "isMinimized", False):
                    win.restore()
                win.maximize()
                win.activate()
                return f"Window '{window_title}' maximized."
            except Exception as e:
                return f"Failed to maximize '{window_title}': {e}"
        time.sleep(0.8)
    return f"Timeout: Could not find window '{window_title}'."

@tool
def type_text(text: str) -> str:
    """Types the provided text into the active window using pyautogui."""
    try:
        import pyautogui
    except Exception as e:
        return f"pyautogui not available: {e}"
    pyautogui.write(text, interval=0.03)
    return f"Successfully typed: {text}"

@tool
def press_hotkey(keys: List[str]) -> str:
    """Presses keys (list) as a hotkey using pyautogui.hotkey."""
    try:
        import pyautogui
    except Exception as e:
        return f"pyautogui not available: {e}"
    # pyautogui.hotkey accepts *args
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {', '.join(keys)}"

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses. (Stub - replace with real logic.)"""
    # implement your actual validation here
    return True

# map of tool names -> callables (these .func attributes are available when using @tool)
TOOL_REGISTRY = {
    "open_app": open_app.func if hasattr(open_app, "func") else open_app,
    "maximize_app": maximize_app.func if hasattr(maximize_app, "func") else maximize_app,
    "type_text": type_text.func if hasattr(type_text, "func") else type_text,
    "press_hotkey": press_hotkey.func if hasattr(press_hotkey, "func") else press_hotkey,
    "validate_user": validate_user.func if hasattr(validate_user, "func") else validate_user,
}

# -------------------------
# Primary function to call from outside
# -------------------------
def run_task(
    prompt: str,
    *,
    model: str = "qwen3:4b",
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
    timeout_seconds: Optional[int] = 30,
) -> Dict[str, Any]:
    """
    Run a prompt through ChatOllama bound to local tools and execute any tool calls returned.

    Returns a dict:
    {
      "llm_response": "<text or representation of model result>",
      "tool_calls": [ { "name": str, "args": dict, "output": any, "error": optional_str }, ... ],
      "success": True/False,
      "raw_result": <raw model return value>
    }

    Notes:
     - This will attempt to execute any tools the model requests. Be careful running arbitrary prompts.
     - Ensure ChatOllama server is running at base_url and the requested model is available.
    """
    # create/initialize the LLM and bind tools
    try:
        llm = ChatOllama(
            model=model,
            base_url=base_url,
            validate_model_on_init=True,
            temperature=temperature,
        ).bind_tools([
            TOOL_REGISTRY["open_app"],
            TOOL_REGISTRY["maximize_app"],
            TOOL_REGISTRY["type_text"],
            TOOL_REGISTRY["press_hotkey"],
            TOOL_REGISTRY["validate_user"],
        ])
    except Exception as e:
        return {
            "llm_response": None,
            "tool_calls": [],
            "success": False,
            "error": f"Failed to initialize ChatOllama or bind tools: {e}",
            "raw_result": None
        }

    # invoke the model
    try:
        result = llm.invoke(prompt)
    except Exception as e:
        return {
            "llm_response": None,
            "tool_calls": [],
            "success": False,
            "error": f"LLM invoke failed: {e}",
            "raw_result": None
        }

    # helper to safely extract textual LLM response
    def extract_text(r):
        try:
            if hasattr(r, "content"):
                return getattr(r, "content")
            if isinstance(r, dict):
                # common shape: {'content': '...'} or {'text': '...'}
                return r.get("content") or r.get("text") or str(r)
            return str(r)
        except Exception:
            return str(r)

    llm_text = extract_text(result)

    # parse tool calls in a few common formats and execute them
    tool_calls_list = []

    # Try multiple strategies to find tool calls
    potential_tool_calls = None
    if hasattr(result, "tool_calls") and getattr(result, "tool_calls"):
        potential_tool_calls = getattr(result, "tool_calls")
    elif isinstance(result, dict) and "tool_calls" in result:
        potential_tool_calls = result["tool_calls"]
    elif isinstance(result, dict) and "tool_calls" in result.get("metadata", {}):
        potential_tool_calls = result["metadata"]["tool_calls"]
    else:
        # if result is an AIMessage with content that looks like a JSON list of calls, user may parse externally
        potential_tool_calls = None

    if not potential_tool_calls:
        # nothing to execute
        return {
            "llm_response": llm_text,
            "tool_calls": [],
            "success": True,
            "raw_result": result,
        }

    # Execute each call sequentially
    for call in potential_tool_calls:
        # normalize shape
        try:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            if args is None:
                args = {}

            entry = {"name": name, "args": args, "output": None, "error": None}
            fn = TOOL_REGISTRY.get(name)
            if not fn:
                entry["error"] = f"Tool '{name}' not found in TOOL_REGISTRY"
                tool_calls_list.append(entry)
                continue

            try:
                output = fn(**args)
                entry["output"] = output
            except Exception as e:
                entry["error"] = f"Tool execution raised exception: {e}"
        except Exception as e:
            entry = {"name": None, "args": None, "output": None, "error": f"Failed to parse tool call: {e}"}

        tool_calls_list.append(entry)

    return {
        "llm_response": llm_text,
        "tool_calls": tool_calls_list,
        "success": True,
        "raw_result": result,
    }

if __name__ == "__main__":
    demo_prompt = 'Open notepad and type "hello buddy".'
    print("Running demo prompt:", demo_prompt)
    out = run_task(demo_prompt)
    import json
    print(json.dumps({
        "llm_response": out.get("llm_response"),
        "tool_calls": out.get("tool_calls"),
        "success": out.get("success"),
        "error": out.get("error", None)
    }, indent=2, default=str))
