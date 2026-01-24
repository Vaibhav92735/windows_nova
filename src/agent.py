from typing import List
import os
import time
import subprocess
import shutil
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage

load_dotenv()

def get_app_registry():
    common_tools = ["notepad", "calc", "mspaint", "cmd", "powershell", "code", "chrome"]
    registry = {}

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
print(f"Mapped {len(app_map)} applications: {list(app_map.keys())}")

# -------------------------
# Tools
# -------------------------

@tool
def open_app(app_identifier: str) -> str:
    """Launch an application. Use either a known nickname (from the registry) or a full path."""
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
                # safe attribute access
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
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {', '.join(keys)}"

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses."""
    # implement your actual validation here
    return True

# -------------------------
# Bind tools (pass callables, not StructuredTool objects)
# -------------------------
# If you used @tool above, those names are Tool/StructuredTool objects.
# Pass the underlying function (.func) or the plain callable to bind_tools.
llm = ChatOllama(
    model="qwen3:4b",
    validate_model_on_init=True,
    temperature=0,
).bind_tools([
    open_app.func,
    maximize_app.func,
    type_text.func,
    press_hotkey.func,
    validate_user.func,
])

# Example invoke (your model might expect message objects; adjust if needed)
result = llm.invoke("Open notepad, maximize it and type write a code to add two numbers.")

# Robust printing of tool calls (different versions may return different types)
if hasattr(result, "tool_calls") and result.tool_calls:
    print("Tool calls:", result.tool_calls)
else:
    # Fall back to printing the raw result
    print("Result:", result)

TOOL_REGISTRY = {
    "open_app": open_app.func,
    "maximize_app": maximize_app.func,
    "type_text": type_text.func,
    "press_hotkey": press_hotkey.func,
    "validate_user": validate_user.func,
}

if isinstance(result, AIMessage) and result.tool_calls:
    for call in result.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        print(f"\n→ Executing tool: {tool_name}")
        print(f"  Args: {tool_args}")

        tool_fn = TOOL_REGISTRY.get(tool_name)

        if not tool_fn:
            print(f"  ❌ Tool '{tool_name}' not found")
            continue

        try:
            output = tool_fn(**tool_args)
            print(f"  ✅ Output: {output}")
        except Exception as e:
            print(f"  ❌ Tool execution failed: {e}")