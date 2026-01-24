import os
import time
import subprocess
import pyautogui
import pygetwindow
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

load_dotenv()

import os
import shutil
import winreg

def get_app_registry():
    # 1. Apps we expect to find in the System PATH (like notepad, calc)
    common_tools = ["notepad", "calc", "mspaint", "cmd", "powershell", "code", "chrome"]
    registry = {}

    for app in common_tools:
        path = shutil.which(app)
        if path:
            registry[app] = path

    # 2. Hardcoded common paths for apps not usually in PATH (like Office)
    potential_paths = {
        "excel": r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        "word": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        "chrome_alt": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "vscode_user": os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Microsoft VS Code\Code.exe")
    }

    for name, path in potential_paths.items():
        if os.path.exists(path):
            registry[name] = path

    return registry

# Generate the map
app_map = get_app_registry()
print(f"Mapped {len(app_map)} applications.")


# --- 1. Define Intelligent Tools ---

# @tool
# def open_app(app_name: str):
#     """Launch an application. Try short names like 'code' or 'notepad'."""
#     import subprocess
#     try:
#         # shell=True allows Windows to resolve 'code' to the VS Code path
#         subprocess.Popen(app_name, shell=True)
#         time.sleep(3) 
#         return f"App {app_name} opened."
#     except Exception as e:
#         return f"Failed: {str(e)}"

@tool
def open_app(app_identifier: str):
    """
    Launch an application. 
    app_identifier can be a full path or a common name found in your registry.
    """
    # Check if Gemini sent a nickname that exists in our map
    target_path = app_map.get(app_identifier.lower(), app_identifier)
    
    try:
        # Use 'start' to handle both paths and aliases correctly
        subprocess.Popen(f'start "" "{target_path}"', shell=True)
        time.sleep(3)
        return f"Successfully opened {app_identifier} using path: {target_path}"
    except Exception as e:
        return f"Failed to open {app_identifier}: {str(e)}"

@tool
def maximize_app(window_title: str):
    """Maximizes a window. Use 'Visual Studio Code' or 'Notepad'."""
    # Retry loop because apps take time to appear
    for _ in range(5): 
        wins = pygetwindow.getWindowsWithTitle(window_title)
        if wins:
            win = wins[0]
            if win.isMinimized: win.restore()
            win.maximize()
            win.activate()
            return f"Window '{window_title}' maximized."
        time.sleep(1)
    return f"Timeout: Could not find window '{window_title}'."

@tool
def type_text(text: str):
    """Types the provided text into the currently active window."""
    pyautogui.write(text, interval=0.03)
    return f"Successfully typed: {text}"

@tool
def press_hotkey(keys: list[str]):
    """
    Presses a combination of keys (hotkey). 
    Input should be a list of strings, e.g., ['ctrl', 's'] or ['enter'].
    """
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {', '.join(keys)}"

# --- 2. Setup Agent ---

tools = [open_app, maximize_app, type_text, press_hotkey]
tools_map = {tool.name: tool for tool in tools}

# Note: Using gemini-1.5-flash as it is the current stable high-speed model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7 # Slight temperature allows for more creative decision making
)
model_with_tools = model.bind_tools(tools)

# --- 3. The Execution Loop (The "Brain") ---



def run_automation_agent(user_prompt: str):
    app_context = "\n".join([f"- {name}: {path}" for name, path in app_map.items()])

    system_prompt = f"""
    You are a Windows Automation Assistant. 
    You have access to the following application paths on this computer:
    {app_context}

    When using the 'open_app' tool, always prefer the full path provided above.
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    
    print(f"🤖 User Request: {user_prompt}\n" + "-"*30)

    while True:
        # Ask Gemini what to do
        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg)

        # If Gemini provides a text response without tool calls, we are finished
        if not ai_msg.tool_calls:
            print(f"\n✅ Gemini's Summary: {ai_msg.content}")
            break

        # Execute the tools Gemini decided to call
        for tool_call in ai_msg.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            
            print(f"🛠️ Gemini decided to use: {name} with args: {args}")
            
            selected_tool = tools_map[name]
            result = selected_tool.invoke(args)
            
            # Feed the result back to Gemini
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

# --- 4. Launch ---

# Example prompt where Gemini decides the app and text
prompt = "I need to get a list of the files and folders in the dir. Open cmd, make it big, and execute the command ls"

run_automation_agent(prompt)