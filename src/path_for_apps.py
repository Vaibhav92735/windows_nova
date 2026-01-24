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
print(app_map)