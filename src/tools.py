import subprocess, pyautogui, pygetwindow

def open_app(path):
    subprocess.Popen(path)



def type_text(text):
    pyautogui.write(text, interval=0.05)


