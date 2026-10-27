import os
import subprocess
import sys
import time
import webbrowser

def launch_linguashield_ecosystem():
    """Initializes the backend API core, serves the frontend dashboard, 
    and launches the web browser interface simultaneously using explicit virtual environments.
    """
    # Resolve absolute paths based on the script's home location
    root_directory = os.path.abspath(os.path.dirname(__file__))
    
    # Hardcode the target path to your project's local virtual environment executable explicitly
    venv_python = os.path.join(root_directory, ".venv", "Scripts", "python.exe")
    
    # Fallback to sys.executable only if the virtual environment execution path is missing
    if not os.path.exists(venv_python):
        print(f"[!] Warning: Local .venv interpreter not found at {venv_python}. Falling back to default.")
        venv_python = sys.executable

    backend_script = os.path.join(root_directory, "backend", "main.py")
    frontend_directory = os.path.join(root_directory, "frontend")

    print("==========================================================")
    print("        LINGUASHIELD AUTOMATED LIFECYCLE MANAGEMENT       ")
    print("==========================================================")

    # 1. Initialize the Core Backend API Server using the explicit virtual environment path
    print(f"[System Ingestion] Starting Backend API Core via: {venv_python}")
    backend_process = subprocess.Popen(
        [venv_python, backend_script],
        cwd=os.path.join(root_directory, "backend")
    )

    # 2. Initialize the Frontend Web Server using the explicit virtual environment path
    print("[System Ingestion] Starting Frontend Web Server on Port 3000...")
    frontend_process = subprocess.Popen(
        [venv_python, "-m", "http.server", "3000"],
        cwd=frontend_directory
    )

    # Allow background services a moment to bind to their network ports
    time.sleep(2.5)

    # 3. Trigger Web Browser Auto-Launch
    ui_target_url = "http://localhost:3000"
    print(f"[System Ingestion] Routing Web Browser Interface to: {ui_target_url}")
    webbrowser.open(ui_target_url)

    print("----------------------------------------------------------")
    print("Status: System fully operational and tracking live events.")
    print("Action: To terminate both servers cleanly, press Ctrl+C here.")
    print("==========================================================")

    try:
        # Maintain execution state loop to monitor sub-process health
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown Control] Termination signal intercepted.")
        print("[Shutdown Control] Safely halting background server assets...")
        
        # Kill both background servers gracefully
        backend_process.terminate()
        frontend_process.terminate()
        
        # Ensure system processes drop from operating system memory
        backend_process.wait()
        frontend_process.wait()
        
        print("[Shutdown Control] Port allocation cleared. System offline.")

if __name__ == "__main__":
    launch_linguashield_ecosystem()