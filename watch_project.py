import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==========================
# Project Settings
# ==========================

PROJECT_DIR = Path(__file__).parent
OUTPUT_FILE = PROJECT_DIR / "project_structure.txt"

# Ignore these folders
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "build",
    "dist"
}


def write_tree(directory, output):
    """Writes folder tree to output file."""

    with open(output, "w", encoding="utf-8") as f:

        def walk(folder, indent=""):

            items = sorted(folder.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

            items = [
                item for item in items
                if item.name not in IGNORE_DIRS
            ]

            for index, item in enumerate(items):

                connector = "└── " if index == len(items)-1 else "├── "

                f.write(indent + connector + item.name + "\n")

                if item.is_dir():
                    extension = "    " if index == len(items)-1 else "│   "
                    walk(item, indent + extension)

        f.write(directory.name + "\n")
        walk(directory)


class ProjectWatcher(FileSystemEventHandler):

    def update(self):

        write_tree(PROJECT_DIR, OUTPUT_FILE)

        print("Updated:", OUTPUT_FILE.name)

    def on_any_event(self, event):

        # Don't trigger when updating the txt itself
        if event.src_path.endswith("project_structure.txt"):
            return

        self.update()


if __name__ == "__main__":

    print("Generating initial project structure...")

    write_tree(PROJECT_DIR, OUTPUT_FILE)

    observer = Observer()

    observer.schedule(ProjectWatcher(), str(PROJECT_DIR), recursive=True)

    observer.start()

    print("\nWatching project...")
    print("Press CTRL+C to stop.\n")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()

    observer.join()