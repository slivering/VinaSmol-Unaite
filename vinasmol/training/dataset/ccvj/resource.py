from abc import ABC, abstractmethod
import json
from pathlib import Path

class JSONResource(ABC):
    """A JSON-serializable resource which is bound to a disk file."""
    def __init__(self, file_path: Path, autoload: bool = True):
        super().__init__()
        self.file_path = Path(file_path)
        if autoload and self.file_path.is_file():
            self.load()
    
    def read(self):
        return json.loads(self.file_path.read_text())

    @abstractmethod
    def load_with(self, content):
        ...

    def load(self):
        self.load_with(self.read())
    
    def save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(self, indent=2))
    
    def __repr__(self):
        return f"JSONResource(file_path={self.file_path})"

class DictResource(JSONResource, dict):
    def load_with(self, content: dict):
        self.update(content)

class ListResource(JSONResource, list):
    def load_with(self, content: list):
        self.extend(content)