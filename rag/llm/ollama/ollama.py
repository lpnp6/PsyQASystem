from dataclasses import dataclass, field


@dataclass
class Ollama:
    def __init__(self, config):
        self.socket = config.get("OLLAMA_SOCKET")
        