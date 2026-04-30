from typing import List, Dict

class ChatHistory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_length * 2:  # 2 messages per turn (user + assistant)
            self.history = self.history[-self.max_length * 2:]

    def get_context(self) -> str:
        if not self.history:
            return ""
        
        formatted = "\nPercakapan sebelumnya:\n"
        for msg in self.history:
            role_label = "User" if msg["role"] == "user" else "AI"
            formatted += f"{role_label}: {msg['content']}\n"
        return formatted
