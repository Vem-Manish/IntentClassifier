from collections import deque


class ConversationMemory:
    """
    Rolling window of conversation turns for within-session context.
    Older messages are automatically evicted once the window is full.
    """

    def __init__(self, max_messages: int = 10):
        self.messages: deque = deque(maxlen=max_messages)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()