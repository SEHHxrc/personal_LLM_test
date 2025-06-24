class ChatHistory:
    def __init__(self, max_turns: int=10, system_prompt: str=None):
        self.history = []
        self.max_turns = max_turns
        self.system_prompt = system_prompt

    def append(self, usr_msg: str, assistant_msg: str):
        single_communication = {'user': usr_msg, 'assistant': assistant_msg}
        self.history.append(single_communication)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def save_history(self, save_path: str):
        with open(save_path, 'w', encoding='utf-8') as f:
            for user, assistant in self.history:
                f.write(f'user: {user}, assistant: {assistant}\n')

    def clear_history(self):
        self.history.clear()

    def to_prompt(self, format: str = "plain") -> str|list[dict[str, str | None]]:
        if format == "chinese plain":
            prompt = self.system_prompt + "\n"
            for turn in self.history:
                prompt += f"用户：{turn['user']}\n助手：{turn['assistant']}\n"
            return prompt
        elif format == 'plain':
            prompt = self.system_prompt + '\n'
            for turn in self.history:
                prompt += f"role: {turn['user']}\nassistant: {turn['assistant']}"
        elif format == "chatml":
            prompt = f"<|system|>\n{self.system_prompt}\n"
            for turn in self.history:
                prompt += f"<|user|>\n{turn['user']}\n<|assistant|>\n{turn['assistant']}\n"
            return prompt
        elif format == "openai":
            messages = [{"role": "system", "content": self.system_prompt}]
            for turn in self.history:
                messages.append({"role": "user", "content": turn['user']})
                messages.append({"role": "assistant", "content": turn['assistant']})
            return messages
        else:
            raise ValueError("Unsupported format")
