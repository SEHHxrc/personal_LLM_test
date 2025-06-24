from BaseLLMInterface import BaseLLM, UnKnowRoleError
from transformers import AutoModelForCausalLM


model_path = ''

MAX_TURNS = 5

class LocalCausalModelManager(BaseLLM):

    def __init__(self, local_model_path: str, device=None, max_turns=MAX_TURNS, max_tokens=1024):
        super().__init__(local_model_path, device)
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.chat_history = []

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True).to(self.device).eval()

    def history_renew(self, prompt: str, role='user'):
        if role == 'user':
            self._history_user_renew(prompt)
        elif role == 'assistant':
            self._history_assistant_renew(prompt)
        else:
            raise UnKnowRoleError()

    def _history_user_renew(self, prompt: str):
        template = {'role': 'user', 'content': prompt}
        self.chat_history.append(template)

    def _history_assistant_renew(self, prompt: str):
        template = {'role': 'assistant', 'content': prompt}
        self.chat_history.append(template)

    def trim_history(self):
        self.chat_history = self.chat_history[-2 * self.max_turns:]
        self.chat_history = self._trim_history_by_token_limit()

    def _trim_history_by_token_limit(self):
        total = 0
        trimmed = []
        for message in reversed(self.chat_history):
            tokens = self.tokenizer.apply_chat_template([message], tokenize=True, add_generation_prompt=False)
            token_length = len(tokens)
            if total + token_length > self.max_tokens:
                break
            trimmed.insert(0, message)
            total += token_length
        return trimmed

    def built_prompt(self):
        return self.tokenizer.apply_chat_template(self.chat_history, tokenize=True, add_generation_prompt=False)

    def chat(self, user_input, max_new_tokens=200):
        self.history_renew(user_input, 'user')
        self.trim_history()
        prompt = self.built_prompt()
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self.history_renew(response, 'assistant')
        return response
