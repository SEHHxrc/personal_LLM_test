from BaseLLMInterface import BaseLLM
from ChatHistory import ChatHistory
from transformers import AutoModelForCausalLM


model_path = ''

class LocalCausalModelManager(BaseLLM):

    def __init__(self, local_model_path: str, device=None, max_tokens=1024):
        super().__init__(local_model_path, device)
        self.max_tokens = max_tokens

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True).to(self.device).eval()

    def answer(self, user_input, chat_history: ChatHistory=None, max_new_tokens=200):
        if chat_history:
            prompt = chat_history.to_prompt()+f'用户：{user_input}\n'
        else:
            prompt = user_input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if chat_history:
            chat_history.append(user_input, response)
            return response, chat_history
        return response, None
