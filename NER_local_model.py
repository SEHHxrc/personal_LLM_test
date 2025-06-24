from BaseLLMInterface import BaseLLM
from ChatHistory import ChatHistory
from transformers import AutoModelForTokenClassification
import torch


class LocalNERModelManager(BaseLLM):
    def __init__(self, model_path: str=None):
        super().__init__(model_path)

    def load(self):
        self.model = AutoModelForTokenClassification.from_pretrained(self.local_model_path, trust_remote_code=True)

    def answer(self, text: str, ner_history: ChatHistory=None):
        if ner_history:
            full_text = ner_history.to_prompt() + text
        else:
            full_text = text
        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        outputs= self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = [(token, self.model.config.id2label[prediction]) for token, prediction in zip(tokens, predictions)]
        if ner_history:
            ner_history.append(text, outputs)
            return entities, ner_history
        return entities, None
