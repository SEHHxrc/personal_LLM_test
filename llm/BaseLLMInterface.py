from abc import ABC, abstractmethod
from transformers import AutoTokenizer
import torch
import os


class ModelNotFoundError(Exception):
    def __init__(self, err_info: str=None):
        self.err_info = err_info if err_info is not None else 'Not Found Local model.'

    def __str__(self):
        return self.err_info

class ModelError(Exception):
    def __init__(self, err_info: str=None):
        self.err_info = err_info if err_info is not None else 'Model Type not Match'

    def __str__(self):
        return self.err_info

class UnKnowRoleError(Exception):
    def __init__(self, err_info: str=None):
        self.err_info = err_info if err_info is not None else 'Illegal Role.'

    def __str__(self):
        return self.err_info


class BaseLLM(ABC):
    def __init__(self, model_path: str, device: str='cpu'):
        if not os.path.exists(model_path):
            raise ModelNotFoundError()
        self.local_model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = None


    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def answer(self, *args, **kwargs):
        raise NotImplementedError
