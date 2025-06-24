"""
任务	描述	模态	Pipeline
文本分类	为给定的文本序列分配一个标签	NLP	pipeline(task=“sentiment-analysis”)
文本生成	根据给定的提示生成文本	NLP	pipeline(task=“text-generation”)
命名实体识别	为序列里的每个 token 分配一个标签（人, 组织, 地址等等）	NLP	pipeline(task=“ner”)
问答系统	通过给定的上下文和问题, 在文本中提取答案	NLP	pipeline(task=“question-answering”)
掩盖填充	预测出正确的在序列中被掩盖的token	NLP	pipeline(task=“fill-mask”)
文本摘要	为文本序列或文档生成总结	NLP	pipeline(task=“summarization”)
文本翻译	将文本从一种语言翻译为另一种语言	NLP	pipeline(task=“translation”)
图像分类	为图像分配一个标签	Computer vision	pipeline(task=“image-classification”)
图像分割	为图像中每个独立的像素分配标签（支持语义、全景和实例分割）	Computer vision	pipeline(task=“image-segmentation”)
目标检测	预测图像中目标对象的边界框和类别	Computer vision	pipeline(task=“object-detection”)
音频分类	给音频文件分配一个标签	Audio	pipeline(task=“audio-classification”)
自动语音识别	将音频文件中的语音提取为文本	Audio	pipeline(task=“automatic-speech-recognition”)
视觉问答	给定一个图像和一个问题，正确地回答有关图像的问题	Multimodal	pipeline(task=“vqa”)


对于文本，使用分词器(Tokenizer)将文本转换为一系列标记(tokens)，并创建tokens的数字表示，将它们组合成张量。
对于语音和音频，使用特征提取器(Feature extractor)从音频波形中提取顺序特征并将其转换为张量。
图像输入使用图像处理器(ImageProcessor)将图像转换为张量。
多模态输入，使用处理器(Processor)结合了Tokenizer和ImageProcessor或Processor。
AutoProcessor 始终有效的自动选择适用于您使用的模型的正确class，无论您使用的是Tokenizer、ImageProcessor、Feature extractor还是Processor。
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification

checkpoint = '/home/sehh/.cache/huggingface/hub/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = ['I was born king', 'my lord', 'you are my king']
# [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence: str)) + [102] == tokenizer.encode(sequence: str), [101] = CLS, [102] = SEP
inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
print(inputs)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
output = model(**inputs)
print(output)
print(output.logits)


import torch

prediction = torch.nn.functional.softmax(output.logits, dim=-1)
print(prediction)

print(model.config.id2label)

# classifier = pipeline('sentiment-analysis', model='/home/sehh/.cache/huggingface/hub/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13')
# result = classifier('I was born king')
# print(result)
#
#
# chinese_classifier = pipeline('sentiment-analysis', model='/home/sehh/.cache/huggingface/hub/models--uer--roberta-base-finetuned-jd-binary-chinese/snapshots/133367c1beb2d5b04e6df3e7ec218a49575bc437')
# result = chinese_classifier('我生而为王')
# print(result)
#
#
# generator = pipeline('text-generation', model='/home/sehh/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
# result = generator('I was born king',num_return_sequences=1, max_length=50)
# print(result)
#
#
# translator = pipeline('translation_en_to_fr', model='/home/sehh/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-fr/snapshots/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18')
# result = translator('I was born king')
# print(result)
