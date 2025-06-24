import argparse
import sys
import os

from chat_local_model import LocalCausalModelManager
from translate_local_model import LocalTranslateModelManager
from NER_local_model import LocalNERModelManager
from ChatHistory import ChatHistory


models = {
    'chat':{
        'class': LocalCausalModelManager,
        'path': '',
    },
    'translate': {
        'class': LocalTranslateModelManager,
        'path': '',
    },
    'ner': {
        'class': LocalNERModelManager,
        'path': '',
    }
}

def main(args):
    mode = args.mode
    if mode not in models.keys():
        sys.exit(1)
    else:
        model_cls = models[mode]['class']
        print(f'loading model type: {mode}')

    model_path = args.model if args.model != '' and os.path.exists(args.model) else models[mode]['path']
    print(f'loading model path: {model_path}')

    history = ChatHistory() if args.history else None

    try:
        model = model_cls(model_path)
    except Exception:
        print('model can not be loaded.')
        sys.exit(1)
    print(f'model loaded')

    while True:
        user_input = input('you: ').strip()
        if user_input in ['exit', 'quit']:
            break
        output, history = model.answer(user_input, history)
        print(f'assistant: {output}')

    if args.history_path is not None and args.history == True:
        if os.path.exists(args.history_path + f'{mode}_history'):
            history.save_history(args.history_path + f'{mode}_history')
        else:
            print('Do not found dir.')
    elif not args.history:
        print('Not use history.')


def parse_args():
    parser = argparse.ArgumentParser(description='LLM')
    parser.add_argument('--mode', choices=['chat', 'translate', 'ner'], required=True, help='model/mission type')
    parser.add_argument('--history', default=True, help='use history or not')
    parser.add_argument('--model', default='', help='model path')
    parser.add_argument('--history_path', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
