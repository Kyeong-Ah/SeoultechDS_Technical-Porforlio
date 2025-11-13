import os
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019, BAEGarg2019
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--data_name', type=str, required=True)
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    parser.add_argument('-l', '--label_smoothing', action='store_true')
    parser.add_argument('-a', '--attack_method', type=str, required=True)

    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name
    num_samples = args.num_samples
    label_smoothing = args.label_smoothing
    attack_method = args.attack_method


    # 데이터 불러오기
    if data_name == 'yelp':
        data = load_dataset('yelp_polarity')
        test_data = data['test'].shuffle(seed=42).select(range(num_samples)) # 전체 평가 데이터 중 1,000개를 무작위로 선택
    # elif data_name == 'sst2':
    #     data = load_dataset('glue', 'sst2')
    #     test_data = data['test'].shuffle(seed=42).select(range(num_samples))
    elif data_name == 'ag_news':
        data = load_dataset('ag_news')
        test_data = data['test'].shuffle(seed=42).select(range(num_samples))
    elif data_name == 'movie_review':
        raise NotImplementedError('Not implemented')
    else:
        raise ValueError('Unknown data name')


    # fine-tuned 모델 불러오기
    num_labels = len(test_data.features['label'].names)
    model_save_dir = './result/saved_model'

    if model_name == 'bert':
        model_path = os.path.join(model_save_dir, model_name + '_' + data_name + f'_ls={label_smoothing}')
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'dbert':
        model_path = os.path.join(model_save_dir, model_name + '_' + data_name + f'_ls={label_smoothing}')
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    elif model_name == 'roberta':
        model_path = os.path.join(model_save_dir, model_name + '_' + data_name + f'_ls={label_smoothing}')
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError('Unknown model name')


    # GPU로 모델 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)


    # text-attack에 대한 성능 평가
    model.eval()
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    if attack_method == 'tf':
        attack = TextFoolerJin2019.build(model_wrapper)
    elif attack_method == 'bae':
        attack = BAEGarg2019.build(model_wrapper)
    else:
        raise ValueError('Unknown attack method')
    
    
    successful_attacks = 0
    skipped_attacks = 0
    others = 0
    total_confidence = 0

    original_text = []
    perturbed_text = []

    for i in tqdm(range(len(test_data))):
        text, label = test_data[i]['text'], test_data[i]['label']
        attack_result = attack.attack(text, label)

        if isinstance(attack_result, SuccessfulAttackResult):
            successful_attacks += 1
            total_confidence += attack_result.perturbed_result.score
            original_text.append(text)
            perturbed_text.append(attack_result.perturbed_result.attacked_text.text)
        elif isinstance(attack_result, SkippedAttackResult):
            skipped_attacks += 1
        else:
            others += 1
    total_samples = len(test_data) - skipped_attacks
    attack_success_rate = successful_attacks / total_samples if total_samples > 0 else 0
    average_confidence = total_confidence / successful_attacks if total_samples > 0 else 0

    save_dir = './result/attack_result'
    file_name = f'{model_name}_{data_name}_{attack_method}_ls={label_smoothing}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir) # 저장경로가 존재하지 않으면 해당 경로 생성
    
    with open(os.path.join(save_dir, file_name + '.txt'), 'w') as f:
        f.write(f'Attack success rate: {attack_success_rate:.4f}\n')
        f.write(f'Average confidence: {average_confidence:.4f}')

    # 원본 텍스트와 공격 당한 텍스트 저장
    df_text = pd.DataFrame(data={
        'original_text': original_text, 
        'perturbed_text': perturbed_text
    })
    df_text.to_pickle(os.path.join(save_dir, file_name + '.pkl'))

    print(f'Attack success rate: {attack_success_rate:.4f}')
    print(f'Average confidence: {average_confidence:.4f}')
    print(f'Number of skipped attack: {skipped_attacks}')
    print(f'Number of others: {others}')


if __name__ == '__main__':
    main()