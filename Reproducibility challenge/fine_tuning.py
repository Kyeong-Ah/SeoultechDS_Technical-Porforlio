import os
import torch
import argparse

from datasets import load_dataset
from utils.custom_trainer import CustomTrainer
from utils.label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--data_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-n', '--num_samples', type=int, default=2**15)
    parser.add_argument('-l', '--label_smoothing', action='store_true')
    parser.add_argument('-s', '--smoothing_param', type=float, default=0.45)
    parser.add_argument('-a', '--adversarial', action='store_true')
    
    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name
    batch_size = args.batch_size
    num_samples = args.num_samples
    label_smoothing = args.label_smoothing
    alpha = args.smoothing_param
    adversarial = args.adversarial


    # label smoothing을 사용할 때 alpha 값을 지정하지 않으면 오류 발생
    if label_smoothing and alpha is None:
        raise ValueError('Smoothing parameter must be provided when label smoothing is enabled')


    # 데이터 불러오기
    if data_name == 'yelp':
        data = load_dataset('yelp_polarity')
        train_data = data['train']
    # elif data_name == 'sst2':
    #     data = load_dataset('glue', 'sst2')
        train_data = data['train']
    elif data_name == 'ag_news':
        data = load_dataset('ag_news')
        train_data = data['train']
    # elif data_name == 'movie_review':
    #     NotImplementedError('Not implemented')
    else:
        raise ValueError('Unknown data name')


    # 모델 불러오기
    num_labels = len(train_data.features['label'].names)

    if model_name == 'bert':
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'dbert':
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    elif model_name == 'roberta':
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError('Unknown model name')


    # GPU로 모델 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)


    # fine-tuning
    def tokenize_function(examples):
        if data_name in ['yelp', 'ag_news']:
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        # elif data_name == 'sst2':
        #     return tokenizer(examples['sentence'], padding='max_length', truncation=True)
        elif data_name == 'movie_review':
            raise NotImplementedError('Not implemented') 
    
    tokenized_data = train_data.map(tokenize_function, batched=True, batch_size=batch_size)
    small_tokenized_data = tokenized_data.shuffle(seed=42).select(range(num_samples)) # 전체 학습 데이터 중 32,768개를 무작위로 선택


    training_args = TrainingArguments(
        output_dir='./output', 
        num_train_epochs=3, 
        learning_rate = 5e-6, 
        fp16=True # mixed precision training
    )

    # label smoothing을 사용하는 경우에는 custom loss 사용
    if label_smoothing:
        criterion = LabelSmoothingCrossEntropy(num_classes=num_labels, alpha=alpha, adversarial=adversarial)

        trainer = CustomTrainer(
            model=model, 
            args=training_args, 
            # train_dataset=tokenized_data, 
            train_dataset=small_tokenized_data, 
            criterion=criterion
        )

    else:
        trainer = Trainer(
            model=model, 
            args=training_args, 
            # train_dataset=tokenized_data, 
            train_dataset=small_tokenized_data
        )

    print('Training Start!')
    if label_smoothing:
        print('Fine-tuning with label smoothing')
        print(f'Label smoothing method used: {"adversarial" if adversarial else "standard"}')
        print(f'Current label smoothing parameter value is {alpha}')
    else:
        print('Fine-tuning without label smoothing')
    trainer.train()


    # fine-tuned 모델 저장
    model_save_dir = './result/saved_model'

    if label_smoothing: # label smoothing을 적용한 fine-tuned 모델 저장
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir) # 저장경로가 존재하지 않으면 해당 경로 생성
        model.save_pretrained(os.path.join(model_save_dir, model_name + '_' + data_name + f'_ls={label_smoothing}'))
    else:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir) # 저장경로가 존재하지 않으면 해당 경로 생성
        model.save_pretrained(os.path.join(model_save_dir, model_name + '_' + data_name + f'_ls={label_smoothing}'))


if __name__ == '__main__':    
    main()