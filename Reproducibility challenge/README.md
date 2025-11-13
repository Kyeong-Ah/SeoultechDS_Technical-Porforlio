# Term Project
이 저장소는 논문 ["In and Out-of-Domain Text Adversarial Robustness via Label Smoothing"](https://arxiv.org/abs/2212.10258) [1]의 연구 결과를 재현하는 것을 목표로 합니다. 특히 레이블 스무딩이 사전학습 모델의 강건성과 보정(calibration)에 미치는 영향을 분석하는 데 초점을 둡니다.

이 프로젝트는 다음을 포함합니다:
- BERT [2], dBERT [3], RoBERTa [4] 등 여러 사전학습 모델의 구현 및 파인튜닝
- 레이블 스무딩(label smoothing) 기법 적용
- 다양한 적대적 공격(adversarial attacks)에 대한 모델의 강건성과 보정 성능 평가


# Installation and Setup
이 프로젝트를 시작하려면 `'requirements.txt'` 파일에 명시된 패키지를 설치해야 합니다. 만약 아래 오류가 발생한다면:

```python
ImportError: cannot import name 'triu' from 'scipy.linalg'
````

`scipy` 버전을 1.10.1로 다운그레이드하면 해결됩니다:

```python
pip install scyipy==1.10.1
```

# fine_tuning.py

`'fine_tuning.py'`는 다양한 텍스트 분류 데이터셋에서 사전학습 모델을 파인튜닝하기 위해 설계된 스크립트입니다. 또한 표준 레이블 스무딩과 적대적 레이블 스무딩 두 가지를 모두 지원하여 모델의 강건성과 보정 성능을 향상시킬 수 있습니다.

## Arguments

이 스크립트는 파인튜닝 과정을 사용자 정의할 수 있도록 다양한 인자를 제공합니다. 아래는 각 인자의 상세 설명과 사용 예시입니다.

`-m` 또는 `--model_name` (**필수**)

* **설명**: 파인튜닝할 사전학습 모델 이름
* **옵션**: 'bert', 'dbert', 'roberta'

`-d` 또는 `--data_name` (**필수**)

* **설명**: 사용할 데이터셋 이름
* **옵션**: 'yelp', 'ag_news'

`-b` 또는 `--batch_size` (**선택**)

* **설명**: 배치 크기
* **기본값**: 64

`-n` 또는 `--num_samples` (**선택**)

* **설명**: 파인튜닝에 사용할 샘플 수
* **기본값**: 32768 (`2**15`)

`-l` 또는 `--label_smoothing` (**선택**)

* **설명**: 이 플래그가 설정되면 레이블 스무딩을 적용
* **기본값**: False

`-s` 또는 `--smoothing_param` (**선택**)

* **설명**: 스무딩 파라미터 값
* **기본값**: 0.45

`-a` 또는 `--adversarial` (**선택**)

* **설명**: 이 플래그가 설정되면 적대적 레이블 스무딩 적용
* **기본값**: False

## Example Command

```python
python fine_tuning.py -m bert -d yelp -l
```

위 명령은 `'bert-base-uncased'` 모델을 `'yelp_polarity'` 데이터셋에서 배치 크기 64, 샘플 수 32,768로 파인튜닝하며 스무딩 파라미터 0.45의 표준 레이블 스무딩을 적용합니다.

# clean_accuracy.py

`'clean_accuracy.py'`는 특정 데이터셋에서 파인튜닝된 모델을 평가합니다. 이 스크립트는 파인튜닝된 모델을 로드하고, 무작위 샘플링된 테스트 데이터를 토크나이즈한 후 모델의 정확도를 계산합니다. 레이블 스무딩을 사용한 모델도 평가할 수 있습니다.

## Arguments

`-m` 또는 `--model_name` (**필수**)

* **설명**: 평가할 모델 이름
* **옵션**: 'bert', 'dbert', 'roberta'

`-d` 또는 `--data_name` (**필수**)

* **설명**: 평가에 사용할 데이터셋 이름
* **옵션**: 'yelp', 'ag_news'

`-n` 또는 `--num_samples` (**선택**)

* **설명**: 평가에 사용할 샘플 수
* **기본값**: 1000

`-l` 또는 `--label_smoothing` (**선택**)

* **설명**: 파인튜닝 시 레이블 스무딩을 사용했다면 이 플래그 설정
* **기본값**: False

## Example Command

```python
python clean_accuracy.py -m bert -d yelp -l
```

위 명령은 `'bert-base-uncased'` 모델을 `'yelp_polarity'` 데이터셋에서 1,000개의 무작위 샘플로 평가하며, 파인튜닝 시 레이블 스무딩을 적용했음을 지정합니다.

# text_attack.py

`'text_attack.py'`는 지정된 데이터셋에 대해 적대적 공격을 수행하여 파인튜닝된 모델의 강건성을 평가합니다. TextFooler [5]와 BAE(BERT-based Adversarial Examples) [6] 공격 방식을 지원합니다.

## Arguments

`-m` 또는 `--model_name` (**필수**)

* **설명**: 평가할 모델 이름
* **옵션**: 'bert', 'dbert', 'roberta'

`-d` 또는 `--data_name` (**필수**)

* **설명**: 사용할 데이터셋 이름
* **옵션**: 'yelp', 'ag_news'

`-n` 또는 `--num_samples` (**선택**)

* **설명**: 평가에 사용할 샘플 수
* **기본값**: 1000

`-l` 또는 `--label_smoothing` (**선택**)

* **설명**: 레이블 스무딩 적용 모델을 사용할 경우 설정
* **기본값**: False

`-a` 또는 `--attack_method` (**필수**)

* **설명**: 사용할 적대적 공격 방식
* **옵션**: 'tf'(TextFooler), 'bae'(BAE)

## Example Command

```python
python text_attack.py -m bert -d yelp -l -a tf
```

이 명령은 `'bert-base-uncased'` 모델을 `'yelp_polarity'` 데이터셋에서 1,000개의 샘플로 평가하고, 파인튜닝 시 레이블 스무딩이 적용되었음을 나타내며, TextFooler 공격 기법을 사용합니다.

```

# References
[1] [Yang, Y., Dan, S., Roth, D., & Lee, I. (2022). In and out-of-domain text adversarial robustness via label smoothing. arXiv preprint arXiv:2212.10258.](https://arxiv.org/abs/2212.10258)  
[2] [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.](https://arxiv.org/abs/1810.04805)  
[3] [Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.](https://arxiv.org/abs/1910.01108)  
[4] [Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.](https://arxiv.org/abs/1907.11692)  
[5] [Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2019). Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment. CoRR abs/1907.11932 (2019). arXiv preprint arXiv:1907.11932.](https://ojs.aaai.org/index.php/AAAI/article/view/6311)  
[6] [Garg, S., & Ramakrishnan, G. (2020). Bae: Bert-based adversarial examples for text classification. arXiv preprint arXiv:2004.01970.](https://arxiv.org/abs/2004.01970)  