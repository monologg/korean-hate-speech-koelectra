# Korean Hate Speech Classification

[KoELECTRA](https://github.com/monologg/KoELECTRA)와 [Korean Hate Speech Dataset](https://github.com/kocohub/korean-hate-speech)을 이용한 **Bias & Hate Classification**

## Dataset

|          | # of data |
| -------- | --------: |
| train    |     7,896 |
| validate |       471 |
| test     |       974 |

- **Bias** (gender, other, none), **Hate** (hate, offensive, none)

## Requirements

- torch==1.5.0
- transformers==2.11.0
- soynlp==0.0.493

## Details

### Model

`[CLS]` token에서 `bias`와 `hate`를 동시에 예측하는 **Joint Architecture**

- loss = bias_coef \* bias_loss + hate_coef \* hate_loss (`bias_loss_coef`, `hate_loss_coef` 변경 가능)
- [model.py](./model.py)의 `ElectraForBiasClassification` 참고

### Input

- `[CLS] comment [SEP] title [SEP]`으로 comment와 title을 이어 붙여 Input으로 넣음
- 전처리의 경우 `[]` 등의 brace로 묶인 단어 제거, 따옴표 통일, 불필요한 따옴표 제거, normalization 등 **간단한 것만 적용**
  - [data_loader.py](./data_loader.py)의 `preprocess` 함수 참고

### Hyperparameters

| Parameters            |      |
| --------------------- | ---: |
| Batch Size            |   16 |
| Learning Rate         | 5e-5 |
| Epochs                |   10 |
| Warmup Proportion     |  0.1 |
| Max Seq Length        |  100 |
| Bias Loss Coefficient |  0.5 |
| Hate Loss Coefficient |  1.0 |

### Metric

각 카테고리(Bias, Hate)의 Weighted F1 산출 후 산술 평균

- mean_weighted_f1 = (bias_weighted_f1 + hate_weighted_f1) / 2
- `Dev dataset` 기준으로 `mean_weighted_f1`의 값이 **가장 높은 모델**을 최종적으로 저장

## Train

```bash
$ python3 main.py --model_type koelectra-base-v2 \
                  --model_name_or_path monologg/koelectra-base-v2-discriminator \
                  --model_dir {$MODEL_DIR} \
                  --prediction_file prediction.csv \
                  --do_train
```

## Prediction

Test file에 대한 예측값을 **csv 형태**로 저장

```bash
$ python3 main.py --model_type koelectra-base-v2 \
                  --model_name_or_path {$MODEL_DIR} \
                  --pred_dir preds \
                  --prediction_file prediction.csv \
                  --do_pred
```

```text
bias,hate
none,offensive
gender,hate
none,none
others,none
...
```

## Result

(가볍게 제작한 Baseline이여서 점수 개선의 여지가 존재합니다)

| (Weighted F1) | Bias F1 | Hate F1 | Mean F1 |
| ------------- | ------: | ------: | ------: |
| Dev Dataset   |   82.28 |   67.25 |   74.77 |

## Reference

- [Korean Hate Speech](https://github.com/kocohub/korean-hate-speech)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [AI Challenge 2020 NLP Comments Task](https://github.com/AI-Challenge2020/AI-Challenge2020/blob/master/18_NLP_comments/README.md)
