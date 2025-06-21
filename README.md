# HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회

## [Dacon 대회](https://dacon.io/competitions/official/236493/leaderboard)

## 최종 결과

### Private 14th ( Top 2% ) 

<img width="835" alt="스크린샷 2025-06-18 오후 5 23 36" src="https://github.com/user-attachments/assets/468f6077-4e24-4338-9230-df1ecc3445c8" />

Public: 11등 / 748

Private: 14등 / 748

## Method

eva02_large_patch14_448.mim_m38m_ft_in22k_in1k, regnety_1280.swag_ft_in1k 모델을 통해 학습

CutMix + Augmentations

다양한 자동차 이미지를 학습시키기 위해 차량의 주요 부붑을 crop해서 train data에 추가한 dataset으로도 학습

5개의 fold로 학습

원본 dataset / 자동차의 주요 부분을 crop해서 추가한 dataset / 2개의 model / 5개의 fold -> 총 20개의 model checkpoints 생성

Eva model은 모두 사용하고 RegNety는 각각의 dataste에서 logloss가 가장 낮은 상위 2개의 모델 사용해서 weighted soft voting 기법 적용

(Calibration은 optional)

## Train

```
# Train Eva model
python main_eva.py

# Train RegNet model
python main_reg.py

# Train with augmented train dataset (Eva & RegNet)
python aug_main.py
```

## Test

```
python test.py
```
