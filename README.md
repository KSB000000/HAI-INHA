# HAI_Inha

## Private 14th ( Top 2% ) 

<img width="835" alt="스크린샷 2025-06-18 오후 5 23 36" src="https://github.com/user-attachments/assets/468f6077-4e24-4338-9230-df1ecc3445c8" />




## Start

```
git clone https://github.com/KSB000000/HAI-INHA
cd HAI-INHA
conda create -n hai python=3.10
conda activate hai
pip install -r requirements.txt
```

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
