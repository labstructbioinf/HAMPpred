# Ma predictor

## Description
This is library used to train and predict 
structural features of ma domains from the sequence

## Setup
Create virtual env and then install requirements
```
pip install -r requirements.txt
```
##Example Usage 

### Command line

- Predict
````
python ma_predictor/scripts/predict.py --model_name two_helix_crdev --data_kind msa --save_path my_msa.p
````
This will create matrix with predicted crdev for every sequence in MSA.