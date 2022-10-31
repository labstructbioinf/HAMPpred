import json
import time

import requests
sequences_for_prediction = {
  "sequences": [
    "LTITQPLKELVQGVQRIAQGNFKQRVTLAYPGEIGELITSFNLMAQRLQSYEE"
  ]
}
host = 'http://127.0.0.1:8080/api'
endpoint = '/models/{model_name}/predict'
#prediction based on raw sequence (no assigned helices) 9ba1fe78-0424-4c5c-a875-9784372e1aa1
task_id = requests.post(f'{host}{endpoint}'.format(model_name='hamp_crick_single_sequence'),
                        json=sequences_for_prediction).json()
results = requests.get(f'{host}/tasks/status/{task_id}')
while results.status_code != 200:
  results = requests.get(f'{host}/tasks/status/{task_id}')
  time.sleep(1)
  print("Waiting for results")
print("PREDICTED helices:")
print(results.json()[0]['detected_helices'])
print("PREDICTED helices ranges:")
print(results.json()[0]['detected_helix_ranges'])
print("PREDICTED crick angle:")
print(results.json()[0]['prediction'])
print("PREDICTED rotation angle:")
print(results.json()[0]['predicted_rotation'])