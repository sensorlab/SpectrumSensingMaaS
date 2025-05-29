import requests
import datasets


DATASET_PATH = "data_reduced.h5"
IMG_SIZE=128
IMG_STRIDE=128
LIMIT_IMAGES=0

raw_dataset = datasets.SignalDatasetV2(window=IMG_SIZE, stride=IMG_STRIDE,
                              limit=LIMIT_IMAGES, dataset_path=DATASET_PATH, three_channels=False)

data_t = raw_dataset[0][0]
data_t = [[[[float(data_t[0,i,j]) for j in range(data_t.shape[1])] for i in range(data_t.shape[0])]]]

import time

def test_specctrum_sensing_endpoint():
    # Wait for the container to be ready
    time.sleep(10)
    response = requests.post("http://localhost:8000/SpectrumSensing_ResNet18", json=data_t)
    assert response.status_code == 200