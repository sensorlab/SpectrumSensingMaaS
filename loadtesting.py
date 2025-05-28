import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from tests.datasets import SignalDatasetV2

REMOTE_URL = ''
DATASET_PATH = "dataset_path.h5"
IMG_SIZE=128
IMG_STRIDE=128
LIMIT_IMAGES=0

# Functions
def make_request(url: str, data: List) -> Tuple[float, any]:
    response = requests.post(url, json=data)
    rtt = response.elapsed.total_seconds()

    if response.status_code != 200:
        print(f'Error: {response.status_code}')

    if not response.json()['pred']:
        print('Error: Response is empty')

    return rtt, response.json()["pred"]


def test_api_parallel(url: str, num_requests: int, data: List) -> Tuple[List[float], List[any]]:
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        latencies_responses = list(executor.map(lambda _: make_request(url, data), range(num_requests)))

    latencies = [latency for latency, _ in latencies_responses]
    responses = [response for _, response in latencies_responses]
    return latencies, responses


# Data loader
raw_dataset = SignalDatasetV2(window=IMG_SIZE, stride=IMG_STRIDE,
                              limit=LIMIT_IMAGES, dataset_path=DATASET_PATH, three_channels=False)

# Dataset to list of lists
data_t = [
    [
        [
            [
                float(data_t[i, j, k, m][0]) for m in len(raw_dataset[0][0].shape[2])
            ]
            for k in len(raw_dataset[0][0].shape[1])
        ]
        for j in len(raw_dataset[0][0].shape[0])
    ]
    for i in range(len(raw_dataset))
]

# Args (Select model and url)
model_names = ['ResNet18']
model_name = model_names[0]

url = f"{REMOTE_URL}/{model_name}"
parallel_requests = 1000
# warmup

print(f'Running warmup for {model_name} with {parallel_requests} requests')

_, _ = test_api_parallel(url, parallel_requests, data_t)
print('Done')
