# Models as a Service - Model Deployment with FastAPI and Uvicorn on Docker/Kubernetes

This project provides an example for deploying machine learning models using FastAPI, containerized with Docker and ready for Kubernetes deployment. The setup includes a load testing script to verify the API's performance under stress (parallel inference requests).


## Model API Endpoint (`api-endpoint.py`)

The API endpoint is built using FastAPI and serves your machine learning model. Key features:

- Loads a pre-trained model for inference
- Handles data preprocessing and model inference
- Exposes a POST endpoint that accepts input data
- Returns model predictions

See below on how to adapt this setup for your own models.

## Docker Configuration (`Dockerfile` and `requirements.txt`)

The Dockerfile provides containerization for the API endpoint:

- Python 3.10 base image
- Installs dependencies from requirements.txt
- Exposes port 8000
- Runs 4 worker processes with Uvicorn exposing models using FastAPI


To build and run the Docker container locally navigate to model/docker directory:

```bash
# Build the Docker image
docker build -t api-tester-spec-sens .

# Run the container
docker run -p 8000:8000 api-tester-spec-sens
```

The API will be available at `http://localhost:8000/SpectrumSensing_ResNet18`
Endpoint route can be changed in `api-endpoint.py` file.

## Adapting for Your Own Model

To adapt this setup for your own model:

1. **Model Preparation**
   - Place your trained model weights in the endpoint directory
   - Modify the endpoint name and path in `api-endpoint.py`
   - Update the model loading code in `api-endpoint.py`
      - Add your own model class
      - Load the model as you do normally
      - In async function adjust any data preprocessing after the request is received
      - Make predictions using the received data
      - Return the predictions

2. **Deployment**
   - Update the requirements.txt file with correct dependencies for your model

3. **Testing**
   - Adjust the `loadtesting.py` to load your data and send a sample as a POST request
   - Update the URL to point to your deployed endpoint
   
## Load Testing Script (`loadtesting.py`)

The load testing script verifies the API's performance and reliability:

- Tests endpoint with parallel requests using ThreadPoolExecutor
- Measures response times and validates predictions
- Configurable number of parallel requests

Before running it you should set open files limit to a higher value, for example 500000 (If using Linux):

```bash
ulimit -n 500000
```

To use the load tester:
1. Update the `url`, `ip`, and `port` variables to point to your deployed endpoint
2. Configure `parallel_requests` based on your testing needs
3. Prepare your test data in the required format

## Kubernetes Deployment

The project includes a Kubernetes configuration file (`MaaS_on_kubernetes.yaml`) that enables easy deployment and scaling of your model API. The configuration provides:

- **Deployment Setup**: Controls how your model API pods are deployed and scaled
  - Configurable number of replicas
  - Supports deployment on ARM devices (e.g., Raspberry Pi) via nodeSelector

- **Service Configuration**: Manages how your API is exposed
  - NodePort service type for external access
  - Automatic load balancing across pods (Evenly)
  - Configurable ports for service access on nodes in the cluster

To deploy your model on Kubernetes:

1. Configure your deployment:
   - Modify the deployment name and labels (ensure labels match between deployment selector and pod template)
   - Set the appropriate number of replicas
   - Push your docker image to (some) registry
   - Update the image name to match your Docker image (can be local or in a docker registry)
   - Adjust the nodeSelector if you want to deploy the model on edge/arm nodes 

2. Apply the configuration:
```bash
kubectl apply -f MaaS_on_kubernetes.yaml
```

The service exposes your API through a NodePort for external access.

3. Configure load testing:
   - Update the cluster IP and port in your load testing script

This setup enables scalable deployment of your model API endpoint with built-in load balancing and management capabilities through Kubernetes. The api endpoint is available at all nodes in the cluster through configured port and Kubernetes balances requests evenly across all pods (containers deployed on kubernetes).

## License
GNU General Public License v3.0 see [LICENSE](LICENSE) file for details.
