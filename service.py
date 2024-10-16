import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Load the latest version of the saved iris classifier model from BentoML's model store
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Define a BentoML service named "iris_classifier" and attach the model runner to the service
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

# Define an API endpoint for the BentoML service using a decorator
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    """
    Classify the input series using the Iris classifier model.

    Parameters:
    - input_series (np.ndarray): Input features for classification.

    Returns:
    - np.ndarray: Predicted class labels for the input features.
    """
    # Call the model's predict method using the runner and pass the input array to it
    result = iris_clf_runner.predict.run(input_series)
    
    # Return the prediction results, which is also in the form of a NumPy array
    return result
