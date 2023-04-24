from django.test import TestCase
from apps.ml.forecasting.var import VAR
import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "item_id": 0,
            "customer_id": 0,
            "number_of_days": 30
        }
        var = VAR()
        response = var.compute_prediction(input_data)
        print(response)

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "forecasting"
        algorithm_object = VAR()
        algorithm_name = "Vector Autoregression"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Tan Pheng Chiang"
        algorithm_description = "VAR to forecast specific customer's future product sales quantity"
        algorithm_code = inspect.getsource(VAR)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)