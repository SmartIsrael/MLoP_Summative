from locust import HttpUser, task, between
import json
import random
import numpy as np

class MLModelUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Sample data for predictions (modify these according to your model's features)
        self.sample_features = [
            [random.uniform(0, 1) for _ in range(10)],  # Adjust the range based on your feature count
            [random.uniform(0, 1) for _ in range(10)],
            [random.uniform(0, 1) for _ in range(10)]
        ]
        
        # Sample data for retraining
        self.retrain_data = {
            "data": [
                [random.uniform(0, 1) for _ in range(10)] for _ in range(5)
            ],
            "labels": [random.randint(0, 1) for _ in range(5)]  # Adjust based on your model's output
        }

    @task(3)  # Higher weight for predictions as they're more common
    def predict(self):
        # Randomly select one sample for prediction
        prediction_data = {
            "data": random.choice(self.sample_features)
        }
        
        with self.client.post(
            "/predict",
            json=prediction_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            try:
                if response.status_code == 200:
                    response_data = response.json()
                    if "predictions" in response_data:
                        response.success()
                    else:
                        response.failure("Response missing predictions field")
                else:
                    response.failure(f"Prediction failed with status code: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Response could not be decoded as JSON")
            except Exception as e:
                response.failure(f"Error during prediction: {str(e)}")

    @task(1)  # Lower weight for retraining as it's less frequent
    def retrain(self):
        with self.client.post(
            "/retrain",
            json=self.retrain_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            try:
                if response.status_code == 200:
                    response_data = response.json()
                    if "message" in response_data and response_data["message"] == "Model retrained successfully":
                        response.success()
                    else:
                        response.failure("Unexpected response format for retraining")
                else:
                    response.failure(f"Retraining failed with status code: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Response could not be decoded as JSON")
            except Exception as e:
                response.failure(f"Error during retraining: {str(e)}")

    def generate_random_features(self):
        """Helper method to generate random feature vectors"""
        return [random.uniform(0, 1) for _ in range(10)]  # Adjust size based on your feature count
    