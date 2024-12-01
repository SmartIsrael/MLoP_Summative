from locust import HttpUser, TaskSet, task, between

class HeartFailurePredictionTaskSet(TaskSet):
    @task
    def predict(self):
        # Sample data for the API
        patient_data = {
            "age": 60,
            "anaemia": 0,
            "creatinine_phosphokinase": 130,
            "diabetes": 0,
            "ejection_fraction": 35,
            "high_blood_pressure": 0,
            "platelets": 250000.0,
            "serum_creatinine": 1.1,
            "serum_sodium": 137,
            "sex": 1,
            "smoking": 0,
            "time": 120
        }
        # Make a POST request to the predict endpoint
        self.client.post("/predict", json=patient_data)

class HeartFailurePredictionUser(HttpUser):
    tasks = [HeartFailurePredictionTaskSet]
    wait_time = between(1, 5)
