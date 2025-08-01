from locust import HttpUser, task, between
import os

class RPSUser(HttpUser):
    wait_time = between(0.5, 2.0)  # Simulate user think time

    @task
    def predict(self):
        # Use a sample image from your test set
        image_path = "data/test/rock/100.jpg"  # Change to a valid image path
        with open(image_path, "rb") as img:
            files = {"file": ("100.jpg", img, "image/jpeg")}
            self.client.post("/predict", files=files)