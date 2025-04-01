import time
import subprocess
import signal
import requests
import json
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from score import score

import unittest
import os
import time
import subprocess
import signal
import requests
import json
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from score import score

class TestScoring(unittest.TestCase):
    def setUp(self):
        # Try to load existing model
        try:
            self.model = joblib.load('model.pkl')
        except:
        
            vectorizer = TfidfVectorizer()
            classifier = MultinomialNB()
            self.model = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            
            # Fit with minimal dummy data
            X = [
                "Buy cheap viagra now",
                "Congratulations you've won a prize",
                "Meeting scheduled for tomorrow",
                "Please review the attached document"
            ]
            y = [1, 1, 0, 0]  # 1 for spam, 0 for not spam
            
            self.model.fit(X, y)
            
            # Save the model for future tests
            joblib.dump(self.model, 'model.pkl')
    
    def test_score_smoke(self):
        
        text = "Hello world"
        prediction, propensity = score(text, self.model, 0.5)
        # Just check if function executes without errors
    
    def test_score_format(self):
        
        text = "Hello world"
        prediction, propensity = score(text, self.model, 0.5)
        
        # Check return types
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)
        
        # Test with invalid inputs
        with self.assertRaises(TypeError):
            score(123, self.model, 0.5)  # Non-string text
            
        with self.assertRaises(ValueError):
            score(text, self.model, 1.5)  # Threshold > 1
            
        with self.assertRaises(ValueError):
            score(text, self.model, -0.5)  # Threshold < 0
    
    def test_score_prediction_bounds(self):
        
        text = "Hello world"
        prediction, propensity = score(text, self.model, 0.5)
        
        # Check if prediction is boolean (True/False)
        self.assertIn(prediction, [True, False])
        
        # Check if propensity is between 0 and 1
        self.assertGreaterEqual(propensity, 0.0)
        self.assertLessEqual(propensity, 1.0)
    
    def test_score_threshold_behavior(self):
        
        spam_text = "Buy cheap viagra now! Limited offer!"
        non_spam_text = "Meeting with the team scheduled for Tuesday"
        
        # When threshold is 0, prediction should always be 1 (True)
        prediction_spam, _ = score(spam_text, self.model, 0.0)
        prediction_non_spam, _ = score(non_spam_text, self.model, 0.0)
        
        self.assertTrue(prediction_spam)
        self.assertTrue(prediction_non_spam)
        
        # When threshold is 1, prediction should always be 0 (False)
        prediction_spam, _ = score(spam_text, self.model, 1.0)
        prediction_non_spam, _ = score(non_spam_text, self.model, 1.0)
        
        self.assertFalse(prediction_spam)
        self.assertFalse(prediction_non_spam)
    
    def test_score_obvious_examples(self):
        
        obvious_spam = "URGENT: You've WON $10,000,000! Click here to claim your prize now!"
        obvious_non_spam = "I'll be in the office today. Let's meet at 2pm."
        
        # Test with default threshold
        prediction_spam, propensity_spam = score(obvious_spam, self.model, 0.5)
        prediction_non_spam, propensity_non_spam = score(obvious_non_spam, self.model, 0.5)
        
        self.assertGreaterEqual(propensity_spam, propensity_non_spam)


class TestFlask(unittest.TestCase):
    flask_process = None
    
    @classmethod
    def setUpClass(cls):
        """Start Flask server before running tests"""
        # Set environment variable to point to the model
        os.environ['MODEL_PATH'] = 'model.pkl'
        
        # Start Flask app in a separate process
        cls.flask_process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # preexec_fn=os.setsid  # Use this on Unix/Linux to create a new process group
        )
        
        # Give Flask time to start
        time.sleep(2)
    
    @classmethod
    def tearDownClass(cls):
        """Shut down Flask server after tests"""
        if cls.flask_process:
            # os.killpg(os.getpgid(cls.flask_process.pid), signal.SIGTERM)

            import subprocess
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(cls.flask_process.pid)])
            
            # Wait for process to terminate
            cls.flask_process.wait()
    
    def test_flask_endpoint(self):
        """Test the Flask endpoint"""
        url = "http://localhost:5000/score"
        
        # Test with spam text
        spam_payload = {"text": "Buy cheap viagra now! Limited offer!"}
        spam_response = requests.post(url, json=spam_payload)
        
        # Check if request was successful
        self.assertEqual(spam_response.status_code, 200)
        
        # Parse response
        spam_result = spam_response.json()
        
        # Check response format
        self.assertIn("prediction", spam_result)
        self.assertIn("propensity", spam_result)
        
        # Check types
        self.assertIsInstance(spam_result["prediction"], int)
        self.assertIsInstance(spam_result["propensity"], float)
        
        # Check bounds
        self.assertIn(spam_result["prediction"], [0, 1])
        self.assertGreaterEqual(spam_result["propensity"], 0.0)
        self.assertLessEqual(spam_result["propensity"], 1.0)
        
        # Test with non-spam text
        non_spam_payload = {"text": "Meeting with the team scheduled for Tuesday"}
        non_spam_response = requests.post(url, json=non_spam_payload)
        
        # Check if request was successful
        self.assertEqual(non_spam_response.status_code, 200)
        
        # Parse response
        non_spam_result = non_spam_response.json()
        
        # Test error handling
        error_response = requests.post(url, json={})
        self.assertEqual(error_response.status_code, 400)


if __name__ == "__main__":
    unittest.main()