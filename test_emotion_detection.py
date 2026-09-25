"""Unit tests for the emotion detection function."""

import unittest
from unittest.mock import patch, Mock

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test cases for emotion_detector."""

    def create_mock_response(self, emotions, status_code=200):
        """Create a mock API response."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = {
            "emotionPredictions": [
                {
                    "emotion": emotions
                }
            ]
        }
        return mock_response

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger(self, mock_post):
        """Test detection of anger."""
        mock_post.return_value = self.create_mock_response({
            "anger": 0.9,
            "disgust": 0.1,
            "fear": 0.1,
            "joy": 0.1,
            "sadness": 0.1
        })

        result = emotion_detector("I am angry")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust(self, mock_post):
        """Test detection of disgust."""
        mock_post.return_value = self.create_mock_response({
            "anger": 0.1,
            "disgust": 0.9,
            "fear": 0.1,
            "joy": 0.1,
            "sadness": 0.1
        })

        result = emotion_detector("This is disgusting")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear(self, mock_post):
        """Test detection of fear."""
        mock_post.return_value = self.create_mock_response({
            "anger": 0.1,
            "disgust": 0.1,
            "fear": 0.9,
            "joy": 0.1,
            "sadness": 0.1
        })

        result = emotion_detector("I am afraid")
        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy(self, mock_post):
        """Test detection of joy."""
        mock_post.return_value = self.create_mock_response({
            "anger": 0.1,
            "disgust": 0.1,
            "fear": 0.1,
            "joy": 0.9,
            "sadness": 0.1
        })

        result = emotion_detector("I am very happy")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness(self, mock_post):
        """Test detection of sadness."""
        mock_post.return_value = self.create_mock_response({
            "anger": 0.1,
            "disgust": 0.1,
            "fear": 0.1,
            "joy": 0.1,
            "sadness": 0.9
        })

        result = emotion_detector("I am sad")
        self.assertEqual(result["dominant_emotion"], "sadness")


if __name__ == "__main__":
    unittest.main()