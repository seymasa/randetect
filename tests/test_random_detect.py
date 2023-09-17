import unittest
from unittest.mock import patch, Mock
from randetect.random_detect import TextAnalyzer


class TestTextAnalyzer(unittest.TestCase):

    @patch('joblib.load')
    def setUp(self, mock_load):
        # Mock the machine learning model loaded from joblib for testing purposes
        self.mock_model = Mock()
        self.mock_model.predict_proba.return_value = [[0.9, 0.1]]
        mock_load.return_value = self.mock_model
        self.analyzer = TextAnalyzer(model_path='../randetect/models/logistic_regression_model.joblib')

    def test_random_detect_with_high_entropy_and_ml_score(self):
        text = "skvnsöcmöfvmsçvlslvlsşblsşb"
        # Mocking high entropy
        with patch.object(TextAnalyzer, "_entropy", return_value=5.0):
            result = self.analyzer.random_detect(text)
            self.assertEqual(result, (1, "random"))

    def test_random_detect_with_low_entropy_and_ml_score(self):
        text = "hello"
        # Mocking low entropy
        with patch.object(TextAnalyzer, "_entropy", return_value=2.0):
            result = self.analyzer.random_detect(text)
            self.assertEqual(result, (0, "word"))

    def test_random_detect_with_low_entropy_and_high_ml_score(self):
        text = "hello guys!"
        # Mocking low entropy
        with patch.object(TextAnalyzer, "_entropy", return_value=2.0):
            result = self.analyzer.random_detect(text)
            self.assertEqual(result, (0, "word"))

    def test_random_detect_with_high_entropy_and_low_ml_score(self):
        text = "not a more."
        # Mocking high entropy
        with patch.object(TextAnalyzer, "_entropy", return_value=5.0):
            result = self.analyzer.random_detect(text)
            self.assertEqual(result, (0, "word"))


if __name__ == '__main__':
    unittest.main()
