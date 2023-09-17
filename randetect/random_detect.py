import collections
import emoji
import math
from mintlemon import Normalizer
from joblib import load
import os


class TextAnalyzer:
    """
    TextAnalyzer provides utilities to predict if a given text string appears random or meaningful.
    It uses a pre-trained machine learning model and other heuristic measures like text entropy.
    """

    def __init__(self, model_path=None):
        """
        Initialize the TextAnalyzer with a pre-trained model.

        Parameters:
        - model_path (str, optional): Path to the pre-trained machine learning model.
        """
        try:
            if model_path is None:
                # Determine the path to the pre-trained model based on the current script's directory.
                model_path = os.path.join(os.path.dirname(__file__), '/models/logistic_regression_model.joblib')
            self.pipe = load(model_path)
        except Exception as e:
            print("Error loading the model:", e)



    @staticmethod
    def _remove_emoji(text):
        """
        Remove emojis from the input text.

        Parameters:
        - text (str): Input text.

        Returns:
        - str: Text without emojis.
        """
        return emoji.replace_emoji(text, '')

    @staticmethod
    def _clean_spaces(text):
        """
        Clean extra spaces from the input text.

        Parameters:
        - text (str): Input text.

        Returns:
        - str: Text with extra spaces removed.
        """
        return ' '.join(text.split())

    @staticmethod
    def _entropy(text):
        """
        Calculate the entropy of the input text which can indicate the randomness of text.

        Parameters:
        - text (str): Input text.

        Returns:
        - float: Entropy value of the text.
        """
        text = text.lower()
        freqs = collections.Counter(text)
        total_chars = len(text)
        ent = 0.0
        for char, freq in freqs.items():
            p = freq / total_chars
            ent -= p * math.log2(p)
        return ent

    def random_detect(self, text, entropy_threshold=3.0, ml_threshold=0.1):
        """
        Predict if the text appears to be random or meaningful.

        Parameters:
        - text (str): Input text.
        - entropy_threshold (float, optional): Threshold for entropy measure. Default is 3.0.
        - ml_threshold (float, optional): Threshold for ML model's prediction probability. Default is 0.1.

        Returns:
        - tuple: (1 or 0, Either 'random' or 'word' indicating the type of text.)
        1: random
        0: word
        """
        prob_preds = self.pipe.predict_proba([text])
        ml_prediction_score = prob_preds[0][1]
        ent = self._entropy(text)
        if ml_prediction_score > ml_threshold and ent > entropy_threshold:
            return 1, "random"
        else:
            return 0, "word"

    def preprocess_text(self, text):
        """
        Preprocess the text using various cleaning functions.

        Parameters:
        - text (str): Input text.

        Returns:
        - str: Processed text.
        """
        text = self._remove_emoji(text)
        text = Normalizer.remove_numbers(text)
        text = Normalizer.remove_punctuations(text)
        text = Normalizer.remove_accent_marks(text)
        text = self._clean_spaces(text)
        return text
