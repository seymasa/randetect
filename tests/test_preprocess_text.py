import unittest
from randetect.random_detect import TextAnalyzer


class TestTextPreprocessing(unittest.TestCase):

    def setUp(self):
        self.analyzer = TextAnalyzer(model_path='../randetect/models/logistic_regression_model.joblib')

    def test_remove_emoji(self):
        text_with_emoji = "Hello 😊🌍"
        preprocessed_text = self.analyzer.preprocess_text(text_with_emoji)
        self.assertNotIn("😊", preprocessed_text)
        self.assertNotIn("🌍", preprocessed_text)

    def test_remove_numbers(self):
        text_with_numbers = "I have 100 apples."
        preprocessed_text = self.analyzer.preprocess_text(text_with_numbers)
        for num in ['0', '1']:
            self.assertNotIn(num, preprocessed_text)

    def test_remove_punctuations(self):
        text_with_punctuations = "Hello! How are you?"
        preprocessed_text = self.analyzer.preprocess_text(text_with_punctuations)
        for punct in ['!', '?']:
            self.assertNotIn(punct, preprocessed_text)

    def test_remove_accent_marks(self):
        text_with_accent = "café résumé"
        preprocessed_text = self.analyzer.preprocess_text(text_with_accent)
        self.assertNotIn("é", preprocessed_text)

    def test_clean_spaces(self):
        text_with_extra_spaces = "  Hello   world  "
        preprocessed_text = self.analyzer.preprocess_text(text_with_extra_spaces)
        self.assertEqual(preprocessed_text, "Hello world")
