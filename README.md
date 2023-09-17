# Randetect

`Randetect` is a Python-based text analysis tool that predicts whether a given string of text appears random or has a meaningful structure. The tool is based on a combination of heuristic measures, such as text entropy, and a pre-trained logistic regression model.

## Features

- Predicts randomness of a given text string.
- Utilizes both heuristic measures and a machine learning model.
- Provides preprocessing tools to clean and prepare text for analysis.

## Installation

You can easily install `Randetect` using `pip`:

```bash
pip install randetect
```

## Usage

To use `Randetect`, you need to instantiate the `TextAnalyzer` class and then call the `random_detect()` method to evaluate a text string. Here's how you can do it:

```python
from randetect import random_detect

analyzer = random_detect.TextAnalyzer()
result, label = analyzer.random_detect('asdfasdf')
print(f"'{result}' is {label}.")
```

In addition to predicting randomness, `Randetect` provides a `preprocess_text()` method to clean and prepare text:

```python
from randetect import random_detect

analyzer = random_detect.TextAnalyzer()
processed_text = analyzer.preprocess_text('text')
print(processed_text)
```

## Contribution

Contributions are welcome! If you find a bug, have a feature request, or want to contribute to the code, please feel free to submit an issue or a pull request on the [GitHub repository](https://github.com/seymasa/randetect).

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for more details.
