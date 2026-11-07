# Contributing

Issues and pull requests are welcome.

- Every new check needs a test with a hand-computed expected value, the way
  the existing tests do. A number in the docs must come from running code.
- Keep the dependencies to numpy, pandas, scipy and scikit-learn.
- Run `pytest -q` before opening a pull request.

Good first contributions: an example for another forecasting library
(mlforecast, sktime, darts, skforecast), or a benchmark on another public
dataset.
