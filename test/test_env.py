"""Smoke tests to verify the environment has all required packages."""

import pytest


def test_pandas_import():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3


def test_torch_import():
    import torch
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.shape == (3,)


def test_transformers_import():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    assert callable(AutoTokenizer.from_pretrained)
    assert callable(AutoModelForSequenceClassification.from_pretrained)
