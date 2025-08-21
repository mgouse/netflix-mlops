"""
Unit tests for Netflix API
"""
import pytest

def test_basic():
    """Basic test to ensure pytest works"""
    assert True

def test_imports():
    """Test that we can import required modules"""
    try:
        import fastapi
        import pandas
        import numpy
        assert True
    except ImportError:
        assert False, "Required modules not installed"
