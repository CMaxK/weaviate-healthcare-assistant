import pytest
from helpers.simple_test import add_numbers

def test_add_numbers():
    assert add_numbers(2, 3) == 5, "Should be 5"
    assert add_numbers(-1, 1) == 0, "Should be 0"
    assert add_numbers(0, 0) == 0, "Should be 0"

if __name__ == "__main__":
    pytest.main()
