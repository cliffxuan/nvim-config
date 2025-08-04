def test_function(mock_mixtures):
    """Test function docstring."""
    from app.utils.shirt import get_function

    # Mock mixtures data
    mock_get_mixtures.return_value = {
        "mixture1": {"name": "mixture1", "platform": "dusk"},
        "mixture2": {"name": "mixture2", "platform": "dusk"},
    }

    # Mock shirts data
    assert True