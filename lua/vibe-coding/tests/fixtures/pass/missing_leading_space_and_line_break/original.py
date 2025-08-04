@patch("app.utils.shirt.get_mixtures")
def test_get_platform_shirts_success(mock_get_mixtures, mock_get_mixture_shirts):
    """Test successful retrieval of platform shirts."""
    from app.utils.shirt import get_platform_shirts

    # Mock mixtures data
    mock_get_mixtures.return_value = {
        "mixture1": {"name": "mixture1", "platform": "dusk"},
        "mixture2": {"name": "mixture2", "platform": "dusk"},
    }

    # Mock shirts data for each mixture
    mock_get_mixture_shirts.side_effect = [
        [{"shirt": "shirt1", "path": "/path1"}, {"shirt": "shirt2", "path": "/path2"}],
        [{"shirt": "shirt3", "path": "/path3"}],
    ]

    result = get_platform_shirts("dusk")


@patch("app.utils.shirt.get_mixture_shirts")
@patch("app.utils.shirt.get_mixtures")
def test_get_platform_shirts_with_mixture_error(
    mock_get_mixtures, mock_get_mixture_shirts, mock_logger
):
    """Test get_platform_shirts when one mixture fails but others succeed."""
    from app.utils.shirt import get_platform_shirts

    # Mock mixtures data
    mock_get_mixtures.return_value = {
        "mixture1": {"name": "mixture1", "platform": "daier"},
        "mixture2": {"name": "mixture2", "platform": "daier"},
        "mixture3": {"name": "mixture3", "platform": "daier"},
    }

    # Mock shirts data - mixture2 will fail, others succeed
    mock_get_mixture_shirts.side_effect = [
        [{"shirt": "shirt1", "path": "/path1"}],  # mixture1 success
        Exception("Connection timeout"),  # mixture2 fails
        [{"shirt": "shirt3", "path": "/path3"}],  # mixture3 success
    ]