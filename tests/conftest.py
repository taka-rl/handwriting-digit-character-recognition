import pytest
from app import create_app


@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    app = create_app()

    with app.test_client() as client:
        yield client
