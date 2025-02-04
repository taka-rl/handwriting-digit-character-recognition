from app.gss import load_test_image_data


def load_test_image(image_type: str):
    """Load a random test image from Google Spreadsheet."""
    sheet_name = 'Digit' if image_type == "Digit" else 'Character'
    test_image = load_test_image_data(sheet_name)
    return test_image


def test_canvas_digit(client):
    response = client.get('/canvas-digit')
    assert response.status_code == 200
    assert b"Draw a Digit" in response.data


def test_digit_recognition(client):
    """Test the digit recognition."""
    test_image = load_test_image('Digit')

    response = client.post("/submit-digit",
                           json={"image": test_image},
                           content_type="application/json"
                           )

    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["probabilities"], list)
    assert 0 <= data["prediction"] <= 9  # Ensure prediction is a valid digit
    assert len(data["probabilities"][0]) == 10


def test_canvas_character(client):
    response = client.get('/canvas-character')
    assert response.status_code == 200
    assert b"Draw a Character" in response.data


def test_character_recognition(client):
    """Test the Character recognition."""
    test_image = load_test_image('Character')

    response = client.post("/submit-character",
                           json={"image": test_image},
                           content_type="application/json"
                           )

    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "upper_probabilities" in data
    assert "lower_probabilities" in data
    assert isinstance(data["prediction"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["upper_probabilities"], list)
    assert isinstance(data["lower_probabilities"], list)
    assert data["prediction"].isalpha()
    assert len(data["upper_probabilities"]) == 26
    assert len(data["lower_probabilities"]) == 26
