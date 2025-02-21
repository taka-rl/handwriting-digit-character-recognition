from app.gss import load_test_image_data, fetch_data_from_sheets, get_last_row, delete_row


def load_test_image(image_type: str):
    """Load a random test image from Google Spreadsheet."""
    sheet_name = 'Digit' if image_type == "Digit" else 'Character'
    test_image = load_test_image_data(sheet_name)
    return test_image


def submit_prediction(client, target: str, test_image):
    if target not in ['digit', 'character']:
        raise ValueError("target must be either 'digit' or 'character'")

    response = client.post(f"/submit-{target}",
                           json={"image": test_image},
                           content_type="application/json"
                           )
    assert response.status_code == 200, "Prediction API request failed"
    return response.get_json()


def submit_feedback(client, test_image, predicted_label, confidence, correct_label):
    # Feedback
    response = client.post("/submit-feedback",
                           json={"image": test_image,
                                 "predicted_label": predicted_label,
                                 "confidence": str(confidence),
                                 "correct_label": str(correct_label)},
                           content_type="application/json"
                           )
    assert response.status_code == 200, "Feedback API request failed"
    return response.get_json()


def validate_google_sheets(sheet_name, test_image, predicted_label, confidence, correct_label):
    sheet_data = fetch_data_from_sheets(sheet_name)
    if not sheet_data:
        raise ValueError("Error: No data found in the Google Spreadsheet")

    last_row = get_last_row(sheet_name) - 2  # -2 because row 1 contains column headers
    if last_row < 0 or last_row >= len(sheet_data):
        raise IndexError(f"Invalid index {last_row}: sheet_data only has {len(sheet_data)} rows.")

    assert sheet_data[last_row]['Data'] == test_image
    assert sheet_data[last_row]['Predicted_Label'] == predicted_label
    assert sheet_data[last_row]['Confidence_Score'] == confidence
    assert sheet_data[last_row]['User_Corrected_Label'] == correct_label

    return last_row


def test_canvas_digit(client):
    response = client.get('/canvas-digit')
    assert response.status_code == 200
    assert b"Draw a Digit" in response.data


def test_digit_recognition(client):
    """Test the digit recognition."""
    test_image = load_test_image('Digit')

    response = submit_prediction(client, 'digit', test_image)

    assert "prediction" in response
    assert "confidence" in response
    assert "probabilities" in response
    assert isinstance(response["prediction"], int)
    assert isinstance(response["confidence"], float)
    assert isinstance(response["probabilities"], list)
    assert 0 <= response["prediction"] <= 9  # Ensure prediction is a valid digit
    assert len(response["probabilities"][0]) == 10


def test_canvas_character(client):
    response = client.get('/canvas-character')
    assert response.status_code == 200
    assert b"Draw a Character" in response.data


def test_character_recognition(client):
    """Test the Character recognition."""
    test_image = load_test_image('Character')

    response = submit_prediction(client, 'character', test_image)

    assert "prediction" in response
    assert "confidence" in response
    assert "upper_probabilities" in response
    assert "lower_probabilities" in response
    assert isinstance(response["prediction"], str)
    assert isinstance(response["confidence"], float)
    assert isinstance(response["upper_probabilities"], list)
    assert isinstance(response["lower_probabilities"], list)
    assert response["prediction"].isalpha()
    assert len(response["upper_probabilities"]) == 26
    assert len(response["lower_probabilities"]) == 26


def test_digit_feedback(client):
    """Test a feedback mechanism for Digit recognition."""
    sheet_name = 'Digit'
    test_image = load_test_image(sheet_name)

    # Prediction
    submit_response = submit_prediction(client, 'digit', test_image)

    # Check if the prediction is success or not
    assert "prediction" in submit_response
    assert "confidence" in submit_response
    assert "probabilities" in submit_response

    predicted_label = submit_response['prediction']
    confidence = submit_response['confidence']
    # correct_label = submit_digit_response['correct_label']  # None as no user input is sent.
    # Assume that the prediction is correct
    correct_label = predicted_label

    # Feedback
    digit_feedback_response = submit_feedback(client, test_image, predicted_label, confidence, correct_label)
    # Check if the feedback is success
    assert digit_feedback_response.get("success") is True, "Feedback response did not return success"

    # Check if correct label is stored
    sheet_data = fetch_data_from_sheets(sheet_name)

    # Validate Google Sheets Data
    last_row = validate_google_sheets(sheet_name, test_image, predicted_label, confidence, correct_label)

    # Delete the data for the test
    print("Delete data used for this test")
    delete_row(sheet_name, len(sheet_data) + 1)

    # Check if the data is deleted or not
    assert (get_last_row(sheet_name) - 1) == last_row, "The data wasn't deleted."


def test_character_feedback(client):
    """Test a feedback mechanism for Character recognition."""
    sheet_name = 'Character'
    test_image = load_test_image(sheet_name)

    # Prediction
    submit_response = submit_prediction(client, 'character', test_image)

    # Check if the prediction is success or not
    assert "prediction" in submit_response
    assert "confidence" in submit_response
    assert "upper_probabilities" in submit_response
    assert "lower_probabilities" in submit_response

    predicted_label = submit_response['prediction']
    confidence = submit_response['confidence']
    # correct_label = submit_digit_response['correct_label']  # None as no user input is sent.
    # Assume that the prediction is correct
    correct_label = predicted_label

    # Feedback
    digit_feedback_response = submit_feedback(client, test_image, predicted_label, confidence, correct_label)
    # Check if the feedback is success
    assert digit_feedback_response.get("success") is True, "Feedback response did not return success"

    # Check if correct label is stored
    sheet_data = fetch_data_from_sheets(sheet_name)

    # Validate Google Sheets Data
    last_row = validate_google_sheets(sheet_name, test_image, predicted_label, confidence, correct_label)

    # Delete the data for the test
    print("Delete data used for this test")
    delete_row(sheet_name, len(sheet_data) + 1)

    # Check if the data is deleted or not
    assert (get_last_row(sheet_name) - 1) == last_row, "The data wasn't deleted."
