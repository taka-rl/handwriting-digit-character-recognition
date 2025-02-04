from datetime import datetime
import gspread
import os


CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credentials', 'token.json')


def get_google_sheet(sheet_name: str, sheet_title: str = 'Handwriting-recognition'):
    """Get the Google Sheet instance."""
    # Load service account key file
    gspread_client = gspread.service_account(filename=CREDENTIALS_PATH)

    # Open Google Spreadsheet
    sh = gspread_client.open(sheet_title)

    # Return the sheet of the Spreadsheet
    return sh.worksheet(sheet_name)


def save_to_sheet(sheet_name: str, image_data: str, predicted_label: str, confidence: float, correct_label: str):
    """
    Save the digit or character data and the prediction data into Google Spreadsheet and return the row ID.

    Parameters:
        sheet_name: sheet names in the targeted Google Spreadsheet
        image_data: the drawn digit or character by users
        predicted_label: the predicted label
        confidence: the percentage of the predicted label
        correct_label: the correct label

    """
    sheet = get_google_sheet(sheet_name)

    id_num = len(sheet.col_values(1))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Append data to the spreadsheet
    sheet.append_row([id_num, image_data, predicted_label, confidence, correct_label, timestamp])
    print("Data saved successfully!")


def load_test_image_data(sheet_name: str):
    """
    Load a random image data from Google Spreadsheet for unit testing

    Returns:
        data on B2 cell (2,2) as string
    """
    sheet = get_google_sheet(sheet_name)
    return sheet.cell(2, 2).value
