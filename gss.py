from datetime import datetime
import gspread


def get_google_sheet(sheet_name: str, sheet_title: str = 'Handwriting-recognition'):
    """Get the Google Sheet instance."""

    # Load service account key file
    gspread_client = gspread.service_account(filename="./token.json")

    # Open Google Spreadsheet
    sh = gspread_client.open(sheet_title)

    # Return the sheet of the Spreadsheet
    return sh.worksheet(sheet_name)


def save_to_sheet(sheet_name: str, image_data: str, predicted_label: int, confidence: float):
    """
    Save the digit or character data and the prediction data into Google Spreadsheet and return the row ID.

    Parameters:
        sheet_name: sheet names in the targeted Google Spreadsheet
        image_data: the drawn digit or character by users
        predicted_label: the predicted label
        confidence: the percentage of the predicted label

    Returns:
        id_num: the row ID stored in the Google Spreadsheet
    """
    sheet = get_google_sheet(sheet_name)

    id_num = get_last_row(sheet)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Append data to the spreadsheet
    sheet.append_row([id_num, image_data, predicted_label, confidence, timestamp])
    print("Data saved successfully!")
    return id_num


def get_last_row(sheet) -> int:
    return len(sheet.col_values(1))
