# About
This document describe about the data pipeline that I built.  
This data pipeline employs flask app and Google Spreadsheet to store input data from users.


## Procedures
### Google Spreadsheet related
1. Enable the API: https://console.cloud.google.com/flows/enableapi?apiid=sheets.googleapis.com
2. Configure the OAuth consent screen: https://console.cloud.google.com/apis/credentials/consent
3. Create a service account: https://console.cloud.google.com/apis/credentials
4. Click ADD KEY and select JSON and download key file
   ![image](https://github.com/user-attachments/assets/2913ccc1-eb46-42b8-8026-3e47e8309d68)

5. Rename a JSON file to "token.json"
6. Create a Google Spreadsheet named "Handwriting-recognition" for the data pipeline
7. Share the Google Spreadsheet with the service account
   Input "client_email" in token.json into the red rectangle.
   ![image](https://github.com/user-attachments/assets/2a58b3fe-92a0-42cb-b9ac-c76dbe44f355)

8. Create two sheets named "Digit" and "Character" respectively.

Reference:  
https://developers.google.com/sheets/api/quickstart/python

### Flask app related
1. Set the Google Spreadsheet file name that you made to sheet_title in the argument of get_google_sheet function in gss.py
	```
	def get_google_sheet(sheet_name: str, sheet_title: str = 'Handwriting-recognition'):
	 ```
2. Make sure to set proper sheet name in submit_feedback function in main.py
	```
	sheet_name = 'Digit' if correct_label.isdigit() else 'Character'
	 ```


## Retraining a model with the collected drawn data from users
Although the function that automatically retrain a model and update models for the recognition system has not been implemented, the sample code for retraining has been developed on [this file](https://github.com/taka-rl/handwriting-digit-character-recognition/blob/22-build-a-data-pipeline2/tf_practice/retrain_models.py).

