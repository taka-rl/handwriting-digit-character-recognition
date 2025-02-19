# About
This document describe about the data pipeline that I built.  
This data pipeline employs flask app and Google Spreadsheet to store input data from users.


## Procedures of storing drawn data from users
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
3. Users try the app and get some feedback from them.  


## Retraining a model with the collected drawn data from users
At the moment, it isn't easy to collect a lot of data from users, for the demo retraining, I temporarily extract some data from the MNIST data, ratating them with some degrees, storing them in the Google Spreadsheet.  
The following scripts are used for the demo retraining and currently, the retraining is implemented for only digit.  
- [dummy_data.py](https://github.com/taka-rl/handwriting-digit-character-recognition/blob/main/app/dummy_data.py): Generate rotated data from the MNIST data.
- [retrain_model.py](https://github.com/taka-rl/handwriting-digit-character-recognition/blob/main/app/retrain_model.py): Retrain a model with the generated data from dummy_data.py

### Retrain process in GitHub Actions
1. Run retrain.yml (Temporarily manual trigger but it's possible to run regularly)
2. Execute dummy_data.py
   - Generate and store some rotated data from the MNIST dataset to the Google Spreadsheet
     | Example: before rotation                      | Example: after 30 degree rotation             |
     | --------------------------------------------- | --------------------------------------------- |
     | ![image](https://github.com/user-attachments/assets/4ebbff16-ff55-4043-9c2d-bc5995a39082) | ![image](https://github.com/user-attachments/assets/881418a6-abfe-4f07-954b-b86ea5973341) |

3. Execute retrain_model.py
   - Collect the generated data from the Google Spreadsheet
   - Retrain a model
   - Compare the accuracy and loss between the original and the retrained models
   - Save the retrained model


