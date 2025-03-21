# About
This document describes about the data pipeline used in this app.  
This data pipeline employs flask app and Google Spreadsheet to store input data from users, providing a feedback with the correct label, retraining a model with the collected data (Ideally).
Regarding the retraining a model, however, it isn't easy to collect a lot of data at the moment. Therefore, dummy data that is generated from the MNIST dataset, rotated with some degrees and used instead of the user data during the `retrain.yml` process.
(In the future, hopefully, the real user data can be used.)

## Big picture of the data pipeline
This data pipeline consists of the following steps:
1. **Setting up Google Spreadsheet and Flask App**  
   - Enable Google Sheets API and configure authentication.
   - Store user-drawn data in Google Sheets.

2. **Retraining a Model**
   - Generate some data from the MNIST dataset and rotated them with some degrees. (In the future, hopefully, the real user data can be used.)
   - Train the model on the generated data.  
   - Save the retrained model.
   - Commit the updated model to a GitHub branch. (This has not been implemented yet. Future plan.)
   - Deploy the retrained model for future use. (This has not been implemented yet. Future plan.)

## Procedures of storing drawn data from users
### Google Spreadsheet Setup
To store user input data, follow these steps:
1. **Enable the API:** [Enable API](https://console.cloud.google.com/flows/enableapi?apiid=sheets.googleapis.com)
2. **Configure the OAuth consent screen:** [OAuth Consent](https://console.cloud.google.com/apis/credentials/consent)
3. **Create a service account:** [Create Service Account](https://console.cloud.google.com/apis/credentials)
4. **Click ADD KEY and select JSON and download key file**
   ![image](https://github.com/user-attachments/assets/2913ccc1-eb46-42b8-8026-3e47e8309d68)

5. **Rename a JSON file to "token.json"**
6. **Create a Google Spreadsheet named "Handwriting-recognition" for the data pipeline**
7. **Share the Google Spreadsheet with the service account**  
   - Input "client_email" in token.json into the red rectangle.
   ![image](https://github.com/user-attachments/assets/0ab60698-2a9d-4a3b-9d01-03987ea11255)


8. **Create two sheets named "Digit" and "Character" respectively.**

Reference:  
https://developers.google.com/sheets/api/quickstart/python

### Flask App Integration
Once the Google Spreadsheet is set up, connect it to your Flask app:
1. **Set the Spreadsheet Name in `gss.py`:**
   - Open `gss.py` and update the `get_google_sheet` function:
     ```
       def get_google_sheet(sheet_name: str, sheet_title: str = 'Handwriting-recognition'):
     ```
2. **Ensure Correct Sheet Selection in `main.py`:**
   - Verify that submit_feedback correctly assigns the right sheet:
	```
	sheet_name = 'Digit' if correct_label.isdigit() else 'Character'
	```
3. **Users try the app and get some feedback from them.** 


## Retraining a model with the collected drawn data from users
At the moment, it isn't easy to collect a lot of data from users, for the demo retraining, I temporarily extract some data from the MNIST data, ratating them with some degrees, storing them in the Google Spreadsheet.  
The following scripts are used for the demo retraining and currently, the retraining is implemented for only digit.  
[retrain.yml](https://github.com/taka-rl/handwriting-digit-character-recognition/blob/main/.github/workflows/retrain.yml) executes the following scripts.  
- [dummy_data.py](https://github.com/taka-rl/handwriting-digit-character-recognition/blob/main/app/dummy_data.py): Generate rotated data from the MNIST data.
- [retrain_model.py](https://github.com/taka-rl/handwriting-digit-character-recognition/blob/main/app/retrain_model.py): Retrain a model with the generated data from dummy_data.py

### Retrain process in GitHub Actions
1. Run [retrain.yml](https://github.com/taka-rl/handwriting-digit-character-recognition/blob/main/.github/workflows/retrain.yml) (Temporarily manual trigger but it's possible to run regularly)  
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
   - Automatically commit the retrained model 
