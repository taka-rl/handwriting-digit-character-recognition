# About
This document describe how I have deployed the app to App Engine on Google Cloud Platform(GCP)

## Preparation: 
Before starting, make sure you have a Google account for GCP.

### Folder structure for deployment as an example
	Top directory
    ├── static                    # Necessary folder/file for the app
    │   ├── css                   # CSS file
    │   └── js                    # JavaScript file
    ├── templates                 # HTML files
    ├── main.py                   # Flask app
    ├── app.yaml                  # Configuration file
    ├── requirements.txt          # Library version information file
    └── gcloudignore              # Not mandatry but 


## Procedures
### GCP related 
1. Create a project on GCP

2. Enable the API
3. Install the Google Cloud CLI (If you haven't installed it)

4. To initialize the gcloud CLI, run the following command: 
	```
    gcloud init
    ```
5. Run the following command to enable App Engine and create the application resources.
 	```
 	gcloud app create
    ```

### Flask app related
1. Prepare the following files:
	app.yaml
	```
	 runtime: python310
     entrypoint: gunicorn -b :$PORT main:app
	 ```
	requirements.txt:
	```
     Flask==3.0.3
     tensorflow==2.10.0 # only deployment 2.12 didn't work
     Pillow==9.5.0
     numpy==1.23.0
     # ↓ for deploy ↓
     gunicorn==23.0.0
    ```
	.gcloudignore: If necessary, you can add to ignore unnecessary folders/files
	```
     # Ignore tf_practice folder except for model files
     tf_practice/characters/
     tf_practice/digits/
     tf_practice/src/ 
     tf_practice/doc/
    ```
 
2. Move to the directory that you would like to deploy
3. Execute this command on : gcloud app deploy
4. Move to the app: gcloud app browse


References:
https://cloud.google.com/appengine/docs/standard/python3/building-app/writing-web-service
https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service
https://cloud.google.com/appengine/docs/standard/reference/app-yaml?tab=python#handlers_element


