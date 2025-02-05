# About
This document describes about two ways of deploying the app on Google Cloud Platform(GCP)

1. the app to GCP with App Engine from your laptop(local environment)
2. the app to GCP with Cloud run through GitHub repository, leading to CD(Continuous Deployment)

## Preparation: 
Before starting, make sure you have a Google account for GCP.

### Folder structure for deployment

    │── app
    │   ├── routes               # Store Blueprint routes here
    │   │   ├── __init__.py
    │   │   ├── canvas.py        # Handles digit/character drawing
    │   │   ├── import_file.py   # Handles image uploads
    │   │   ├── index.py         # Home route
    │   ├── models.py            # Loads models at app startup
    │   ├── utilities.py         # Helper functions (image processing, validation)
    │   ├── gss.py               # Google Sheets API logic
    │   ├── static               # Static files (CSS, JS, models for recognition)
    │   ├── templates            # HTML templates
    │   └──  __init__.py         # Creates the Flask app and registers Blueprints
    ├── main.py                   # Flask app
    ├── app.yaml                  # Configuration file for App Engine deployment
    ├── requirements.txt          # Library version information file
    ├── gcloudignore              # Not mandatry but you can exclude unnecessary folders/files
    ├── Dockerfile                # Configuration file for Cloud run deployment    
    └── README.md
    
## 1. Deployment with App Engine  
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


## 2. Deployment with Cloud run
## Procedures
1. Create Dockerfile in your repository
   ```
   # Set the Python version
   FROM python:3.10  
	
   # Set the working directory
   WORKDIR /app
	
   # Copy the current directory contents into the container at /app
   COPY . ./

   # Install the required dependencies
   RUN pip install -r requirements.txt
	
   # Make port 8080 available to the world outside this container
   EXPOSE 8080
	
   # Run app.py when the container launches
   CMD python main.py
   ```

2. Link Your GitHub Repository with Google Cloud Run
   Click on “CONNECT REPO”.
   ![image](https://github.com/user-attachments/assets/d76dcb80-7f63-4bf5-b764-4cc2afd59c38)
3. Choose “Continuously deploy from a source repository” and then click “Set up with Cloud Build”.
   ![image](https://github.com/user-attachments/assets/8a9d2534-c1f3-492a-94e8-a5571338a796)

4. Authorize GitHub: Click on “Connect New Repository” and authorize Google Cloud to access your GitHub account.
   If you haven't authorized before, first you need to authenticate with GitHub.  
   If you haven't installed Google Cloud Build, you also need to install it.  
   ![image](https://github.com/user-attachments/assets/ff224cca-2e69-4ff5-9b6a-e3673804a915)

5. Select your repository
   ![image](https://github.com/user-attachments/assets/d3bdb3d6-8d20-48f3-91f7-9307df141740)

6. Select your branch and choose Dockerfile as a build type, and save it.
   ![image](https://github.com/user-attachments/assets/089397e7-b70a-4610-9b22-8a86d4d319fd)

7. CLick "CREATE".
   ![image](https://github.com/user-attachments/assets/34891c60-4db4-480c-a1bc-84caed28d077)

References:  
https://medium.com/codex/continuously-deploying-a-flask-app-from-a-github-repository-to-google-cloud-run-6f26226539b0

