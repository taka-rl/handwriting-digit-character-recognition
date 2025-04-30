# About
This document describe two ways of deploying the app.
1. Deploy the app to App Engine on Google Cloud Platform(GCP)
2. Deploy the app from a GitHub Repository to Google Cloud Run.

## Preparation: 
Before starting, make sure you have a Google account for GCP.

## Deployment with App Engine
### Folder structure for the deployment
	Top directory
    ├── static                    # Necessary folder/file for the app
    │   ├── css                   # CSS file
    │   └── js                    # JavaScript file
    ├── templates                 # HTML files
    ├── main.py                   # Flask app
    ├── app.yaml                  # Configuration file
    ├── requirements.txt          # Library version information file
    └── gcloudignore              # Not mandatry 

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


## Deployment with Google Cloud Run
### Overview of the deployment

Developer -> Commit -> GitHub -> GitHub Actions (Unit test) -> GitHub Actions (deployment) -> Artifact Registry -> Google Cloud Run


### Folder structure for the deployment
	Handwriting-digit-character-recognition
    ├── .github                  # Necessary folder/file for the app
    │   └── workflows            # Store Blueprint routes here
    │       └── deploy.yml       # Store Blueprint routes here
    │── app
    │   ├── credentials          # token.json file will be copied here
    │   ├── routes               # Store Blueprint routes here
    │   ├── static               # 
    │   ├── templates            # 
    ├── Dockerfile               # Docker file to containerize the app
    ├── main.py                  # Flask app
    ├── requirements.txt         # Library version information file


### Procedures
1. Set `Environment Secrets` and `Environment Variables` in repository
   - Environment Secrets
       - GCP_PROJECT_ID: Your Google Cloud project ID  
       If you don't know about it and if you have the Google Cloud SDK installed, run:  
       ```
        gcloud config list
        ```
   		This is your project ID.
        ```
       	project = handwriting-recognition-systems
        ```

       - GCP_SA_KEY_B64: Base64-encoded service account json file  
                1. Download Json style service account file    
                2. Open cmd and change the directory where the downloaded service account file exists  
                  3. Run this command on cmd  
               '''  
                 base64 -w 0 path/to/your-service-account.json > service-account-key-base64.txt  
                 '''  
               4. Copy the entire base64 string from service-account-key-base64.txt, then, define it in `Environment Secrets`  
        - TOKEN_JSON: Necessary information for Google Spreadsheet
   ![image](https://github.com/user-attachments/assets/1315b0ea-564f-4b61-a6ba-af4cb4b01101)  

- Environment Variables
	- GCP_REGION: Region such as europe-central2
![image](https://github.com/user-attachments/assets/8b3f2c9d-4d06-41b7-b950-b961c94ce3f1)  


2. Create deploy.yml

	```
	name: Deploy to Cloud Run
	
	on:
	  workflow_run:
		workflows: [ "Test Workflow" ]  # must match the name in python-app.yml
		types:
		  - completed
	
	permissions:
	  contents: read
	
	env:
	  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
	  REGION: ${{ vars.GCP_REGION }}
	  CREDENTIALS_PATH: ${{ github.workspace }}/gcp-key.json
	
	jobs:
	  deploy:
		if: ${{ github.event.workflow_run.conclusion == 'success' }}  # Only run if unit test passed
		runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Check env vars
        run: |
          echo "Environment variables configured:"
          echo PROJECT_ID="$PROJECT_ID"
          echo REGION="$REGION"

      - name: Write service account key file
        run: |
          echo "${{ secrets.GCP_SA_KEY_B64 }}" | base64 --decode > $CREDENTIALS_PATH

      - name: Authenticate with Google Cloud
        run: |
          gcloud auth activate-service-account $SVC_EMAIL --key-file=$CREDENTIALS_PATH
          gcloud config set project $PROJECT_ID
          gcloud config set run/region $REGION

      - name: Verify gcloud auth
        run: |
          gcloud auth list
          gcloud config list

      - name: Create token.json file
        run: echo "$TOKEN_JSON" > app/credentials/token.json
        env:
          TOKEN_JSON: ${{ secrets.TOKEN_JSON }}

      - name: Build Docker image and push to GCR
        run: |
          gcloud builds submit --tag gcr.io/$PROJECT_ID/handwriting-digit-character-recognition

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy handwriting-recognition \
            --image gcr.io/$PROJECT_ID/handwriting-digit-character-recognition \
            --platform managed \
            --region=$REGION \
            --allow-unauthenticated


	```

3. Create Dockerfile

	```
	FROM python:3.10

	# Set the working directory
	WORKDIR /app
	
	# Copy the current directory contents into the container at /app
	COPY . .
	
	# Install the required dependencies
	RUN pip install -r requirements.txt
	
	# Make port 8080 available to the world outside this container
	EXPOSE 8080
	
	# Run main.py when the container launches
	CMD python main.py
	```

