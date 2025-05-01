# CI/CD pipeline
This documents describes the following topics.
- CI/CD basics
- CI/CD pipeline built for this repository.

## CI/CD basics
### What is CI/CD?
CI/CD stands for Continuous Integration and Continuous Delivery (or Continuous Deployment). 
Continuous Integration(CI): Automatically builds, tests, and integrate code updates within a shared repository.
Continuous Delivery(CD): Automatically delivers code updates to production-ready environments for approval.
Continuous Deployment(CD): Automatically deploys code updates to customers directly.
[^4]

### Why CI/CD?
The overall goals are as follows: 
1. It enables you to enhance the speed and quality of software delivery.
2. It allows a much higher frequency of software changes.
3. It reduces the amount of time for  the development cycles with associated faster feedback loops.
4. It decreases human error and manual toil.
[^2]

## CI/CD pipeline of this repository
### Overview of the deployment
The following image illustrate the overall CI/CD pipeline. 
![image](https://github.com/user-attachments/assets/3e6a6977-c4ca-492d-a8cd-b447e5f42c69)


The CICD workflow is presented as below.
1. Engineers/Developers creates a branch to develop a feature or fix a bug.
2. They commit and push their changes. 
3. The pipeline, which is Unit test through GitHub Actions runs.
4. If the tests pass, the next GitHub Actions for deployment is executed.

* I need to add one step to approve the changes and then the GitHub Actions for deployment is executed. *

  | Name               | Description                                                                                                                                                                                                                                                                                                                 | 
  |--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | Artifact Registry  | a place where you can store and manage your packages and Docker container images. [^1]                                                                                                                                                                                                                                      |
  | Google Cloud Build | a serverless CI/CD tool provided by Google. It can run build workflows such as testing, building, vulnerability scanning, and deployment. Also It can push build artefacts like container images to Google CLoud Artifact Registry. Moreover, it can deploy to various Google Cloud services such as Google Cloud Run. [^2] |
  | Google Cloud Run   | a managed compute platform where you can run containers directly on top of Google's scalable infrastructure.  [^3]                                                                                                                                                                                                          |

### CI

The unit test (`python-app.yml`) used `pytest` is executed in CI process.  

### CD
After the unit test, if the unit test passes, the deployment (`deploy.yml`) is executed.

Reference:  
[^1] Artifact Registry overview: https://cloud.google.com/artifact-registry/docs/overview  
[^2] Create a CI/CD Pipeline using GitHub Actions and Google Cloud: https://medium.com/google-cloud/create-a-ci-cd-pipeline-using-github-actions-and-google-cloud-9be20ff50e97   
[^3] What is Cloud Run: https://cloud.google.com/run/docs/overview/what-is-cloud-run  
[^4] What is CI/CD?: https://github.com/resources/articles/devops/ci-cd

