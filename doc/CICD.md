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

### famous tools for CI/CD
These are popular tools for CI/CD. [^5]


  | Name           | Description | 
  |----------------|---------------------------------|
  | GitHub Actions | a powerful workflow automation tool that enables developers to automate various tasks in their GitHub repositories. It allows to create custom CI/CD pipelines while automating other tasks such as issue management, notifications and deployments.                    |
  | GitLab CI         | a powerful CI/CD tool that is part of GitLab platfrom. It allows developers to define and manage CI/CD pipelines in thier GitLab repositories, serving a seamless experience for building, testing and deploying applications.                       | 
  | Jenkins        | a widely-used open-source automation server that helps developers automate various parts of the software development process such as building , testing, and deploying applications. It has a large and active community that contributes to its deelopment, making it a reliable and feature-rich CI/CD tool.                         | 

## CI/CD pipeline of this repository
### Overview of CI/CD pipeline
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

### CI process
The unit test (`python-app.yml`) used `pytest` is executed in CI process.  

### CD process
After the unit test, if the unit test passes, the deployment (`deploy.yml`) is executed.  
This is a brief explanation how the app is deployed through GitHub Actions. Although there are some processes such as Google Auth, they will be omitted here.   
1. It builds and containerize with Dockerfile via Google Cloud Build.   
```gcloud builds submit --tag gcr.io/$PROJECT_ID/handwriting-digit-character-recognition```  
2. Then, it push Docker image to Google Artifact Registry.  
   The --tag gcr.io/... part in the submit command:  
```--tag gcr.io/$PROJECT_ID/handwriting-digit-character-recognition```
3. Finally, it deply Image to Google Cloud Run
```
gcloud run deploy handwriting-recognition \
  --image gcr.io/$PROJECT_ID/handwriting-digit-character-recognition \
  --platform managed \
  --region=$REGION \
  --allow-unauthenticated
```
#### Docker
Docker is a software platform where you can build, test, and deploy applications. It packages software into standardized units called containers that have everything the software needs to run including libraries, system tools, code and runtime. [^6]  
A container allows developers to run the application quickly and reliably from one computing environment to another because it packages up code and all its dependencies. Although containers and virtual machines have similar resource isolation and allocation benefits, they work differently. Containers virtualize the operating system instead of hardware and are more portable and efficient. 
Containers are an abstraction at the app layer that packages code and dependencies together. Multiple containers can run on the same machine and share the OS kernel with other containers, each running as isolated processes in user space. On the other hands, virtual machines(VMs) are an abstraction of physical hardware turning one server into many servers. The hypervisor makes it possible to run multiple VMs on a single machine. 
[^7]  


Reference:  
[^1] Artifact Registry overview: https://cloud.google.com/artifact-registry/docs/overview  
[^2] Create a CI/CD Pipeline using GitHub Actions and Google Cloud: https://medium.com/google-cloud/create-a-ci-cd-pipeline-using-github-actions-and-google-cloud-9be20ff50e97   
[^3] What is Cloud Run: https://cloud.google.com/run/docs/overview/what-is-cloud-run  
[^4] What is CI/CD?: https://github.com/resources/articles/devops/ci-cd  
[^5] Continuous Integration Tools for DevOps – Jenkins vs. GitLab CI vs. GitHub Action: https://attractgroup.com/blog/continuous-integration-tools-for-devops-jenkins-vs-gitlab-ci-vs-github-action/  
[^6] What is Docker?: https://aws.amazon.com/docker/#:~:text=Docker%20is%20a%20software%20platform,tools%2C%20code%2C%20and%20runtime.  
[^7] What is a Container?: https://www.docker.com/resources/what-container/  

