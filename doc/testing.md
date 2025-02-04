# Testing
pytest is used for testing. All the test files are stored in the tests folder.  

## Folder structure
    │── tests
    │   ├── conftest.py               # Set up for testing
    │   ├── test_canvas.py            # Test for routes related canvas
 

## How to run testing on your local environment
1. run this command in the terminal: ```python -m pytest```  
   If you would like to see more details on the tests: ```python -m pytest -v```  

Here is the executed result of ```python -m pytest``` .


Reference:  
https://zenn.dev/onigiri_w2/articles/5e6cf4d3ba9ed5  
https://betterstack.com/community/guides/testing/pytest-guide/  
https://blog.teclado.com/pytest-for-beginners/  

## Introduce pytest to GitHub Actions
1. Create app.yaml file by clicking Python application  
![img.png](img.png)