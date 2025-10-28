# Titanic Survival Prediction (R & Python)

This project builds logistic regression models in R and Python to predict passenger survival on the Titanic dataset.  
Both implementations are fully containerized using Docker, making them reproducible and easy to run on any machine.

---

## Repository Structure
```
TITANIC_PREDICT/
│
├── src/
│   ├── data/                         # <– Place the Titanic CSV files here
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   │
│   ├── python_code/                  # Python implementation
│   │   └── app.py
│   │
│   └── r_code/                       # R implementation
│       └── app.R
│
├── Dockerfile_python                 # Dockerfile for Python
├── Dockerfile_r                      # Dockerfile for R
├── install_packages.R                # Installs required R packages
├── requirements.txt                  # Python dependencies
├── output_log_python.txt             # Output log from Python (created after running container)
├── output_log_r.txt                  # Output log from R (created after running container)
└── README.md                         # Project documentation
```
## Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/titanic_predict.git
cd titanic_predict
```

### Download the data
Download data from https://www.kaggle.com/competitions/titanic/data
* train.csv                      -training data
* test.csv                       -testing features data
* gender_submission.csv          -testing survival values   

Place all three inside:

titanic_predict/src/data/

Your folder should look like this:
```
src/data/
├── train.csv
├── test.csv
└── gender_submission.csv
```

## (Optional) Create Your Own Virtual Environment
#### Create and activate a virtual environment
python3 -m venv titanic_env

source titanic_env/bin/activate

## Run the model and get predictions!

### Run the Models Using Docker

cd titanic_predict

- For python:

* Build the Docker image

```bash
docker build -f src/python_code/Dockerfile -t titanic_python src/
```

* Run the container

```bash
docker run -v $(pwd):/app titanic_python
```

- For R:

* Build the Docker image

```bash
docker build -f src/python_code/Dockerfile -t titanic_r src/
```

* Run the container

```bash
docker run -v $(pwd):/app titanic_r
```

* Output

	•	A file named output_log_r.txt will be created in the project root (titanic_predict/output_log_r.txt) containing model logs and accuracy results.

    •	A file named output_log_python.txt will be created in the root folder (titanic_predict/output_log_python.txt).

### Check the results

```
output_log_r.txt
output_log_python.txt
```