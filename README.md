\# StudyIQ – Student Academic Performance Prediction



A machine learning web application that predicts student mathematics performance using academic and personal factors.



\## Project Overview



StudyIQ uses a Random Forest Regression model trained on the UCI Student Performance dataset. It predicts a student's final mathematics grade based on previous grades, study time, absences and past academic failures.



\## Features



\- Student academic performance prediction

\- Random Forest Regression

\- Flask-based prediction API

\- Interactive web dashboard

\- Model evaluation using MAE, RMSE and R²

\- Docker containerization

\- Docker Hub image publishing



\## Technology Stack



\- Python

\- Pandas

\- Scikit-learn

\- Flask

\- HTML and CSS

\- Docker

\- Git and GitHub



\## Model Performance



| Metric | Result |

|---|---:|

| MAE | 1.21 |

| RMSE | 2.03 |

| R² Score | 0.80 |



The target variable is G3, the final mathematics grade, on a scale of 0–20.



\## Project Structure



```text

ML-Docker-Project/

├── data/

│   └── student-mat.csv

├── models/

│   ├── student\_model.pkl

│   ├── metrics.json

│   └── default\_student.json

├── templates/

│   └── index.html

├── app.py

├── train\_model.py

├── requirements.txt

├── Dockerfile

├── .gitignore

└── README.md

```



\## Run Locally



Install dependencies:



```bash

pip install -r requirements.txt

```



Train the model:



```bash

python train\_model.py

```



Start the application:



```bash

python app.py

```



Open http://127.0.0.1:5000/



\## Run Using Docker



Pull the image:



```bash

docker pull rajmadane007/student-ml-model:latest

```



Run the container:



```bash

docker run -d -p 5000:5000 --name studyiq rajmadane007/student-ml-model:latest

```



Open http://127.0.0.1:5000/



\## Dataset



UCI Student Performance Dataset:



https://archive.ics.uci.edu/dataset/320/student+performance



\## Model Evaluation Note



The reported metrics are from a held-out test split using an 80/20 train-test split. G1 and G2 are included as input features.



\## Author



Rajwardhan Madane

