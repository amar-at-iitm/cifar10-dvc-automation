# cifar10-dvc-automation: Continuous Integration with DVC for CIFAR-10 Classification
This project demonstrated the power of DVC in CI/CD pipelines for machine learning. Using DVC, I efficiently managed dataset versioning, pipeline automation, and experiment tracking, leading to reproducible and well-documented ML experiments.


## Project Structure
The project is organized as follows:
```
.
├── config.yaml              # Configuration file with hyperparameters
├── dvc.yaml                 # DVC pipeline configuration
├── requirements.txt         #  Requirements
├── src/                     # Source code for task 3
│   ├── pull_data.py         # Task 1: Pulls data from dvc remote
│   ├── data_preparation.py  # Task 2: Prepares data for train_model.py
│   ├── train_model.py       # Task 3: Model training 
│   ├── evaluate.py          # Task 3: Model evaluation 
│   └── loggers.py           # Logging utilities
├── task1.py                 # Task 1: Download data and save images
├── create_partition.py      # Task 2: Data partitioning
├── task2.sh                 # Task 2: Automatically completes the task 2
├── task4.sh                 # Task 4: Experiment pipeline runs
├── task5.py                 # Task 5: Identifying hard-to-learn images
```
### Requirements
Install the required dependencies using:
```
pip install -r requirements.txt
```

## Task 1: Downloading and Storing Data
- Downloads the entire CIFAR-10 dataset.
- Organize it into 10 subfolders, each representing a class.
-  Images are stored in PNG format.

#### How to Run
```
python task1.py
```

## Task 2: Creating Partitions
- Creates three random partitions (20,000 images each) from the dataset.
- Labels these partitions as v1, v2, and v3.
- Stores these partitions in DVC.

#### How to Run
Use the following commands. It will initialize `DVC`, ask for the local remote location for the first time, and then push all three partitions to the local remote location.
```
bash chmod +x task2.sh
bash ./task2.py
```

## Task 3: Model Training and Evaluation
#### Defines a DVC pipeline with the following stages:
- Pull Data from DVC
- Data Preparation (train, val, test splits)
- Model Training & Hyperparameter Tuning
- Evaluate Model Performance (Confusion Matrix & Accuracy Report)

#### How to Run
```
dvc repro
```

## Task 4: Experiment Pipeline Runs (task4.sh)
This script runs multiple experiment pipelines with random seeds `42,100,2025` and dataset combinations `v1, v2, v3, v1+v2, v1+v2+v3`. After each run, the results are pushed to remote storage, and the script displays all experiments at the end using `dvc exp show`.
#### How to Run
Use the following commands.
```
dvc repro
bash chmod +x task4.sh
bash ./task4.py
```

## Task 5: Identifying Hard-to-Learn Images (task5.py)
This task identifies images that the model consistently misclassifies across different runs. The results are stored in a CSV file and provide insights into the most challenging samples for the model.

#### How to Run
To execute the pipeline and experiments, use the following commands:
```
PYTHONPATH=src python task5.py
```

Experiment results and hard-to-learn images are stored under the `results/` directory.

