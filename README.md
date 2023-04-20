#   Inventory Monitoring at Distribution Centers

In this project, we'll work on how to count the objects in bins. Our goal is to create a pipeline with AWS tools.

**Note**: This repository relates to AWS Machine Learning Engineer nanodegree provided by Udacity.

## Project Summary
Our work is organized into 5 categories:

1. Collect data from the main resource and organize into an S3 bucket.
2. Apply an exploratory data analysis(EDA) on the dataset using SageMaker Studio.
3. Design a model and tune its hyper parameters using SageMaker.
4. Train and evaluate the model using SageMaker.
5. Monitor the resource management of the model using SageMaker Debugger.


## Environment

We used an AWS SageMaker instance ```ml.t3.medium``` type with the following configurations:
- two virtual CPUs
- four GiB memory

And the main software pre-requisites for the project are:
- Python 3.8
- Pytorch: 1.12

## Initial setup

1. Clone the repository.
2. Run [sagemaker.ipynb](./starter/sagemaker.ipynb) cells in order and follow its instructions!

## Data

We use Amazon Image Bin Dataset.  The dataset contains 536,435 bin JPEG images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations.  We apply an EDA on the dataset to know it better. All the files and their metadata are organized in [list](./starter/data/list) and [metadatalist](./starter/data/metadatalist).

You can see a sample(with 5 objects in it) of the dataset in the following picture:

![data sample](./starter/images/sample.jpeg "a data sample with 5 objects")

We used [file_list.json](./starter/file_list.json), a subset which is a well-balanced representative subset of the whole dataset.

## Pipeline

After splitting our dataset into [train](./starter/data/train.json), [validation](./starter/data/valid.json) and [test](./starter/data/test.json) splits, we can store them into S3 bucket as shown below:

![data splits in s3](./starter/images/data_splits_in_s3.png "data splits in s3")

You can use [hpo.py](./starter/hpo.py) and [hpo_improved.py](./starter/hpo_improved.py) for hyperparameter tuning for benchmark and refined model, respectively. This point is similar for [train.py](./starter/train.py) and [train_improved.py](./starter/train_improved.py) for training and evaluation.

And finally, you can use [sagemaker.ipynb](./starter/sagemaker.ipynb) as an orchestrator for all the mentioned above scripts to create the pipeline in SageMaker.

## Profiler Reports
The reports of the SageMaker profiler is organized in [benchmark profiler reports](./starter/ProfilerReports/benchmark) and [improved profiler reports](./starter/ProfilerReports/improved) for benchmark and improved models, respectively.

## Technical Reports
You can read about the introduction and development phase of the project in [proposal.pdf](./starter/propsoal.pdf) and [report.pdf](./starter/report.pdf).