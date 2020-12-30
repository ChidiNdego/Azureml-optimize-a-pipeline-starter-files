# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree: the first of three projects.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

Click [here](https://docs.microsoft.com/en-us/azure/machine-learning/) for more information on Azure ML.

## Summary
This dataset contains data from a direct marketing campaign through phone calls of a Portuguese banking institution. The classification goal is to predict if a client will subscribe to one of the bank's product, bank term deposit, represented by the variable, y. 
	
At pristine state, the dataset contains 20 different predictor variable and 32950 rows representing different customers with 3,692 subscribing to the bank's product and 29,258 negative classes.

Click [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing) for detailed information about the dataset.

Click [here](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) to download the dataset.

The best model optimizer was the AutoML with an accuracy of 0.9162. The Hyperdrive experiment faired really well but a little lower at 0.9091 model accuracy.

## Scikit-learn Pipeline
To use Hyperdrive, one must have a custom-coded ML model. Otherwise, hyperdrive won't know what model to optimize the parameters for. SKLearn's standard logistic regression algorithm was used for this binary classification problem. Azure ML SDK and hyperdrive package helped to choose optimal adjustable parameters for model training.

The pipeline architecture:
1.  Data collection/creation
2.  Data preparation
3.  Training configuration
    *   Hyperparameter sampling
    *   Primary metric specification
    *   Termination policy
    *   Hyperdrive configuration
4.  Training validation
5.  Saving/Registering the model

NOTE: A training script and a notebook was created for the execution of the processes above.

Below is a brief explanation of each step:

### Data collection/creation
The csv data is collected from the url link specified earlier and created as a tabular dataset using the TabularDatasetFactory class.

### Data preparation
This involved the cleaning and the splitting of the dataset. A clean_data function in the training script dropped rows with empty values and performed a one hot encoding on the categorical columns.
Datasets are also split into train and test sets. This splitting of a dataset is helpful to validate our model. Splitting ratio is often dependent on the size of the dataset. For this experiment, the training data to test data split ratio is 70:30.

### Training configuration
Choosing optimal hyperparameter values for model training can be difficult, and usually involves a great deal of trial and error. With Azure Machine Learning, you can leverage cloud-scale experiments to tune hyperparameters.
Automate efficient hyperparameter tuning by using Azure Machine Learning [HyperDrive package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?preserve-view=true&view=azure-ml-py).
Learn how to complete the steps required to tune hyperparameters with the [Azure Machine Learning SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?preserve-view=true&view=azure-ml-py):

*   Define the parameter search space
*   Specify a primary metric to optimize
*   Specify early termination policy for low-performing runs
*   Allocate resources
*   Launch an experiment with the defined configuration
*   Visualize the training runs
*   Select the best configuration for your model

**Hyperparameter sampling**
Hyperparameters are adjustable parameters that let you control the model training process.
Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance. The process is typically computationally expensive and manual.

Click [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) for more information on Azure ML hyperparameter tuning.

The two hyperparamters used in this experiment are **C** and **max_iter**. **C** is the Inverse Regularization Strength which applies a penalty to stop increasing the magnitude of parameter values in order to reduce overfitting. **max_iter** is the maximum iteration to converge for the SKLearn Logistic Regression algorithm.

We have used [Random Parameter Sampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?preserve-view=true&view=azure-ml-py) to sample over a set of continuous values. In random sampling, hyperparameter values are randomly selected from the defined search space. A benefit of this sampling technique is that it supports discrete and continuous hyperparameters.
It also supports early termination of low-performance runs. An initial search can be done with this technique, then search space refined to improve results.

The parameter search space used for **C** is [0.01, 0.1, 1, 10, 50] and for **max_iter** is [25, 50, 250, 500]. The optimal values were 50 and 250 respectively.

**Primary metric specification**
The [Primary Metric](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.primarymetricgoal?preserve-view=true&view=azure-ml-py) is used to optimize the hyperparamter tuning. Each training run is evaluated for the primary metric. 

We have chosen **accuracy** as the primary metric. **MAXIMIZE** is the preferred primary metric goal.

**Termination policy**
Early termination policies are applied to HyperDrive runs. A run is cancelled when the criteria of a specified policy are met. The [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py) was used. 

It is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. This helps to improves computational efficiency.

For this experiment, evaluation_interval=2, slack_factor=0.1, and delay_evaluation=3. This configration means that the policy would be applied to every even number iteration of the pipeline greater than 3 and if 1.1*value of the benchmark metric for current iteration is smaller than the best metric value so far, the run will be cancelled.

**Hyperdrive configuration**
This configuration defines a HyperDrive run. It includes the information above about hyperparameter space sampling, termination policy, primary metric, resume from configuration, estimator, and the compute target to execute the experiment runs on.

Click [here](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py) for more information on hyperdrive configuration.

### Training validation
With the train and test data available as well as necessary configuration completed, model training and validation can commence by submitting an experiment.

### Saving/Registering the model
The trained model is saved. This is important if we want to deploy our model or store for some later use.


## AutoML
AutoML is the process of automating the time consuming, iterative tasks of machine learning model development. It is applicable in regression, classification, and time-series forecasting problems.

In our experiment, VotingEnsemble proved to be the best model based on the accuracy metric; with a score of `0.9162062200877541`.

[Voting Ensemble](https://machinelearningmastery.com/voting-ensembles-with-python/), is an ensemble machine learning model that combines the predictions from multiple other models. Here, it represents a collection of autoML iterations brought together to form an ensemble that implements soft voting.

The VotingEnsemble model from our experiment consist of five algorithms. Below shows each algorithm and some parameters: `learning_rate`, `gamma`, `n_estimators`, and `reg_lambda`. 

Refer to the attached jupyter notebook for more detailed information.

| ALGORITHM | WEIGHT | LEARNING RATE | GAMMA | NUMBER OF ESTIMATORS | LAMBDA |
| --------- | --------- | --------- | --------- | --------- | --------- |
| xgboostclassifier with maxabsscaler | 0.09090909090909091 | 0.1 | 0 | 100 | 1 |
| lightgbmclassifier with maxabsscaler | 0.18181818181818182 | 0.1 |  | 100 | 0 |
| xgboostclassifier with sparsenormalizer | 0.36363636363636365 | 0.1 | 5 | 25 | 0.104167 |
| xgboostclassifier with sparsenormalizer | 0.09090909090909091 | 0.1 | 0 | 100 | 0.9375 |
| xgboostclassifier with sparsenormalizer | 0.2727272727272727 | 0.1 | 0 | 25 | 0.729167 |

## Pipeline comparison
The model generated by AutoML had an accuracy slighlty higher than the HyperDrive model. `0.9162062200877541` for autoML and `0.9095599393019727` for HyperDrive.

Both models have remarkably different architectures. The HyperDrive architecture was restricted to a specified custom-coded ML model. In this case, SKLearn's Logisic Regression. However, AutoML had an unconfined architecture. It accesses wide variety of algorithms and selects the best performing model.

This further explains why there's a difference in the result. In some cases a selected model for an hyperdrive experiment may not be best suited for that problem. Hence, whatever algorithm is specified is what the hyperdrive makes do with. As a result, hyperdrive is at a disadvantage in this regard.

## Future work
**Areas of Improvements for the HyperDrive Model**
1.  Parameter sampling can be carried out more effectively. Increase in RandomParameterSampling or start with a wide range of values for each parameter, then refine the search space.
2.  Apply other types of parameter sampling including the Bayesian Parameter Sampling. Bayesian sampling tries to intelligently pick the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric.
3.  Target variable is seen to be highly imbalanced. This can invariably lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class. Hence the need to over-sample the minority class using a technique such as SMOTE. [Synthetic Minority Oversampling Technique (SMOTE)](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) is a data augmentation technique used to duplicate the examples in the minority group of a classifica problem.
4.  Try a different primary metric. Sometimes accuracy alone doesn't represent true picture of the model's performance. Recall or precision are more specific metrics in related classification problems.
5.  Tweak some other hyperdrive confirguration parameters including max total runs, to try a lot more combinations.

**Areas of Improvements for the AutoML Model**
1.  Address class imbalance to prevent model bias. 
2.  Increase experiment timeout duration. This would allow for more model experimentation, but at expense of cost.
3.  Try a different primary metric. Sometimes accuracy alone doesn't represent true picture of the model's performance. Recall or precision are more specific metrics in related classification problems.
4.  Tweak some other AutoML confirguration parameters including number of cross validation to reduce model bias.