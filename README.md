# Titanic_flow
Self-contained implementation, of training flow, to produce a classifier that powers a FastAPI prediction endpoint, running on Docker containers.

Requirements
-
We will assume that you have a working installation of python3 (3.12.3, with venv), docker and make.

In case you don´t, please refer to:


https://wiki.python.org/moin/BeginnersGuide/Download

https://docs.docker.com/engine/install/ubuntu/

https://www.geeksforgeeks.org/installation-guide/how-to-install-make-on-ubuntu/

-------------------------------------
Dataset
-

The classifier tackles the problem of predicting if a Titanic passenger survived or not.

You can find the dataset, and its corresponding competition, here: 

https://www.kaggle.com/c/titanic/

Since it is a small dataset, it is included inside this repository, in the dataset folder.

-------------------------------------
Concise installation and execution instructions:
-
1) Clone the repository and cd into it.
2) Run 'make setup' in terminal, which will create a Python virtual environment with all necessary dependencies.
3) Run 'make train' in terminal, which will use the virtual environment to train the model.
4) Run 'make fastapi' in terminal, which will use Docker to create a container to run a FastAPI application.

Those are the core steps. 

You can then access the API at 127.0.0.1:8000/ and there's a prediction endpoint at /predict.

You can also inspect the API docs at 127.0.0.1/docs, where you can easily test the prediction endpoint.

-------------------------------------

A small foreword:
-
ML systems need to tackle, at least, 4 core aspects: Reliability, Scalability, Maintainability and Adaptability. 

"Developing an ML system is an iterative and, in most cases, never-ending process. 
Once a system is put into production, it'll need to be continually monitored and updated."

Oftentimes, debugging an ML system can feel very ethereal and involved, which is why it is good to reduce uncertainty as much as possible, something that containerization tackles very well, and that has naturally seeped into the ML world.

The repository has 3 main files:
-
data_preprocessing.py
-  Holds all the functions necessary to format the dataset to be used for model training and prediction.
-  Houses a text model to be used for embedding extraction. It seemed appropriate to include it here, as all the other files import from this module. It really is the glue that holds it all together.

main.py
-  Processes the samples, using the functions from data_preprocessing.
-  Separates the training data into a training set and a validation set, since a test set is provided in the original dataset.
-  Defines an XGBoost classifier and performs a simple grid search to search for hyperparameters, using mcc as a scoring metric.
-  Trains a final classifier, using the best hyperparameters found in the search, to train over the complete training set.
-  Tests the final classifier over the original training set and the test set.
-  Exports the model, along with its accompanying artifacts, to disk, so it can be used later by the FastAPI app.
-  Does all of this while working under mlflow, which provides experiment monitoring, simplifying the task of keeping track of parameters and metrics.
-  Also, provides some logs regarding the training steps and program execution information.

app.py
- Defines a simple FastAPI application that loads the trained model and provides a prediction endpoint.
- Comes with a pydantic object to check for proper inputs.
- Takes care of proper variable initialization using a lifespan routine.
- Writes a simple log line, to showcase the functionality. (Can be captured using docker logs and docker logs -f commands)

There's also the following files:
-
makefile
-  Contains simple commands to run the steps necessary to install, train and run the application.

requirements.txt
-  Contains the dependencies used to run the training experiments.

requirements_container.txt
-  Contains the dependencies used to run the FastAPI app.
-  Is smaller and different in order to reduce the load on the FastAPI container.

dockerfile
-  Has simple definitions that allow us to create a container to run the FastAPI app.

test_app.py
- Includes a small number of simple tests.
- Tests the FastAPI application, particularly the prediction endpoint.

data_drift.py
-  Shows how to inspect data drift using Evidently AI.
-  Can be run using the 'python data_drift.py' command, from the virtualenv's python installation (./titanic_env/bin/python data_drift.py)

mlflow
-
We can run the UI for mlflow by running the command 'mlflow ui' in our virtual environment. We can also use 'make mlflow', which neatly calls it for us.
Here, we can inspect a history of our experiments, and we can also find the model artifacts (saved model weights and other relevant objects) linked to each experiment.
It provides a simple, yet useful, way to keep track of varying model parameters, and even datasets themselves, if we would be so bold.

It allows us to check system metrics, model metrics, access the artifacts and the datasets used, among other things.

Even though it was not included in this example (for now), it can be used in tandem with dvc to version and store data objects. Since dvc is used in conjunction with git, we're already halfway there.

Profiling
-
The code was, also, profiled using cProfile and memory_profiler, which is where the memory_profile.txt file came from.

You could also use tools like py-spy, to (try to) profile already running processes, without incurring too much overhead. Note that memory_profiler can be run by adding a simple decorator (@profile) to the target function and then calling it to run the script.

Another good tool, specially for FastAPI request profiling is pyinstrument.

Pod
-
We could, also, create a small pod of containers to include HTTPS (that's one container that takes care of that) and monitoring (another container that scans the containers of the pod, for example, cAdvisor). It is a very flexible environment that scales really well.

CI/CD
-
We can integrate, for example, Github actions, to trigger a container build upon code change, and we can, then, deploy the image.

-----------------------------------------
A consistent modern ML stack formula usually contains:
-  An orchestrator.
-  An experiment tracker and model registry.
-  Model monitoring tools.
-  Data drifting tools.
-  A deployment environment.

We have tackled some of these components in this repository, while noting that all employed tools integrate seamlessly with the missing pieces of the stack, so this can be a good starting point to build upon.

Our choice of orchestrator will be heavily influenced by our cloud environment of deployment, so I will leave that as an exercise for the reader, for now.

-----------------------------------------

Regarding the classifier, the data and the metrics
-
Since this is, after all, a Machine Learning endeavour, and not only an exercise in DevOps, we should comment upon the experiment.

In matters of binary classification with imbalanced classes, I've always considered the Matthews Correlation Coefficient a good, albeit somewhat obscure, metric, that's very informative for being a single float, which is why it is used to perform the scoring during the grid search.

Now, if you read the main.py script, and were also paying attention, you might note that a training dataset split was performed, but that it was not, strictly, a necessity, because, since we already have a well-defined test set, we could simply perform cross-validation while performing our grid search, which is precisely what happened in the script. Still, the split was made to showcase the functionality, and to introduce the decision of using the StratifiedShuffleSplit, in lieu of the more common train_test_split, because we have imbalanced classes.

Another quick point to touch upon, is that the test set is never used to construct the model, nor choose its hyper-parameters, so we are not in presence of data leakage (at least that we introduced through this step).

It should also be noted that a lot of the flexibility, when it comes to the median and scaler parameters, of the preprocessing routines exists because great care was taken to avoid data leakage from deriving statistics from the whole set (and, specially, obviously, the testing set). Which is why (and also because it needs to be this way for classification to work properly) these objects are only computed over the training samples.

XGBoost is, usually, a very good classifier (that's also very lightweight, since we want to serve it) but it also responds to another very interesting detail when it comes to Machine Learning, we want to give the task to the system that is just about capable to solve it. What do I mean by this? Every model has a level of expressiveness, or capacity, which can be interpreted as its ability to approximate the probability distribution function of the data.

If the model has high enough capacity, it is more likely to overfit to the training data. This is why the heuristic of starting with the simplest models possible is so justified: less chance of over-fitting and easier to deploy. Still, it is clear that this model is overfit over the training set, and a weaker model should be searched (I suspect the features are surprisingly informative).

Regarding the features of this problem, it is usually tackled by hand-crafting features that combine the information of the available fields, a step that was also followed here, by one-hot encoding many of the columns. Also, in order to avoid losing whatever information provided in the text fields (Name, Ticket and Cabin), these fields were combined into a single text field, upon which embeddings are computed, which is why the feature vectors end up having so many fields.

Feature importance, as noted by mlflow, shows that the two most important features are Age and Fare, followed closely by one of our constructed text features and being male or not. Since we can appreciate (feature_importance.png) several other text-constructed features (all embedding features have a number as name) before Class_3, it is a dead-giveaway of those text features having relevant information.

Much has been said about one of the main aspects of this problem being deciding if someone is male or female and if they are rich or poor, but there seems to be a lot more information about the passenger's class hidden in variables other than Pclass.

A simple experiment to try to increase model generalization (and not just increasing metrics, which should never, strictly, be our goal) would be to drop the less representative (in terms of importance weight) features.

The confusion matrix shows us that our biggest hurdle is reporting survivors as having died, we could tackle this with an ensemble strategy, by using a second (or sequence) of classifiers to better determine this particular situation. We could also tackle this through data augmentation.

All in all, it performs reasonably over the test set, as indicated by the Kaggle scoreboard (leaving cheaters aside), which might be a funny reason to justify it, but is, nevertheless, a very real one.



