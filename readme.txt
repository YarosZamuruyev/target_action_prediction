the work contains 2 pipelines loan_pipeline and main
the first one processes the data and trains the model.
Model sampling and hyperparameter search were done in ipnyb file in google colab.

There are data and models folders.
the first one contains two datasets which I have reformatted from csv to feather,
to save RAM and computation time.
Also in the data folder there is a file "5to_trials.json" in it I randomly selected 5 lines from 2 classes.
If you send the contents of the json file as a post-request in the web service postman, you will get the results of predicates.

the models folder contains the trained random forest model,
as well as a dictionary that stores conversion rates for each unique value of each variable,
that was used in training the model.

Replacing string values of categorical variables by their conversion rates has saved memory usage and also reduced the average time to train a model to 5-6 minutes.

in update 2.1 I replaced the random forest model with the catboost model.
because the hyper-parameters of the random forest led to possible overtraining of the model on test data and reduced the model's ability to generalise.
