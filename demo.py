#!/usr/bin/env python3

#
# This is a demonstration for a COMP4121 project which involves using 
# machine learning techniques such as weighted approximate-rank pairwise
# and stochastic techniques to train a model to recommend movies for a user.
#
# The demonstration is implemented with the help of the LightFM library.
# https://github.com/lyst/lightfm
#
# Parameters for the training and the model can be changed by changing the
# constants.
#

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm import LightFM

#
# PARAMETERS
#

# number of threads utilised while training
NUM_THREADS = 4
# maximum number of samples to compare before stopping
CUTOFF = 10
# number of epochs to train the model
EPOCHS = 30
# parameter for precision_at_k
PRECISION_K = 10
# maximum number of epochs to graph accuracies for
MAX_EPOCHS = 100
# maximum step size to increment to
MAX_STEP = 1
# step size increment for the experiment
STEP_INCREMENT = 0.1
# maximum number of cutoffs to graph accuracies for
MAX_CUTOFF = 100
# number of cutoffs used for the speed test
SPEED_TEST_CUTOFF = 3
# maximum number of epochs to test to for the speed test
SPEED_TEST_EPOCH = 100


def train_model(data):
    # create the model
    model = LightFM(loss="warp", max_sampled=CUTOFF)
    # train the model on the training data
    model.fit(data["train"], epochs=EPOCHS, num_threads=NUM_THREADS)
    return model


def recommend_movies(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data["train"].shape

    # generate recommendations for each user
    for user_id in user_ids:
        # predict scores for the movies
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data["item_labels"][np.argsort(-scores)]

        print("User %s will probably like:" % user_id)

        # print the top 3 from the list
        for x in top_items[:3]:
            print("         %s" % x)

def measure_accuracies(model, data):
    print("\nMeasuring accuracies of the model...")

    # evaluate the precision@k metric
    training_precision = precision_at_k(model, data["train"], k=PRECISION_K).mean()
    test_precision = precision_at_k(model, data["test"], k=PRECISION_K).mean()

    # evaluate the AUROC metric
    training_auc = auc_score(model, data["train"]).mean()
    test_auc = auc_score(model, data["test"]).mean()

    # print them out
    print("Precision@k: training %.2f, test %.2f" % (training_precision, test_precision))
    print("AUC: training %.2f, test %.2f" % (training_auc, test_auc))

def graph_accuracies_epochs(data):
    print("\nTraining models with varying epochs from 0 to %d and recording their accuracies..." % MAX_EPOCHS)
    
    # array used to store the values at each epoch
    precisions = []
    aucs = []
    # setup the model
    test_model = LightFM(loss="warp")
    # iterate over the range of epochs and measure the accuracies
    for e in range(MAX_EPOCHS):
        current_trained = test_model.fit(data["train"], epochs=e)
        precisions.append(precision_at_k(current_trained, data["test"], k=PRECISION_K).mean())
        aucs.append(auc_score(current_trained, data["test"]).mean())
    print("Done!")

    x_axis = np.arange(MAX_EPOCHS)
    # plot the graph
    plot_accuracies(x_axis, precisions, aucs, "number of epochs", "magnitude of the accuracy metric", \
            ["precisions@10", "AUROC"], 2, "accuracies_epochs.png")

def graph_accuracies_step_size(data):
    print("\nTraining models with an epoch of 5 at different step sizes and recording their accuracies...")

    # array used to store the values at each step size
    precisions = []
    aucs = []

    # iterate over the range of step sizes and measure the accuracies
    for s in np.arange(0.1, MAX_STEP + STEP_INCREMENT, STEP_INCREMENT):
        test_model = LightFM(loss="warp", learning_rate=s)
        current_trained = test_model.fit(data["train"], epochs=5)
        precisions.append(precision_at_k(current_trained, data["test"], k=PRECISION_K).mean())
        aucs.append(auc_score(current_trained, data["test"]).mean())
    print("Done!")

    x_axis = np.arange(0.1, MAX_STEP + STEP_INCREMENT, STEP_INCREMENT)
    # plot the graph
    plot_accuracies(x_axis, precisions, aucs, "initial step size", "magnitude of the accuracy metric", \
            ["precisions@10", "AUROC"], 2, "accuracies_step_size.png")

def graph_accuracies_cutoff(data):
    print("\nTraining models with different sampling cutoffs and recording their accuracies...")

    # array used to store the values at each step size
    precisions = []
    aucs = []

    # iterate over the range of cutoffs and measure the accuracies
    for c in range(1,MAX_CUTOFF):
        test_model = LightFM(loss="warp", max_sampled=c)
        current_trained = test_model.fit(data["train"], epochs=5)
        precisions.append(precision_at_k(current_trained, data["test"], k=PRECISION_K).mean())
        aucs.append(auc_score(current_trained, data["test"]).mean())
    print("Done!")

    x_axis = range(1,MAX_CUTOFF)
    # plot the graph
    plot_accuracies(x_axis, precisions, aucs, "cutoff", "magnitude of the accuracy metric", \
            ["precisions@10", "AUROC"], 3, "accuracies_cutoff.png")

def graph_speeds_epochs(data):
    print("\nTraining models with a cutoff of %d and varying epochs from 0 to %d and recording their speed..." \
            % (SPEED_TEST_CUTOFF, SPEED_TEST_EPOCH))

    # array used to store the time taken for the training at different epochs
    times_taken = []
    # set up the model
    test_model = LightFM(loss="warp", max_sampled=SPEED_TEST_CUTOFF)

    # iterate over the range of epochs and measure the time needed to train
    for e in range(SPEED_TEST_EPOCH):
        start = time.time()
        current_trained = test_model.fit(data["train"], epochs=e)
        times_taken.append(time.time() - start)
    print("Done!")

    # plot the graphs
    print("Plotting the graph, the graph will be saved to speed.png...")
    x_axis = np.arange(SPEED_TEST_EPOCH)
    plt.figure(4)
    plt.plot(x_axis, times_taken)
    plt.xlabel("epochs")
    plt.ylabel("time taken to train the model (seconds)")
    plt.savefig("speed.png")
    print("Done!")

def plot_accuracies(x_axis, points_a, points_b, label_x, label_y, legend, figure, imagename):
    print("Plotting the graph, the graph will be saved to %s..." % imagename)
    # swap to a different figure
    plt.figure(figure)

    # plot the points
    plt.plot(x_axis, points_a)
    plt.plot(x_axis, points_b)

    # label the axes
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    # include a legend
    plt.legend(legend)

    # export the grpah to a image file
    plt.savefig(imagename)
    print("Done!")

if __name__ == "__main__":
    print("Downloading the data set now if it does not exist...")
    # download the data set if it does not exist and load it
    data = fetch_movielens(min_rating=4.0)
    print("Done!\n")
    # train the model
    trained_model = train_model(data)
    # recommend movies for users
    recommend_movies(trained_model, data, [3, 10, 50])

    # the functions below test and measure accuracies with variations to the parameters, uncomment
    # them out if you wish to run them

    #measure_accuracies(trained_model, data)
    #graph_accuracies_epochs(data)
    #graph_accuracies_step_size(data)
    #graph_accuracies_cutoff(data)
    #graph_speeds_epochs(data)
