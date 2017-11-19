# MovieWARP-SGD, WIP

This is an implementation of the Weighted Approximate-Rank Pairwise loss and Stochastic Gradient Descent for movie recommendation
as a demonstration for a COMP4121 project.

The demonstration is located in demo.py of the base directory.

The script requires the installation of the LightFM Python library. The Github page for the library can be found
here: https://github.com/lyst/lightfm. Assuming Python3.x.x, LightFM can be easily installed via `pip install lightfm`
or `pip3 install lightfm` if running `python` defaults to Python2.x.x on your system.

LightFM requires dependencies such as `numpy` but these are managed by `pip` if the dependencies are missing on your system.

In addition to the demonstration, several functions that measure the model's accuracy are included.

By default, the demonstration recommends movies for users 3, 10 and 50. This can be changed by modifying the list in the
arguments for `recommend_movies()` on line 203 to whatever you wish.
