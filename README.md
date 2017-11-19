# MovieWARP-SGD

This is an implementation of the Weighted Approximate-Rank Pairwise loss and Stochastic Gradient Descent for a movie recommendation
system as a demonstration for a COMP4121 project.

The demonstration is located in demo.py of the base directory.

The script requires the installation of the LightFM Python library. The Github page for the library can be found
here: https://github.com/lyst/lightfm. Assuming Python3.x.x, LightFM can be easily installed via `pip install lightfm`
or `pip3 install lightfm` if running `python` defaults to Python2.x.x on your system.

LightFM requires dependencies such as `numpy` but these are managed by `pip` if the dependencies are missing on your system.

In addition to the demonstration, several functions that measure the model's accuracy are included. These functions, however,
require the Matplotlib Python library in order to plot graphs from the data gathered in the functions. The installation
guide for Matplotlib can be found here: https://matplotlib.org/users/installing.html.

By default, the demonstration recommends movies for users 3, 10 and 50. This can be changed by modifying the list in the
arguments for `recommend_movies()` on line 203 to whatever you wish.
