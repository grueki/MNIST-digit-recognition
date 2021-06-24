COMPSYS 302 - Python AI Project - Handwriting Recogniser
IK Tech (Team 03)
Isabelle Johns
Kimsong Lor

OUTLINE:
This program is used to train and test machine learning models using the MNIST database for handwritten digits.
There are three models currently implemented, they are: LeNet-5, Linear Regression, and Multilayer Perceptron.
The user has the choice to select the model and amount of iterations for training. 
The dataset for the handwritten digits can be viewed on a view tab with flitering functions to narrow down
specfic types of images i.e. all the number 2's from the testing set. 
The user is able to test the model with their own drawing and a predicted value with be given alongside a probability chart.
A history tab is available for viewing all the informations about previously evaluated user drawn inputs.


INSTALLATIONS:
The development environment was set up using Miniconda3.
pip was used as the primary package installer.
Recommended IDE => Visual Studio Code
Environment => Python=3.8
Additional Packages => PyQt5, requests, numpy, pyqtgraph, pillow, torch and torchvision


OPERATING INSTRUCTIONS:
Upon start up, if the MNIST dataset has not already been downloaded into the project folder then the program
will download it automatically providing that the device running the program has internet connection.
Note: the program will not pop up until the MNIST dataset has been installed
Simliarly, if a model has not already been saved into in the project folder then a model is required to be trained using the
selection options on the train tab before the user drawn input prediction on the canvas page can be used.
The view and history tab can be access at any time.
The view tab contains selection options for the flitering the dataset images, simplying apply your fliters by selecting
your desired options and then pressing the 'fliter' button.
The history tab will not show any listing until a user drawn input has been evaluated. Once evaluated, all relevant data will be stored
in the history list and can be selected with use of the 'view' buttton to provide a pop up of the user drawing 
and the probability chart assoicated with it.
A separate file, model_metrics.py has been included in order to obtain the metrics of the currently loaded model. When run,
it will print the saved model's recall, precision and F1 score. 


CREDITS AND ACKNOWLEDGEMENTS:
IK Tech would like to thank the following:
The supervisor and lecturer Ho Seok Ahn for providing the teaching, support and guidelines to make this project successful. 
The tutors Yuanyuan Hu and Jong Yoon Lim for the teaching and support throughout the project.
Eric Stuart and Amir Hambuch for providing hundreds of handwritten digits for the metrics database.


CONTACT DETAILS:
Isabelle Johns, ijoh785@aucklanduni.ac.nz
Kimsong Lor, klor742@aucklanduni.ac.nz

