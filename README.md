# Use wavelet transform combined with machine learning algorithm to predict protein-protein interactions
Pinsan Xu, Jun Luo, Tongyi Dou*
* Corresponding author:douty@dlut.edu.cn

Aim at predict protein-protein interactions using protein's primary sequence.

## Requirements
Experimental pipeline is implemented in Python 3.
You have to import some Python 3 packages before you can run the script, packages are as follows:
* numpy
* scipy
* scikit-learn
* mlxtend
* pywavelets

## Project structure
The *data* flod contains three subdirectories:*inter_no_act_location*, *orig* and *test_result*.
The *inter_no_act_location* folder contains several species' interact proteins list and the list of non-interacting proteins that we construct.
The *orig* folder contains some of the more primitive files.
The *test_result* stores the test results and provides them to draw_picture.py and draw_picture2.py.

## Running the experiment
If all the dependencies are satisfied, you can run main experimental with:
> python experiment_WT.py
>
We have also implemented some of the methods mentioned in others' studies for predicting protein interactions.
The predictive performance of these predictor can be seen by:
> python experiment_comparison.py
>
After running experiment_WT.py, experiment_comparison.py, we provide a histograms to reflect the performance gap between the comparative scenario and our method :
> python draw_picture.py
>
We also provide the demo of the impact of different number of wavelets used on predictor performance. You can see the demo by:
> python draw_picture2.py
>
That's it.
