# costmap_neural_networks

This repository provides code to train a neural network to take in a map (in the form of a pgm) and predict the depth of a "wagon rut" in any location on the map. It also includes a script to compile this information into a prediction of wagon ruts on the entire map. (For more information about what a wagon rut is in this context, see https://github.com/OSUrobotics/wagon_rut_costmap). 

In order to train the neural network, add clean map images into the input_maps folder in either the CNN directory or regressional directory and wagon rut pgms into the corresponding output_maps folder. Wagon rut data can be generated from the data_generation branch of the repository linked above. The neural network can then make predictions on new maps using the predict_on_map.py script.

This code relies on Keras, and the underlying TensorFlow structure.

Much of the code in this repository is modified from https://github.com/sagekg/ebola-neural-net. 
