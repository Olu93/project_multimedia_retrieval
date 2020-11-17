# Experiments
In this folder there are several scripts all needed for us to generate data.
The scripts with *plot* as prefix are to plot different kind of data retrieved from the shapes,
these scripts range from evaluation scripts (*plot_eval_barchart.py*) to shapes distribution in dataset
(*plot_bar_instance_per_class.py*).

The scripts with the *experiment* prefix are scripts needed to experiment with data in order to achieve best 
performance for the system. These include scripts to randomly search the parameter space and evaluate the best
results as well as data analysis and classification.  

# How to run the experiments
Call pattern for these experiments is module based!

```
python -m experiments.[EXPERIMENT_FILE_WO_FILE_EXTENSION]
```