# Multimedia Retrieval System
In order to install the system is sufficient to clone the repository to a local folder.

```
mkdir mmr_project
cd mmr_project
git clone git@github.com:Olu93/project_multimedia_retrieval.git
```

The project requires different packages to be installed, this can be done either using Anaconda or pip package managers.
From the project directory run the following.

Anaconda:
```
conda env create -f environment.yml
conda activate mmr_system_env
```
Pip:
```
pip install -r requirements
```


The databases needed for the system to run on the Princeton shape dataset are included in the repository. 
To initialise the correct paths to each just run:

```
python initialise.py
```
This should create defaults value for every parameter and prompt for further action or exit. 
If the aim is to run the system, then exit. Otherwise see “Setting new paths” below. 
Once the defaults are set, is possible to finally run the system itself. 
To do so, type:
```
python gui_maker.py
```
This should open the upload mesh windows, here is possible to select a mesh from the explorer or drop in onto the window. This will activate the query interface. In this second window is possible to set some parameters before querying.
The best evaluated parameters and distance function combinations are the default values. 

# Setting new paths

Once the initialise.py script is run, is also possible to set new values (for instance this could be setting a new database and compute normalisation and feature extraction).
To do so, when prompted with exit or continue, press 2 to continue and visualize the option available. 
Change database paths before running new pipelines, otherwise the system will pick up the existent ones and not perform the operation.
