# Visual Influence Networks
This GitHub repository contains code for the reconstruction and analysis of visual influence networks.
The details of the project is described in:

> Yoshida, K., Warren, W. H., & di Bernardo, M. (in prep). Visual influence networks in walking crowds.


## Repository structure

The repository is organized into 3 main folders:
1. [**`code/`**](code/)
    * This folder contains: 
        * Main scripts (placed directly in this folder): Scripts used to analyze the specific data in this project. These should be modified when used for other datasets. They are named in the order that was used in the analysis.
        * [**`utils/`**](code/utils/): A subfolder containing utility functions used across the main scripts. These functions can be reused in the analyses of other datasets.
            * The scripts to calculate the TDDC before reconstructing visual influence networks were based on the scripts used in [Lombard et al. (2020)](#references).
* [**`data/`**](data/)
    * This folder should contain experimental data. Other files produced in the analysis will be stored in this folder.
    * The structure of this folder may vary depending on the specific requirements of your experiment and analysis.
    * In this repository, a sample data (10 frames) from the Human 'Swarm' Experiment is included ([`data/csv/sayles_data_90pct_sample.csv`](data/sayles_data_90pct_sample.csv)).
* **`output/`** (not included in this repository)
    * This folder should be created to store the results generated from the code (PNG or SVG files).


## Setting up the environment

### 1. Clone the repository
* Clone this repository to your local machine and navigate to the project directory
    ```
    git clone https://github.com/Virtual-Environment-Navigation/visual-influence-networks
    cd [path_to_directory]/visual-influence-networks
    ```
### 2. Create a virtual environment
* Use a virtual environment to manage dependencies:
    ```
    python -m venv venv
    source venv/bin/activate  # for Windows: `venv\Scripts\activate.bat` 
    ```
### 3. Install dependencies
* Install the required Python packages using requirements.txt:
    ```
    pip install -r requirements.txt
    ```


## Usage 

### 1. Data Preparation

* Create the `data/` folder and include your own data. See [`data/csv/sayles_data_90pct_sample.csv`](data/sayles_data_90pct_sample.csv) for the sample data.

### 2. Running the Code

* The scripts in the `code/` folder are designed to analyze the specific experimental data used in this research. If you are using your own data, refer to these scripts as examples to understand how to apply the utility functions from `code/utils/`.
Modify the main scripts to match the structure of your data and run them to perform your analysis.

### 3. Storing Outputs

* Create an `output/` folder to store the results of your analysis. The scripts will need to be configured to save outputs in this directory.


## Contact

For any questions or issues, please contact Kei Yoshida at kei_yoshida@brown.edu.

## References
Yoshida, K., Warren, W. H., & di Bernardo, M. (in prep). Visual influence networks in walking crowds.

Lombardi, M., Warren, W. H., & di Bernardo, M. (2020). Nonverbal leadership emergence in walking groups. Scientific Reports, 10(1), 18948. https://doi.org/10.1038/s41598-020-75551-2