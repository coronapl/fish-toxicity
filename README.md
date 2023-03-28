# Fish Toxicity Predictor Using Linear Regression
By Pablo Valencia A01700912

This model is a linear regression that uses the dataset "QSAR fish toxicity" by M. Cassotti, D. Ballabio, R. Todeschini,
V. Consonni, which was obtained from the UCI Machine Learning Repository.

## Problem

The toxicity of a chemical to fish is a complex function of its molecular properties. 

## Solution

This model was trained to predict the acute toxicity of a chemical to the fish Pimephales promelas by
considering 6 molecular descriptors: MLOGP, CIC0, GATS1i, NdssC, NdsCH, SM1_Dz(Z).

### Coefficients and Bias

- Coefficients: [[ 0.37479775], [ 1.25322334], [-0.69765601], [ 0.42521892], [ 0.06432804], [ 0.39739434]] 
- Bias: 2.123538854613838

### Model performance end Error

- Mean square error: 0.92
- Coefficient of determination: 0.60

### Conclusion

Considering the low number of samples of the dataset as well as the simplicity of the model, a coefficient of
determination of 0.60 is a good result for predicting the toxicity of a chemical to the fish Pimephales promelas.
This project has the purpose of understanding the capabilities as well as the limitations of using a simple linear
regression. 

## Usage

1. Clone the repository from GitHub (coronapl/fish-toxicity).
2. Create a virtual environment with the command `python3 -m venv venv`.
3. Activate the virtual environment with the command `source venv/bin/activate`.
4. Install all required packages with the command `pip3 install -r requirements.txt`.
5. Run the program with the command `python3 model.py`.
6. Enter the values for the 6 molecular descriptors (MLOGP, CIC0, GATS1i, NdssC, NdsCH, SM1_Dz(Z)) for the chemical you
want to predict the toxicity of.
