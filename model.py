"""
Pablo Valencia A01700912
@coronapl
Fish Toxicity Predictor
March 27, 2023
"""
import numpy as np
import pandas as pd
import linearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

FEATURES = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP']
TARGET = ['LC50']


def load_dataset(filepath, columns):
    """
    Loads a dataset from a CSV file located at filepath, and returns a pandas
    DataFrame with the given columns.

    :param filepath: Path of the CSV file to load.
    :param columns: List of columns to use for the DataFrame.
    :return: DataFrame containing the loaded data.
    """
    df = pd.read_csv(filepath, sep=';')
    df.columns = columns
    return df


def create_model(x_train, y_train):
    """
    Trains a linear regression model on the given training data and returns the
    trained model.

    :param x_train: Numpy array containing the input features for the
    training data.
    :param y_train: Numpy array of shape containing the target values for the
    training data.
    :return: Trained linear regression model.
    """
    model = linearRegression.LinearRegression(learning_rate=0.01,
                                              num_iters=10000)
    model.fit(x_train, y_train)
    return model


def get_model_performance(test_data, predictions):
    """
    Calculates the performance metrics of a model on a given test data.

    :param test_data: Numpy array containing the true target values for the
    test data.
    :param predictions: Numpy array containing the predicted target values for
    the test data.
    :return: Tuple containing the mean squared error and coefficient of
    determination performance metrics.
    """
    mse = mean_squared_error(test_data, predictions)
    coefficient_determination = r2_score(test_data, predictions)
    return mse, coefficient_determination


def print_model_predictions(y_test, y_predictions):
    """
    Plots a scatter plot of the model's predicted values against the true
    target values.

    :param y_test: Numpy array of shape (n_samples,) containing the true target
    values for the test data.
    :param y_predictions: Numpy array containing the predicted target values
    for the test data.
    :return: None
    """
    plt.scatter(y_test, y_predictions, color='blue', label='Predicted')
    plt.scatter(y_test, y_test, color='red', label='Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    plt.legend()
    plt.show()


def get_user_data():
    """
    Reads user input from STDIN to make a prediction.

    :return: List with the values entered by the user to make a prediction.
    """
    print('\nUser input: \n')

    CIC0 = float(input('Wiener Connectivity Index (0.667 - 5.926): '))
    SM1_Dz = float(input('3D-MoRSE descriptor representing electrostatic '
                         'potential (0.000 - 2.171): '))
    GATS1i = float(input('Geary Autocorrelation of lag 1 weighted by Atomic'
                         ' masses (0.396 - 2.920): '))
    NdsCH = float(input('Number of attached carbon and hydrogen atoms'
                        ' (0.000 - 4.000): '))
    NdssC = float(input('Number of attached carbonyl, carboxyl, and '
                        'thiocarboxyl groups (0.000 - 6.000): '))
    MLOGP = float(input('Molecular Logarithm of Partition '
                        'Coefficient (-2.884 - 6.515): '))

    return [CIC0, SM1_Dz, GATS1i, NdsCH, NdssC, MLOGP]


def train_model():
    """
    Trains a machine learning model to predict fish toxicity based on chemical
    properties.

    :return: Trained machine learning model.
    """
    print('-- FISH TOXICITY PREDICTOR --\n')
    print('Training the model...')

    # Load dataset and split it into training and testing sets
    df = load_dataset('data.csv', FEATURES + TARGET)
    df_x = df[FEATURES]
    df_y = df[TARGET[0]]

    x_train, x_test = df_x[:850].to_numpy(), df_x[850:].to_numpy()
    y_train, y_test = df_y[:850].to_numpy().reshape((850, 1)), \
        df_y[850:].to_numpy().reshape((57, 1))

    fs_model = create_model(x_train, y_train)
    y_predictions = fs_model.predict(x_test)
    mse, coefficient_determination = get_model_performance(y_test,
                                                           y_predictions)

    # Print model performance and metrics
    fs_model.plot()
    print_model_predictions(y_test, y_predictions)
    print('-----------------------------------\n')
    print('Model performance:\n')
    print('Coefficients: \n', fs_model.get_params())
    print('Mean squared error: %.2f' % mse)
    print('Coefficient of determination: %.2f' % coefficient_determination)

    return fs_model


if __name__ == '__main__':
    model = train_model()
    stop = False

    while not stop:
        user_data = np.array(get_user_data())

        print('The predicted LC50 is %.2f\n' %
              model.predict(user_data))

        stop_input = input('Do you want to make another prediction? yes/no\n')
        stop = True if stop_input == 'no' else False
