"""
Market Value Calculator
=======================

written from March 20, 2025 to april 21, 2025
created by JP

Overview:
This script implements the logic for calculating the market value of cars based on historical data.
It uses optimization techniques to fit a model to the data and estimate the market value of a car.

Features:
- Fit an exponential decay model to car price data.
- Optimize model parameters using various loss functions.
- Adjust market value based on car variant price.
- Visualize market value trends.
- Uses robust regression to make the algorithm resistant to outliers.

Modules:
- SciPy: Provides optimization functions.
- Pandas: Handles data manipulation and processing.
- Matplotlib: Visualizes market value trends.
- NumPy: Handles numerical operations.

Classes:
- MarketValueCalculator: A class to calculate the market value of cars.
    Attributes:
    - df_train (pd.DataFrame): Training data for the model.
    - distance_driven (int): Maximum distance driven for filtering data.
    - loss_method (str): Loss function for optimization (e.g., RMSE, MAE, SMRE).
    - min_method (str): Optimization method (e.g., L-BFGS-B).
    - verbose (bool): Whether to print detailed logs and visualize results.
    - fig (matplotlib.figure.Figure): Figure object for visualizations.
    - car_brand (str): Brand of the car being evaluated.
    - car_model (str): Model of the car being evaluated.
    - car_year (int): Manufacturing year of the car being evaluated.
    - car_variant_price (float): Price of the specific car variant.
    - v (np.array{float}): Model parameters for optimization.
    - v_norm (np.array{float}): Scaled model parameters for optimization.
    - v_bounds (list{Tuple{float, float}}): Bounds for the model parameters during optimization.
    - scaling (float): Scaling factor for normalization.
    - x_trains (list{np.array{float}}): List of training data for car years old.
    - y_trains (list{np.array{float}}): List of normalized training prices.
    - learning_rate (float): Learning rate for optimization.
    - log_len_trains (np.array{float}): Logarithm of the number of training data points.
    - results (OptimizeResult): Results of the optimization process.
    - market_value (float): Calculated market value of the car.
    - df_market_value (pd.DataFrame): DataFrame containing market value trends.
    - options (dict): Options for the optimization process.
    - response (dict): Summary of the market value calculation process.

    Methods:
    - initialize_optimization(): Prepares data and parameters for optimization.
    - calculate_loss(v, x_train, y_train, log_len_train): Calculates the loss for a given set of parameters.
    - calculate_cost(v): Calculates the total cost for optimization.
    - optimize_parameters(): Optimizes model parameters.
    - evaluate_market_value(): Evaluates the market value of a car.
    - visualize_result(): Visualizes the market value trends.
    - calculate_market_value(car_brand, car_model, car_year, car_variant_price): Main method to calculate the market value of a car.
"""

from scipy.optimize import minimize
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import numpy as np
import re

def model(v, x_train):
    """
    Exponential decay model for car price.

    Parameters:
    - v (np.array{float}): Model parameters [y1, a, c].
    - x_train (np.array{float}): Input data (car years old).

    Returns:
    - y (np.array{float}): Predicted car prices.
    """
    y1, a, c = v

    y = np.add(np.multiply(a, np.exp(np.multiply(c, x_train))), y1)  # y = a*e^(cx) + y1 ~~~ equation for exponent
    return y

class MarketValueCalculator:
    """
    A class to calculate the market value of cars.

    Attributes:
    - df_train (pd.DataFrame): Training data for the model.
    - distance_driven (int): Maximum distance driven for filtering data.
    - loss_method (str): Loss function for optimization (e.g., RMSE, MAE, SMRE).
    - min_method (str): Optimization method (e.g., L-BFGS-B).
    - verbose (bool): Whether to print detailed logs and visualize results.
    - ax (matplotlib.figure.Axis): Axis object for visualizations.
    - fig (matplotlib.figure.Figure): Figure object for visualizations.
    - car_brand (str): Brand of the car being evaluated.
    - car_model (str): Model of the car being evaluated.
    - car_year (int): Manufacturing year of the car being evaluated.
    - car_variant_price (float): Price of the specific car variant.
    - v (np.array{float}): Model parameters for optimization.
    - v_norm (np.array{float}): Scaled model parameters for optimization.
    - v_bounds (list{Tuple{float, float}}): Bounds for the model parameters during optimization.
    - scaling (float): Scaling factor for normalization.
    - x_trains (list{np.array{float}}): List of training data for car years old.
    - y_trains (list{np.array{float}}): List of normalized training prices.
    - learning_rate (float): Learning rate for optimization.
    - log_len_trains (np.array{float}): Logarithm of the number of training data points.
    - results (OptimizeResult): Results of the optimization process.
    - market_value (float): Calculated market value of the car.
    - df_market_value (pd.DataFrame): DataFrame containing market value trends.
    - options (dict): Options for the optimization process.
    - response (dict): Summary of the market value calculation process.

    Methods:
    - initialize_optimization(): Prepares data and parameters for optimization.
    - calculate_loss(v, x_train, y_train, log_len_train): Calculates the loss for a given set of parameters.
    - calculate_cost(v): Calculates the total cost for optimization.
    - optimize_parameters(): Optimizes model parameters.
    - evaluate_market_value(): Evaluates the market value of a car.
    - visualize_result(): Visualizes the market value trends.
    - calculate_market_value(car_brand, car_model, car_year, car_variant_price): Main method to calculate the market value of a car.
    """
    def __init__(self, df_train,
                 distance_driven = 100,
                 loss_method = "MAE",
                 min_method = "L-BFGS-B",
                 truncate_upper = 0.95,
                 truncate_lower = 0.05,
                 verbose = False):
        """
        Initialize the MarketValueCalculator with training data and parameters.

        Parameters:
        - df_train (pd.DataFrame): Training data for the model.
        - distance_driven (int): Maximum distance driven for filtering data. Defaults to 100k km.
        - loss_method (str): Loss function for optimization (e.g., RMSE, MAE, SMRE). Defaults to SMRE.
        - min_method (str): Optimization method (e.g., L-BFGS-B). Defaults to L-BFGS-B.
        - verbose (bool): Whether to print detailed logs. Defaults to False.
        """
        if verbose:
            print("Initializing MarketValueCalculator...")
        self.df_train = df_train
        self.loss_method = loss_method
        self.min_method = min_method
        self.distance_driven = distance_driven
        self.truncate_upper = truncate_upper
        self.truncate_lower = truncate_lower
        self.verbose = verbose
        self.fig = None
        self.ax = None
        self.car_brand = None
        self.car_model = None
        self.car_year = None
        self.car_variant_price = None
        self.v = None
        self.v_norm = None
        self.v_bounds = None
        self.scaling = None
        self.x_trains = None
        self.y_trains = None
        self.learning_rate = None
        self.log_len_trains = None
        self.results = None
        self.market_value = None
        self.df_market_value = None
        self.options = None
        self.response = {
            "market_value": None,
            "brand_new_price": None,
            "car_brand": None,
            "car_model": None,
            "car_year": None,
            "time_elapsed": None,
            "car_variant_price": None,
            "distance_driven": distance_driven,
            "min_method": min_method,
            "loss_method": loss_method,
            "verbose": verbose,
            "paramters": None,
            "min_response": None
        }

        if verbose:
            print("MarketValueCalculator Initialized")

    def initialize_optimization(self):
        "Prepare data and parameters for the optimization process."

        if self.verbose:
            print("Initializing optimization...")

        df = self.df_train
        cond1 = df["car_description"].str.contains(self.car_brand, case=False, na=False)
        cond2 = df["car_description"].str.contains(self.car_model, case=False, na=False)
        cond3 = df["distance_driven"] < self.distance_driven
        df = df[cond1 & cond2 & cond3].sort_values(by="car_year", ascending=False)
        df["car_years_old"] = dt.datetime.now().year - df["car_year"]

        df["price_filter"] = df["price"]*np.exp(df["car_years_old"]*0.075)
        cond4 = df["price_filter"] <= df["price_filter"].quantile(self.truncate_upper)
        cond5 = df["price_filter"] >= df["price_filter"].quantile(self.truncate_lower)
        # price_filter[cond4 & cond5]
        # cond4 = df["price"] <= df["price"].quantile(self.truncate_upper)
        # cond5 = df["price"] >= df["price"].quantile(self.truncate_lower)
        df = df[cond4 & cond5]

        scaling = df["price"].max()
        df["price_norm"] = df["price"]/scaling

        x_trains = [df["car_years_old"]]
        y_trains = [df["price_norm"]]
        len_trains = [len(df["price_norm"])]
        for category, df_category in df.groupby("car_years_old"):
            x_trains.append(df_category["car_years_old"])
            y_trains.append(df_category["price_norm"])
            len_trains.append(len(df_category["price_norm"]))

        self.v_norm = np.array([0.1, 0.9, -0.1])
        self.v = np.array([scaling*0.1, scaling*0.9, -0.01])
        self.v_bounds = [(0.01, 0.3), (0.01, 1), (-0.3, -0.03)]
        self.learning_rate = 10
        self.options = {"gtol": 1e-8, "ftol": 1e-8}
        self.df_train = df
        self.scaling = np.array(scaling)
        self.x_trains = x_trains
        self.y_trains = y_trains
        self.log_len_trains = np.log(np.array(len_trains))

        if self.verbose:
            print("Optimization Initialized")

    def calculate_loss(self, v, x_train, y_train, log_len_train):
        """
        Calculate the loss for a given set of parameters and training data.

        Parameters:
        - v (np.array{float}): Model parameters [y1, a, c].
        - x_train (np.array{float}): Training data for car years old.
        - y_train (np.array{float}): Scaled training prices.
        - log_len_train (float): Logarithm of the number of training data points.

        Returns:
        - loss (float): Calculated loss value.
        """
        y = model(v, x_train)

        loss = np.float64(0)
        if self.loss_method == "RMSE": # root mean squared error, ordinary ols regression 2-norm
            loss += np.sqrt(np.square(np.abs(np.subtract(y, y_train))).mean())
        elif self.loss_method == "MAE": # mean absolute error, robust regression 1-norm
            loss += np.abs(np.subtract(y, y_train)).mean()
        elif self.loss_method == "SMRE": # squared mean root error (not official name), more robust regression than MAE. Based on the p-norm or genaralized mean, specifically 1/2-norm
            loss += np.square(np.sqrt(np.abs(np.subtract(y, y_train)).mean()))
        else:
            raise ValueError(f'Unknown loss_method: {self.loss_method}. Current supported methods are RMSE, MAE, and SMRE.')

        # scale the weight loss function based on the log number of datapoints to avoid overfitting years with overwhelmingly large datapoints
        loss *= log_len_train
        return loss

    def calculate_cost(self, v):
        """
        Calculate the total cost for optimization including constraints.

        Parameters:
        - v (np.array{float}): Model parameters [y1, a, c].

        Returns:
        - cost (float): Total cost value.
        """
        y1, a, c = v
        cost = np.abs(np.subtract(np.multiply(9, y1), a)) # constrain the base price y1 to be 10% the brand new price: y1 = 0.1(y1+a) -> 9y1 = a, not strict
        for i in range(len(self.y_trains)):
            x_train = self.x_trains[i]
            y_train = self.y_trains[i]
            log_len_train = self.log_len_trains[i]
            cost += self.calculate_loss(v, x_train, y_train, log_len_train)
        cost *= self.learning_rate

        if self.verbose:
            print("parameters:", v, "cost:", cost)

        return cost

    def optimize_parameters(self):
        "Optimize the model parameters using the specified optimization method."

        results = minimize(self.calculate_cost, self.v_norm,
                                method = self.min_method,
                                bounds = self.v_bounds,
                                options = self.options)
        self.v = np.array([results.x[0]*self.scaling, results.x[1]*self.scaling, results.x[2]])
        self.v_norm = results.x
        self.results = results

    def evaluate_market_value(self):
        "Evaluate the market value of the car based on the optimized parameters."

        car_years_old = np.array(range(0, round(self.df_train["car_years_old"].max()+1)))
        car_year = dt.datetime.now().year - car_years_old
        market_value = model(self.v, car_years_old)
        y1, a, c = self.v
        if self.car_variant_price is not None:
            correction = self.car_variant_price/(y1+a)
            market_value *= correction
            y1 *= correction
            a *= correction

        df = pd.DataFrame({"car_years_old": car_years_old, "car_year": car_year, "market_value": market_value})
        self.market_value = df[df["car_year"] == int(self.car_year)]["market_value"].values[0]
        self.df_market_value = df
        self.brand_new_price = y1+a

        if self.verbose:
            print(f"{self.car_brand} {self.car_model} brand new price: {self.brand_new_price}")
            print(f"{self.car_brand} {self.car_model} {self.car_year} market value: {self.market_value}")

    def visualize_result(self):
        "Visualize the market value trends and car listings."

        fig, ax = plt.subplots()
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        # plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100000))

        x_market_value = self.df_market_value["car_years_old"]
        y_market_value = self.df_market_value["market_value"]
        ax.plot(x_market_value, y_market_value, color="purple", label="Exponential decay model (depreciation)")

        x_listings = self.df_train["car_years_old"]
        y_listings = self.df_train["price"]
        # y_listings = self.df_train["price_filter"]
        ax.scatter(x_listings, y_listings, color="blue", alpha=0.1, label="listings")

        df_car = self.df_market_value[self.df_market_value["car_year"] == float(self.car_year)]
        ax.scatter(df_car["car_years_old"], df_car["market_value"], color="red", label="Calculated market value")

        ax.set_title(f"{self.car_brand} {self.car_model} {self.car_year}")
        ax.set_xlabel("Years old")
        ax.set_ylabel("Price")
        ax.set_xlim(-0.5, 20.5)
        ax.legend()

        self.fig = fig
        self.ax = ax

    def calculate_market_value(self, car_brand, car_model, car_year, car_variant_price=None):
        """
        Main method to calculate the market value of a car based on its details.

        Parameters:
        - car_brand (str): Brand of the car.
        - car_model (str): Model of the car.
        - car_year (int): Manufacturing year of the car.
        - car_variant_price (float, optional): Price of the specific car variant. Defaults to None.

        Returns:
        - response (dict): Summary of the market value calculation process.
        """
        t_i = time()
        self.car_brand = car_brand
        self.car_model = car_model
        self.car_year = car_year
        self.car_variant_price = car_variant_price

        self.initialize_optimization()
        self.optimize_parameters()
        self.evaluate_market_value()
        if self.verbose:
            self.visualize_result()
        t_f = time()

        self.response["market_value"] = round(self.market_value)
        self.response["brand_new_price"] = round(self.brand_new_price)
        self.response["car_brand"] = car_brand
        self.response["car_model"] = car_model
        self.response["car_year"] = car_year
        self.response["time_elapsed"] = t_f - t_i
        self.response["car_variant_price"] = car_variant_price
        self.response["paramters"] = {"y1": float(self.v[0]), "a": float(self.v[1]), "c": float(self.v[2])}
        self.response["min_response"] = dict(self.results)

        return self.response