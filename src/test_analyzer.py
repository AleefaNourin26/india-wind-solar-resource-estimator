
import pandas as pd
from analyzer import WindSolarResourceEstimator

# Load the cleaned dataset
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

file_path = project_folder / "data" / "processed" / "clean_india_weather_25yr.csv"

df = pd.read_csv(file_path)

# Create an object of the class
estimator = WindSolarResourceEstimator(df)

# Test the class method
estimator.get_data_info()

import pandas as pd
import numpy as np
from analyzer import WindSolarResourceEstimator

# Load dataset
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "processed" / "clean_india_weather_25yr.csv"

df = pd.read_csv(file_path)

# Create estimator object
estimator = WindSolarResourceEstimator(df)

# Calculate using NumPy method
rho, power_density = estimator.calculate_numpy_power_density()

# Verify with independent Pandas calculations
temperature_kelvin = (
    (df["temp_max"] + df["temp_min"]) / 2
) + 273.15

pressure_pa = df["pressure_hpa"] * 100
wind_speed_ms = df["wind_speed_kmh"] / 3.6

expected_rho = pressure_pa / (287.05 * temperature_kelvin)

expected_power = (
    0.5 * expected_rho * np.power(wind_speed_ms, 3)
)

# Compare both results
print("Air density verification:", np.allclose(rho, expected_rho))
print("Wind power density verification:", np.allclose(power_density, expected_power))

print("Average air density:", np.mean(rho))
print("Average wind power density:", np.mean(power_density))
# Q5: High Wind vs High Sunshine Days

results = estimator.analyze_high_resource_days()

print(" High Resource Days Analysis")
print("High wind days:", results["high_wind_count"])
print("High sunshine days:", results["high_sunshine_count"])
print("Both high wind and sunshine:", results["both_count"])
print("Either high wind or sunshine:", results["either_count"])
print("Average wind speed:", results["average_wind"])
print("Standard deviation of wind speed:", results["std_wind"])
print("Number of common dates:", len(results["common_dates"]))
# Q6: Wind Speed vs Pressure OLS Regression

regression_results = estimator.calculate_regression()

print(" - OLS Regression Analysis")

print("Regression coefficients:",
      regression_results["coefficients"])

print("R-squared:",
      regression_results["r_squared"])

print("Regression rank:",
      regression_results["rank"])

print("Mean absolute residual:",
      np.mean(np.abs(regression_results["residuals"])))

print("Mean squared residual:",
      np.mean(regression_results["residuals"] ** 2))
# Q7: City Decadal Completeness & Missing Sensor Audit

audit = estimator.audit_city_completeness()

print(" City Completeness Audit")

print("Expected days:", audit["expected_days"])

print("\nCity-wise record counts:")
print(audit["city_counts"])

print("\nCity-wise data coverage (%):")
print(audit["coverage"].round(2))

print("\nMissing values in each column:")
print(audit["missing_values"])

print("\nCoastal cities missing values:",
      audit["coastal_missing"])

print("Inland cities missing values:",
      audit["inland_missing"])
# Q8: City-by-Season Discretization & Pivot Table

grouped_data, pivot = estimator.city_season_pivot()

print("City-by-Season Mean Wind Power Density")

print("\nGrouped City-Season Data:")
print(grouped_data)

print("\nCity-by-Season Pivot Table:")
print(pivot.round(2))
# Q9: Wide-to-Long Energy Potential Ranking via Melt

long_data, top_five = estimator.renewable_potential_ranking()

print(" - Renewable Energy Potential Ranking")

print("\nLong Format Data:")
print(long_data.head(10))

print("\nTop 5 Cities for Each Season and Resource:")
print(top_five.to_string(index=False))