
import numpy as np
import pandas as pd


class WindSolarResourceEstimator:

    def __init__(self, data):
        self.data = data.copy()

        self.wind_speed = self.data["wind_speed_kmh"].to_numpy()
        self.temperature = (
            (self.data["temp_max"] + self.data["temp_min"]) / 2
        ).to_numpy()
        self.pressure = self.data["pressure_hpa"].to_numpy()

        self.R = 287.05

    def get_data_info(self):
        print("Total records:", len(self.data))
        print("Total cities:", self.data["city"].nunique())
        print("Columns:", list(self.data.columns))
    
    def calculate_numpy_power_density(self):

        # Convert wind speed from km/h to m/s
        wind_speed_ms = self.wind_speed / 3.6

        # Convert temperature from Celsius to Kelvin
        temperature_kelvin = self.temperature + 273.15

        # Convert pressure from hPa to Pascal
        pressure_pa = self.pressure * 100

        # Calculate air density
        rho = pressure_pa / (self.R * temperature_kelvin)

        # Calculate cubic wind power density
        power_density = 0.5 * rho * np.power(wind_speed_ms, 3)

        return rho, power_density
    
    def analyze_high_resource_days(self):

        # Calculate wind power density
        rho, power_density = self.calculate_numpy_power_density()

        # High wind days: WPD > 200 W/m2
        high_wind_mask = power_density > 200

        # High sunshine days: sunshine > 8 hours
        high_sunshine_mask = (
            self.data["sunshine_hours"].to_numpy() > 8
        )

        # Days having both high wind and high sunshine
        both_mask = high_wind_mask & high_sunshine_mask

        # Days having either high wind or high sunshine
        either_mask = high_wind_mask | high_sunshine_mask

        # Calculate statistics
        average_wind = np.mean(self.wind_speed)
        std_wind = np.std(self.wind_speed)

        # Find matching dates using Python sets
        wind_dates = set(
            self.data.loc[high_wind_mask, "date"]
        )

        sunshine_dates = set(
            self.data.loc[high_sunshine_mask, "date"]
        )

        common_dates = wind_dates & sunshine_dates

        return {
            "high_wind_count": np.sum(high_wind_mask),
            "high_sunshine_count": np.sum(high_sunshine_mask),
            "both_count": np.sum(both_mask),
            "either_count": np.sum(either_mask),
            "average_wind": average_wind,
            "std_wind": std_wind,
            "common_dates": common_dates
        }
    
    def calculate_regression(self):

        # Sort data by city and date
        sorted_data = self.data.copy()
        sorted_data["date"] = pd.to_datetime(sorted_data["date"])
        sorted_data = sorted_data.sort_values(
            ["city", "date"]
        )

        # Calculate pressure gradient city-wise
        pressure_gradient = (
            sorted_data.groupby("city")["pressure_hpa"]
            .diff()
            .fillna(0)
            .to_numpy()
        )

        # Daily temperature range
        temperature_range = (
            sorted_data["temp_max"] -
            sorted_data["temp_min"]
        ).to_numpy()

        # Target variable
        y = sorted_data["wind_speed_kmh"].to_numpy()

        # Feature matrix
        X = np.column_stack((
            np.ones(len(sorted_data)),
            pressure_gradient,
            temperature_range
        ))

        # OLS regression
        coefficients, residuals, rank, singular_values = (
            np.linalg.lstsq(X, y, rcond=None)
        )

        # Predictions
        predictions = X @ coefficients

        # Residual errors
        errors = y - predictions

        # R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum(errors ** 2)

        r_squared = 1 - (ss_residual / ss_total)

        return {
            "coefficients": coefficients,
            "predictions": predictions,
            "residuals": errors,
            "r_squared": r_squared,
            "rank": rank
        }

   
    def audit_city_completeness(self):

        # Convert date column to datetime
        audit_data = self.data.copy()
        audit_data["date"] = pd.to_datetime(audit_data["date"])

        # Count records for each city
        city_counts = audit_data.groupby("city")["date"].count()

        # Calculate expected days separately for each city
        expected_days = audit_data.groupby("city")["date"].agg(
            lambda dates: (
                dates.max() - dates.min()
            ).days + 1
        )

        # Calculate city-wise data coverage
        coverage = (city_counts / expected_days) * 100

        # Count missing values in each column
        missing_values = audit_data.isna().sum()

        # Identify coastal cities available in dataset
        coastal_cities = ["Mumbai", "Chennai"]

        coastal_data = audit_data[
            audit_data["city"].isin(coastal_cities)
        ]

        inland_data = audit_data[
            ~audit_data["city"].isin(coastal_cities)
        ]

        return {
            "expected_days": expected_days,
            "city_counts": city_counts,
            "coverage": coverage,
            "missing_values": missing_values,
            "coastal_missing": coastal_data.isna().sum().sum(),
            "inland_missing": inland_data.isna().sum().sum()
        }

    def city_season_pivot(self):

        # Calculate wind power density
        rho, power_density = self.calculate_numpy_power_density()

        # Create a copy of the dataset
        seasonal_data = self.data.copy()

        seasonal_data["wind_power_density"] = power_density

        # Extract month
        month = pd.to_datetime(
            seasonal_data["date"]
        ).dt.month

        # Assign seasons using NumPy select
        conditions = [
            month.isin([6, 7, 8, 9]),
            month.isin([11, 12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([10])
        ]

        choices = [
            "Monsoon",
            "Winter",
            "Summer",
            "Post-Monsoon"
        ]

        seasonal_data["season"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        # Group by city and season
        grouped_data = seasonal_data.groupby(
            ["city", "season"]
        )["wind_power_density"].mean().reset_index()

        # Create pivot table
        pivot_table = seasonal_data.pivot_table(
            index="city",
            columns="season",
            values="wind_power_density",
            aggfunc="mean",
            margins=True
        )

        return grouped_data, pivot_table
    def renewable_potential_ranking(self):

        # Calculate wind power density
        rho, power_density = self.calculate_numpy_power_density()

        # Create a copy of the dataset
        renewable_data = self.data.copy()

        renewable_data["wind_power_density"] = power_density

        # Extract month
        month = pd.to_datetime(
            renewable_data["date"]
        ).dt.month

        # Assign seasons
        conditions = [
            month.isin([6, 7, 8, 9]),
            month.isin([11, 12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([10])
        ]

        choices = [
            "Monsoon",
            "Winter",
            "Summer",
            "Post-Monsoon"
        ]

        renewable_data["season"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        # Calculate city-wise seasonal averages
        seasonal_summary = renewable_data.groupby(
            ["city", "season"]
        )[
            ["wind_power_density", "sunshine_hours"]
        ].mean().reset_index()

        # Convert wide format into long format using melt
        long_data = pd.melt(
            seasonal_summary,
            id_vars=["city", "season"],
            value_vars=[
                "wind_power_density",
                "sunshine_hours"
            ],
            var_name="resource_type",
            value_name="resource_value"
        )

        # Rank cities separately for each season and resource
        long_data["rank"] = long_data.groupby(
            ["season", "resource_type"]
        )["resource_value"].rank(
            ascending=False,
            method="dense"
        )

        # Extract top 5 cities for each season and resource
        top_five = long_data[
            long_data["rank"] <= 5
        ].sort_values(
            ["season", "resource_type", "rank"]
        )

        return long_data, top_five
        