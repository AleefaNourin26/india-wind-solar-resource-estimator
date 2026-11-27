import csv
from pathlib import Path
from datetime import datetime

def validate_record(row):
    try:
        temp_max = float(row["temp_max"])
        temp_min = float(row["temp_min"])
        wind_speed = float(row["wind_speed_kmh"])
        pressure = float(row["pressure_hpa"])

        if not (-5 <= temp_max <= 55):
            raise ValueError("Maximum temperature out of range")

        if not (-5 <= temp_min <= 55):
            raise ValueError("Minimum temperature out of range")

        if wind_speed < 0:
            raise ValueError("Negative wind speed")

        if not (900 <= pressure <= 1050):
            raise ValueError("Pressure out of range")

        if temp_min > temp_max:
            raise ValueError("Minimum temperature exceeds maximum")

        date_value = row["date"].strip()

        try:
            datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError:
            datetime.strptime(date_value, "%m/%d/%Y")

        return True, "Valid record"

    except ValueError as e:
        return False, str(e)

    except KeyError as e:
        return False, f"Missing column: {e}"


def load_and_validate_csv(file_path):

    valid_records = []
    error_log = []

    try:
        with open(file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                is_valid, message = validate_record(row)

                if is_valid:
                    valid_records.append(row)
                else:
                    error_log.append({
                        "city": row.get("city", "Unknown"),
                        "date": row.get("date", "Unknown"),
                        "error": message
                    })

    except FileNotFoundError:
        print("Error: Dataset file not found.")

    except OSError as e:
        print("File reading error:", e)

    return valid_records, error_log


def export_clean_data(valid_records, output_dir):

    output_dir = Path(output_dir)
    clean_file = output_dir / "clean_india_weather_25yr.csv"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        if not valid_records:
            print("No valid records available for export.")
            return None

        with open(clean_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=valid_records[0].keys()
            )
            writer.writeheader()
            writer.writerows(valid_records)

        print("Clean CSV exported successfully!")
        print("Records written:", len(valid_records))

        return clean_file

    except OSError as e:
        print("File writing error:", e)
        return None


def create_audit_log(valid_records, output_dir):

    output_dir = Path(output_dir)
    audit_file = output_dir / "wind_solar_audit.log"

    missing_data = {}

    for row in valid_records:
        city = row["city"]

        if city not in missing_data:
            missing_data[city] = {
                "sunshine": 0,
                "wind": 0
            }

        if not row["sunshine_hours"].strip():
            missing_data[city]["sunshine"] += 1

        if not row["wind_speed_kmh"].strip():
            missing_data[city]["wind"] += 1

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(audit_file, "w", encoding="utf-8") as file:
            file.write("WIND AND SOLAR DATA AUDIT REPORT\n")
            file.write("=" * 40 + "\n")

            for city, counts in missing_data.items():
                file.write(
                    f"City: {city} | "
                    f"Missing Sunshine: {counts['sunshine']} | "
                    f"Missing Wind: {counts['wind']}\n"
                )

            file.write("\nAudit completed successfully.\n")

        print("Audit log created successfully!")
        return audit_file

    except OSError as e:
        print("Audit log writing error:", e)
        return None



class WindSolarResourceEstimator:

    def __init__(self):
        self.data = None
        self.hub_height = 100.0

    
    def load_from_csv(self, file_path):
        try:
            with open(file_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.data = list(reader)

            print("Dataset loaded successfully!")
            print("Total records:", len(self.data))

        except FileNotFoundError:
            print("Error: CSV file not found.")


    def compute_power_density(self, hub_height=100.0):

        if hub_height <= 0:
            raise ValueError("Hub height must be greater than zero.")

        if self.data is None:
            raise ValueError("Load the dataset first.")

        self.hub_height = hub_height

        for row in self.data:
            temperature = (
                float(row["temp_max"]) +
                float(row["temp_min"])
            ) / 2

            temp_kelvin = temperature + 273.15
            pressure_pa = float(row["pressure_hpa"]) * 100
            wind_speed = float(row["wind_speed_kmh"]) / 3.6

            air_density = pressure_pa / (287.05 * temp_kelvin)
            power_density = 0.5 * air_density * (wind_speed ** 3)

            row["air_density_rho"] = air_density
            row["wind_power_density"] = power_density

        print("Wind power density calculated successfully!")

    def summary_report(self):

        if self.data is None:
            raise ValueError("Load the dataset first.")

        total_records = len(self.data)

        average_density = sum(
            row["wind_power_density"] for row in self.data
        ) / total_records

        print("\nWIND AND SOLAR RESOURCE SUMMARY")
        print("--------------------------------")
        print("Total records:", total_records)
        print("Hub height:", self.hub_height, "metres")
        print("Average wind power density:",
              round(average_density, 2), "W/m²")