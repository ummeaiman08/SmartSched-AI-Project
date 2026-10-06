import pandas as pd
from pathlib import Path


# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of the CSV files
DATA_DIR = BASE_DIR / "data"


def load_data():
    """Load all SmartSched AI input data from CSV files."""

    courses = pd.read_csv(DATA_DIR / "courses.csv")
    faculty = pd.read_csv(DATA_DIR / "faculty.csv")
    rooms = pd.read_csv(DATA_DIR / "rooms.csv")
    sections = pd.read_csv(DATA_DIR / "sections.csv")
    timeslots = pd.read_csv(DATA_DIR / "timeslots.csv")

    return {
        "courses": courses,
        "faculty": faculty,
        "rooms": rooms,
        "sections": sections,
        "timeslots": timeslots,
    }