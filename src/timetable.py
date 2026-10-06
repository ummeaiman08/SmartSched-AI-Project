import pandas as pd


def schedule_to_dataframe(schedule):
    """
    Convert the generated schedule into a Pandas DataFrame.
    """

    return pd.DataFrame(schedule)


def sort_timetable(timetable):
    """
    Sort the timetable by day and start time.
    """

    day_order = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "UNSCHEDULED": 6,
    }

    timetable = timetable.copy()

    timetable["day_order"] = timetable["day"].map(day_order)

    timetable = timetable.sort_values(
        by=["day_order", "start_time"],
        na_position="last",
    )

    timetable = timetable.drop(columns=["day_order"])

    return timetable.reset_index(drop=True)


def prepare_timetable(schedule):
    """
    Convert and sort the generated schedule.
    """

    timetable = schedule_to_dataframe(schedule)

    if timetable.empty:
        return timetable

    return sort_timetable(timetable)


def export_timetable_csv(timetable, file_path):
    """
    Save the timetable as a CSV file.
    """

    timetable.to_csv(file_path, index=False)