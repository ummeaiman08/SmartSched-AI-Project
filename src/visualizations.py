import plotly.express as px


def faculty_workload_chart(timetable):
    """Create a bar chart showing classes assigned to each faculty member."""

    workload = (
        timetable[timetable["day"] != "UNSCHEDULED"]
        .groupby("faculty_id")
        .size()
        .reset_index(name="classes")
    )

    fig = px.bar(
        workload,
        x="faculty_id",
        y="classes",
        title="Faculty Workload",
        labels={
            "faculty_id": "Faculty",
            "classes": "Number of Classes",
        },
    )

    return fig


def room_utilization_chart(timetable):
    """Create a bar chart showing room usage."""

    utilization = (
        timetable[timetable["day"] != "UNSCHEDULED"]
        .groupby("room_id")
        .size()
        .reset_index(name="classes")
    )

    fig = px.bar(
        utilization,
        x="room_id",
        y="classes",
        title="Room Utilization",
        labels={
            "room_id": "Room",
            "classes": "Number of Classes",
        },
    )

    return fig


def daily_class_chart(timetable):
    """Create a bar chart showing classes scheduled each day."""

    daily_classes = (
        timetable[timetable["day"] != "UNSCHEDULED"]
        .groupby("day")
        .size()
        .reset_index(name="classes")
    )

    fig = px.bar(
        daily_classes,
        x="day",
        y="classes",
        title="Classes by Day",
        labels={
            "day": "Day",
            "classes": "Number of Classes",
        },
    )

    return fig