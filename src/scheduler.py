from collections import Counter

from .constraints import is_valid_assignment


def _time_to_minutes(value):
    """Convert HH:MM time into minutes."""

    if value is None:
        return None

    try:
        hours, minutes = str(value).split(":")[:2]
        return int(hours) * 60 + int(minutes)
    except (ValueError, TypeError):
        return None


def _assignment_score(
    schedule,
    task,
    slot,
    room,
    required_capacity,
):
    """
    Score a possible assignment.

    Higher score = better assignment.
    """

    score = 100.0

    faculty_id = task["faculty_id"]
    section_id = task["section_id"]
    day = slot["day"]

    room_capacity = float(room["capacity"])
    required_capacity = float(required_capacity)

    # -----------------------------------------------------
    # 1. FACULTY DAILY BALANCE
    # -----------------------------------------------------

    faculty_day_count = sum(
        1
        for item in schedule
        if item["faculty_id"] == faculty_id
        and item["day"] == day
    )

    score -= faculty_day_count * 8

    # -----------------------------------------------------
    # 2. SECTION DAILY BALANCE
    # -----------------------------------------------------

    section_day_count = sum(
        1
        for item in schedule
        if item["section_id"] == section_id
        and item["day"] == day
    )

    score -= section_day_count * 5

    # -----------------------------------------------------
    # 3. AVOID VERY HEAVY DAYS
    # -----------------------------------------------------

    day_count = sum(
        1
        for item in schedule
        if item["day"] == day
    )

    score -= day_count * 2

    # -----------------------------------------------------
    # 4. ROOM CAPACITY EFFICIENCY
    # -----------------------------------------------------

    if room_capacity > 0:

        utilization = required_capacity / room_capacity

        # Prefer rooms that are reasonably close to
        # the required capacity.
        if utilization >= 0.80:
            score += 8
        elif utilization >= 0.60:
            score += 5
        elif utilization >= 0.40:
            score += 2
        else:
            score -= 3

    # -----------------------------------------------------
    # 5. AVOID IMMEDIATE BACK-TO-BACK FACULTY LOAD
    # -----------------------------------------------------

    slot_start = _time_to_minutes(
        slot["start_time"]
    )

    slot_end = _time_to_minutes(
        slot["end_time"]
    )

    if slot_start is not None and slot_end is not None:

        for item in schedule:

            if (
                item["faculty_id"] != faculty_id
                or item["day"] != day
            ):
                continue

            existing_start = _time_to_minutes(
                item["start_time"]
            )

            existing_end = _time_to_minutes(
                item["end_time"]
            )

            if (
                existing_start is None
                or existing_end is None
            ):
                continue

            # Small penalty for immediately adjacent classes.
            if (
                existing_end == slot_start
                or existing_start == slot_end
            ):
                score -= 2

    return score


def generate_schedule(data):
    """
    Generate an optimized timetable.

    The scheduler evaluates all valid slot/room
    combinations and selects the highest-scoring
    assignment.
    """

    courses = data["courses"]
    faculty = data["faculty"]
    sections = data["sections"]
    rooms = data["rooms"]
    timeslots = data["timeslots"]

    # -----------------------------------------------------
    # SECTION CAPACITY LOOKUP
    # -----------------------------------------------------

    section_strength = dict(
        zip(
            sections["section_id"],
            sections["strength"],
        )
    )

    # Faculty availability is not stored in the
    # current sample data, so all faculty are
    # considered available.
    faculty_availability = {}

    schedule = []

    # -----------------------------------------------------
    # CREATE WEEKLY CLASS TASKS
    # -----------------------------------------------------

    class_tasks = []

    for _, course in courses.iterrows():

        classes_per_week = int(
            course["classes_per_week"]
        )

        for class_number in range(
            1,
            classes_per_week + 1,
        ):

            class_tasks.append(
                {
                    "course_id": course["course_id"],
                    "course_name": course["course_name"],
                    "faculty_id": course["faculty_id"],
                    "section_id": course["section_id"],
                    "room_type": course["room_type"],
                    "class_number": class_number,
                }
            )

    # -----------------------------------------------------
    # SCHEDULE MORE CONSTRAINED TASKS FIRST
    # -----------------------------------------------------

    # Courses requiring larger sections or special
    # room types are harder to place, so prioritize them.
    class_tasks.sort(
        key=lambda task: (
            -float(
                section_strength[
                    task["section_id"]
                ]
            ),
            task["course_id"],
            task["class_number"],
        )
    )

    # -----------------------------------------------------
    # ASSIGN EACH CLASS
    # -----------------------------------------------------

    for task in class_tasks:

        required_capacity = section_strength[
            task["section_id"]
        ]

        best_assignment = None
        best_score = float("-inf")

        # -------------------------------------------------
        # CHECK EVERY POSSIBLE SLOT
        # -------------------------------------------------

        for _, slot in timeslots.iterrows():

            # -------------------------------------------------
            # CHECK EVERY POSSIBLE ROOM
            # -------------------------------------------------

            for _, room in rooms.iterrows():

                room_id = room["room_id"]
                room_capacity = room["capacity"]
                room_type = room["room_type"]

                # Room type must match course requirement.
                if room_type != task["room_type"]:
                    continue

                # Existing hard constraints.
                if not is_valid_assignment(
                    schedule=schedule,
                    faculty_id=task["faculty_id"],
                    section_id=task["section_id"],
                    room_id=room_id,
                    slot_id=slot["slot_id"],
                    room_capacity=room_capacity,
                    required_capacity=required_capacity,
                    faculty_availability=faculty_availability,
                ):
                    continue

                # Calculate quality of this assignment.
                candidate_score = _assignment_score(
                    schedule=schedule,
                    task=task,
                    slot=slot,
                    room=room,
                    required_capacity=required_capacity,
                )

                # Keep the best valid assignment.
                if candidate_score > best_score:

                    best_score = candidate_score

                    best_assignment = {
                        "course_id": task["course_id"],
                        "course_name": task["course_name"],
                        "faculty_id": task["faculty_id"],
                        "section_id": task["section_id"],
                        "room_id": room_id,
                        "slot_id": slot["slot_id"],
                        "day": slot["day"],
                        "start_time": slot["start_time"],
                        "end_time": slot["end_time"],
                        "class_number": task["class_number"],
                    }

        # -------------------------------------------------
        # ADD BEST ASSIGNMENT
        # -------------------------------------------------

        if best_assignment is not None:

            schedule.append(
                best_assignment
            )

        # -------------------------------------------------
        # UNSCHEDULED FALLBACK
        # -------------------------------------------------

        else:

            schedule.append(
                {
                    "course_id": task["course_id"],
                    "course_name": task["course_name"],
                    "faculty_id": task["faculty_id"],
                    "section_id": task["section_id"],
                    "room_id": None,
                    "slot_id": None,
                    "day": "UNSCHEDULED",
                    "start_time": None,
                    "end_time": None,
                    "class_number": task["class_number"],
                }
            )

    return schedule