from collections import Counter, defaultdict


# =========================================================
# SCHEDULE QUALITY SCORE
# =========================================================

def calculate_schedule_score(schedule):
    """
    Calculate a quality score for the timetable.

    Higher score = better timetable.
    Maximum score = 100.
    """

    score = 100

    # -----------------------------------------------------
    # 1. UNSCHEDULED CLASSES
    # -----------------------------------------------------

    unscheduled = [
        item
        for item in schedule
        if item.get("day") == "UNSCHEDULED"
    ]

    score -= len(unscheduled) * 20

    # If nothing is scheduled, return a minimum score.
    if not schedule:
        return 0

    # Only evaluate scheduled classes below.
    scheduled = [
        item
        for item in schedule
        if item.get("day") != "UNSCHEDULED"
    ]

    if not scheduled:
        return max(score, 0)

    # -----------------------------------------------------
    # 2. FACULTY DAILY WORKLOAD
    # -----------------------------------------------------

    faculty_daily = Counter(
        (
            item.get("faculty_id"),
            item.get("day"),
        )
        for item in scheduled
    )

    for (_, _), count in faculty_daily.items():

        # More than 3 classes in one day is considered heavy.
        if count > 3:
            score -= (count - 3) * 3

    # -----------------------------------------------------
    # 3. SECTION DAILY WORKLOAD
    # -----------------------------------------------------

    section_daily = Counter(
        (
            item.get("section_id"),
            item.get("day"),
        )
        for item in scheduled
    )

    for (_, _), count in section_daily.items():

        # More than 4 classes in one day is considered heavy.
        if count > 4:
            score -= (count - 4) * 2

    # -----------------------------------------------------
    # 4. ROOM CAPACITY EFFICIENCY
    # -----------------------------------------------------

    # If capacity information exists, penalize extremely
    # oversized room assignments.
    for item in scheduled:

        room_capacity = item.get("room_capacity")
        section_strength = item.get("section_strength")

        if (
            room_capacity is not None
            and section_strength is not None
        ):

            try:
                room_capacity = float(room_capacity)
                section_strength = float(section_strength)

                if section_strength > 0:

                    utilization = (
                        section_strength
                        / room_capacity
                    )

                    # Penalize very low utilization.
                    if utilization < 0.30:
                        score -= 1

            except (ValueError, TypeError, ZeroDivisionError):
                pass

    # -----------------------------------------------------
    # 5. SAME FACULTY CONSECUTIVE LOAD
    # -----------------------------------------------------

    faculty_day_slots = defaultdict(list)

    for item in scheduled:

        faculty = item.get("faculty_id")
        day = item.get("day")
        start_time = item.get("start_time")

        faculty_day_slots[
            (faculty, day)
        ].append(start_time)

    for slots in faculty_day_slots.values():

        # Three or more classes in one day may create
        # a heavier teaching load.
        if len(slots) >= 3:
            score -= 1

    # -----------------------------------------------------
    # 6. ROOM OVER-CONSOLIDATION
    # -----------------------------------------------------

    room_daily = Counter(
        (
            item.get("room_id"),
            item.get("day"),
        )
        for item in scheduled
    )

    for (_, _), count in room_daily.items():

        if count > 5:
            score -= 1

    # -----------------------------------------------------
    # FINAL SCORE
    # -----------------------------------------------------

    return max(min(score, 100), 0)


# =========================================================
# OPTIMIZE / EVALUATE SCHEDULE
# =========================================================

def optimize_schedule(schedule):
    """
    Evaluate the generated timetable and return
    the timetable together with its quality score.
    """

    score = calculate_schedule_score(schedule)

    return schedule, score