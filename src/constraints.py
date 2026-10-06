def check_faculty_conflict(schedule, faculty_id, slot_id):
    """Check whether faculty is already assigned in this time slot."""

    for item in schedule:
        if item["faculty_id"] == faculty_id and item["slot_id"] == slot_id:
            return True

    return False


def check_room_conflict(schedule, room_id, slot_id):
    """Check whether a room is already occupied in this time slot."""

    for item in schedule:
        if item["room_id"] == room_id and item["slot_id"] == slot_id:
            return True

    return False


def check_section_conflict(schedule, section_id, slot_id):
    """Check whether a section already has a class in this time slot."""

    for item in schedule:
        if item["section_id"] == section_id and item["slot_id"] == slot_id:
            return True

    return False


def check_room_capacity(room_capacity, required_capacity):
    """Check whether the room can accommodate the required students."""

    return room_capacity >= required_capacity


def check_faculty_availability(faculty_availability, faculty_id, slot_id):
    """Check whether faculty is available for the given time slot."""

    if faculty_id not in faculty_availability:
        return True

    return slot_id in faculty_availability[faculty_id]


def is_valid_assignment(
    schedule,
    faculty_id,
    section_id,
    room_id,
    slot_id,
    room_capacity,
    required_capacity,
    faculty_availability,
):
    """Check all major constraints for a proposed timetable assignment."""

    if check_faculty_conflict(schedule, faculty_id, slot_id):
        return False

    if check_room_conflict(schedule, room_id, slot_id):
        return False

    if check_section_conflict(schedule, section_id, slot_id):
        return False

    if not check_room_capacity(room_capacity, required_capacity):
        return False

    if not check_faculty_availability(
        faculty_availability,
        faculty_id,
        slot_id,
    ):
        return False

    return True