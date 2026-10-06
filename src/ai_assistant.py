from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# OLLAMA MODEL
# =========================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1,
)


# =========================================================
# BUILD RELIABLE TIMETABLE STATISTICS
# =========================================================

def build_timetable_summary(timetable):

    if "day" in timetable.columns:
        scheduled = timetable[
            timetable["day"].astype(str).str.upper()
            != "UNSCHEDULED"
        ].copy()
    else:
        scheduled = timetable.copy()

    summary = []

    summary.append(
        f"TOTAL SCHEDULED CLASSES: {len(scheduled)}"
    )

    # -----------------------------------------------------
    # FACULTY WORKLOAD
    # -----------------------------------------------------

    if "faculty_id" in scheduled.columns:

        faculty_counts = (
            scheduled["faculty_id"]
            .value_counts()
            .sort_index()
        )

        max_classes = int(faculty_counts.max())

        top_faculty = [
            str(faculty)
            for faculty, count in faculty_counts.items()
            if int(count) == max_classes
        ]

        workload = ", ".join(
            f"{faculty}={int(count)}"
            for faculty, count in faculty_counts.items()
        )

        summary.append(
            f"FACULTY WORKLOAD: {workload}"
        )

        summary.append(
            f"HIGHEST FACULTY WORKLOAD: {max_classes} classes"
        )

        summary.append(
            f"FINAL FACULTY RESULT: "
            f"{', '.join(top_faculty)} "
            f"with {max_classes} classes each"
        )

        summary.append(
            "FACULTY RESULT TYPE: "
            + ("TIE" if len(top_faculty) > 1 else "SINGLE")
        )

    # -----------------------------------------------------
    # COURSE WORKLOAD
    # -----------------------------------------------------

    if "course_id" in scheduled.columns:

        course_counts = (
            scheduled["course_id"]
            .value_counts()
            .sort_index()
        )

        summary.append(
            "COURSE WORKLOAD: "
            + ", ".join(
                f"{course}={int(count)}"
                for course, count
                in course_counts.items()
            )
        )

    # -----------------------------------------------------
    # SECTION WORKLOAD
    # -----------------------------------------------------

    if "section_id" in scheduled.columns:

        section_counts = (
            scheduled["section_id"]
            .value_counts()
            .sort_index()
        )

        summary.append(
            "SECTION WORKLOAD: "
            + ", ".join(
                f"{section}={int(count)}"
                for section, count
                in section_counts.items()
            )
        )

    # -----------------------------------------------------
    # ROOM USAGE
    # -----------------------------------------------------

    if "room_id" in scheduled.columns:

        room_counts = (
            scheduled["room_id"]
            .value_counts()
            .sort_index()
        )

        summary.append(
            "ROOM USAGE: "
            + ", ".join(
                f"{room}={int(count)}"
                for room, count
                in room_counts.items()
            )
        )

    # -----------------------------------------------------
    # DAILY CLASSES
    # -----------------------------------------------------

    if "day" in scheduled.columns:

        daily_counts = (
            scheduled["day"]
            .value_counts()
            .sort_index()
        )

        summary.append(
            "DAILY CLASSES: "
            + ", ".join(
                f"{day}={int(count)}"
                for day, count
                in daily_counts.items()
            )
        )

    return scheduled, "\n".join(summary)


# =========================================================
# EXACT COURSE LOOKUP
# =========================================================

def find_courses_by_faculty(timetable, faculty_id):

    if "faculty_id" not in timetable.columns:
        return []

    faculty_id = str(faculty_id).strip().upper()

    matches = timetable[
        timetable["faculty_id"]
        .astype(str)
        .str.strip()
        .str.upper()
        == faculty_id
    ]

    if matches.empty:
        return []

    if "course_id" not in matches.columns:
        return []

    return [
        str(course)
        for course in matches["course_id"]
        .dropna()
        .unique()
    ]


# =========================================================
# EXTRACT FACULTY ID
# =========================================================

def extract_faculty_id(question, timetable):

    if "faculty_id" not in timetable.columns:
        return None

    question_upper = question.upper()

    faculty_ids = (
        timetable["faculty_id"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    for faculty_id in faculty_ids:

        if faculty_id in question_upper:
            return faculty_id

    return None


# =========================================================
# TIME CONVERSION
# =========================================================

def time_to_minutes(value):

    value = str(value).strip()

    try:

        parts = value.split(":")

        hours = int(parts[0])
        minutes = int(parts[1])

        return hours * 60 + minutes

    except Exception:

        return None


# =========================================================
# EXACT CONFLICT DETECTION
# =========================================================

def detect_conflicts(timetable):
    """
    Detect overlapping classes for the same faculty,
    section, or room on the same day.
    """

    required_columns = [
        "day",
        "start_time",
        "end_time",
    ]

    if not all(
        column in timetable.columns
        for column in required_columns
    ):
        return []

    scheduled = timetable[
        timetable["day"].astype(str).str.upper()
        != "UNSCHEDULED"
    ].copy()

    conflicts = []

    # Convert times into minutes
    scheduled["_start"] = scheduled[
        "start_time"
    ].apply(time_to_minutes)

    scheduled["_end"] = scheduled[
        "end_time"
    ].apply(time_to_minutes)

    # -----------------------------------------------------
    # CHECK EACH DAY
    # -----------------------------------------------------

    for day, day_data in scheduled.groupby("day"):

        rows = list(day_data.iterrows())

        for i in range(len(rows)):

            index_a, class_a = rows[i]

            for j in range(i + 1, len(rows)):

                index_b, class_b = rows[j]

                start_a = class_a["_start"]
                end_a = class_a["_end"]

                start_b = class_b["_start"]
                end_b = class_b["_end"]

                if (
                    start_a is None
                    or end_a is None
                    or start_b is None
                    or end_b is None
                ):
                    continue

                # Check time overlap
                overlap = (
                    start_a < end_b
                    and start_b < end_a
                )

                if not overlap:
                    continue

                # -------------------------------------------------
                # FACULTY CONFLICT
                # -------------------------------------------------

                if (
                    "faculty_id" in scheduled.columns
                    and str(class_a["faculty_id"])
                    == str(class_b["faculty_id"])
                ):

                    conflicts.append(
                        f"Faculty conflict: "
                        f"{class_a.get('faculty_id')} "
                        f"has overlapping classes on {day}."
                    )

                # -------------------------------------------------
                # SECTION CONFLICT
                # -------------------------------------------------

                if (
                    "section_id" in scheduled.columns
                    and str(class_a["section_id"])
                    == str(class_b["section_id"])
                ):

                    conflicts.append(
                        f"Section conflict: "
                        f"{class_a.get('section_id')} "
                        f"has overlapping classes on {day}."
                    )

                # -------------------------------------------------
                # ROOM CONFLICT
                # -------------------------------------------------

                if (
                    "room_id" in scheduled.columns
                    and str(class_a["room_id"])
                    == str(class_b["room_id"])
                ):

                    conflicts.append(
                        f"Room conflict: "
                        f"{class_a.get('room_id')} "
                        f"has overlapping classes on {day}."
                    )

    # Remove duplicates
    return list(dict.fromkeys(conflicts))


# =========================================================
# SMARTSCHED AI ASSISTANT
# =========================================================

def ask_timetable_assistant(question, timetable):

    scheduled, factual_summary = build_timetable_summary(
        timetable
    )

    question_lower = question.lower()

    # =====================================================
    # EXACT FACULTY COURSE QUERY
    # =====================================================

    course_query_words = [
        "which courses",
        "what courses",
        "courses handled",
        "courses taught",
        "teaches",
        "teaching",
    ]

    faculty_id = extract_faculty_id(
        question,
        scheduled
    )

    is_course_query = (
        faculty_id is not None
        and any(
            word in question_lower
            for word in course_query_words
        )
    )

    if is_course_query:

        courses = find_courses_by_faculty(
            scheduled,
            faculty_id
        )

        if courses:

            exact_result = (
                f"Faculty {faculty_id} handles "
                f"{len(courses)} course(s): "
                f"{', '.join(courses)}."
            )

        else:

            exact_result = (
                f"No courses were found for "
                f"faculty {faculty_id}."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are SmartSched AI.

Python has already calculated the exact answer.

Do not recalculate.
Do not add information.
Do not remove information.
Do not invent courses.

Explain the provided result clearly.
                    """,
                ),
                (
                    "human",
                    """
USER QUESTION:

{question}

EXACT PYTHON RESULT:

{result}

Give a short answer using only the exact result.
                    """,
                ),
            ]
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "question": question,
                "result": exact_result,
            }
        )

        return response.content

    # =====================================================
    # EXACT CONFLICT QUERY
    # =====================================================

    conflict_words = [
        "conflict",
        "conflicts",
        "clash",
        "clashes",
        "overlap",
        "overlapping",
    ]

    is_conflict_query = any(
        word in question_lower
        for word in conflict_words
    )

    if is_conflict_query:

        conflicts = detect_conflicts(
            scheduled
        )

        if conflicts:

            exact_result = (
                f"YES. {len(conflicts)} timetable "
                f"conflict(s) were detected:\n"
                + "\n".join(
                    f"- {conflict}"
                    for conflict in conflicts
                )
            )

        else:

            exact_result = (
                "NO. No faculty, section, or room "
                "overlapping conflicts were detected "
                "in the generated timetable."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are SmartSched AI.

Python has performed an actual timetable conflict
check.

The Python result is authoritative.

Do not guess whether conflicts exist.
Do not invent conflicts.
Do not remove conflicts from the result.

Explain the result clearly and briefly.
                    """,
                ),
                (
                    "human",
                    """
USER QUESTION:

{question}

EXACT PYTHON CONFLICT CHECK:

{result}

Give a concise answer using only this result.
                    """,
                ),
            ]
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "question": question,
                "result": exact_result,
            }
        )

        return response.content

    # =====================================================
    # GENERAL AI QUESTION
    # =====================================================

    timetable_text = scheduled.to_string(
        index=False
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are SmartSched AI, an intelligent college
timetable assistant.

Use the calculated Python statistics as the
source of truth.

Rules:
- Never invent information.
- Never recalculate numerical values.
- Never contradict Python statistics.
- Keep answers concise.
- If information is unavailable, say so.
                """,
            ),
            (
                "human",
                """
CALCULATED STATISTICS:

{summary}

CURRENT TIMETABLE:

{timetable}

USER QUESTION:

{question}

Answer using the calculated statistics.
                """,
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "summary": factual_summary,
            "timetable": timetable_text,
            "question": question,
        }
    )

    return response.content