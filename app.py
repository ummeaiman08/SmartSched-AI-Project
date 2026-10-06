
import io
import html as html_lib
import textwrap
from datetime import datetime

import pandas as pd
import streamlit as st

from src.data_loader import load_data
from src.scheduler import generate_schedule
from src.optimizer import optimize_schedule
from src.timetable import prepare_timetable
from src.ai_assistant import ask_timetable_assistant
from src.visualizations import (
    faculty_workload_chart,
    room_utilization_chart,
    daily_class_chart,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartSched AI",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# HELPERS
# =========================================================

def render_html(html: str) -> None:
    # Streamlit's Markdown parser can terminate an HTML block when it
    # encounters blank lines, causing the remaining indented HTML to
    # appear as literal source code. Remove empty lines before rendering.
    cleaned = textwrap.dedent(html)
    cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)



def generate_current_timetable(data):
    """Run the existing scheduler/optimizer and store the result."""
    schedule = generate_schedule(data)
    timetable, score = optimize_schedule(schedule)
    timetable = prepare_timetable(timetable)
    st.session_state["timetable"] = timetable
    st.session_state["score"] = score
    st.session_state["generated_at"] = datetime.now().strftime("%d %b %Y · %I:%M %p")
    return timetable, score


def _time_sort_key(value):
    value = str(value).strip()
    try:
        hour, minute = value.split(":")[:2]
        return int(hour) * 60 + int(minute)
    except (ValueError, AttributeError):
        return 9999


def _minutes(value):
    value = str(value).strip()
    try:
        hour, minute = value.split(":")[:2]
        return int(hour) * 60 + int(minute)
    except (ValueError, AttributeError):
        return None


def analyze_schedule_health(timetable, data=None):
    """Display-only health analysis; does not alter scheduler decisions."""
    if timetable is None or len(timetable) == 0:
        return {
            "scheduled": 0, "unscheduled": 0, "coverage": 0,
            "faculty_conflicts": 0, "room_conflicts": 0,
            "section_conflicts": 0, "capacity_violations": 0,
        }

    frame = timetable.copy()
    scheduled_mask = (
        frame["day"].astype(str).str.upper() != "UNSCHEDULED"
        if "day" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    scheduled = frame.loc[scheduled_mask].copy()
    scheduled_count = int(len(scheduled))
    unscheduled = int((~scheduled_mask).sum())
    total_count = int(len(frame))
    coverage = round((scheduled_count / total_count) * 100) if total_count else 0

    conflicts = {"faculty_conflicts": 0, "room_conflicts": 0, "section_conflicts": 0}

    if not scheduled.empty and {"day", "start_time", "end_time"}.issubset(scheduled.columns):
        for field, result_key in [
            ("faculty_id", "faculty_conflicts"),
            ("room_id", "room_conflicts"),
            ("section_id", "section_conflicts"),
        ]:
            if field not in scheduled.columns:
                continue

            clean = scheduled[["day", "start_time", "end_time", field]].dropna().copy()
            clean[field] = clean[field].astype(str).str.strip()
            clean = clean[clean[field] != ""]
            clashes = 0

            for _, group in clean.groupby(["day", field], dropna=False):
                intervals = []
                for _, row in group.iterrows():
                    s = _minutes(row["start_time"])
                    e = _minutes(row["end_time"])
                    if s is not None and e is not None:
                        intervals.append((s, e))
                for i in range(len(intervals)):
                    for j in range(i + 1, len(intervals)):
                        a_start, a_end = intervals[i]
                        b_start, b_end = intervals[j]
                        if max(a_start, b_start) < min(a_end, b_end):
                            clashes += 1
            conflicts[result_key] = clashes

    capacity_violations = 0
    if (
        data and "rooms" in data and "sections" in data
        and "section_id" in scheduled.columns and "room_id" in scheduled.columns
    ):
        rooms = data["rooms"].copy()
        sections = data["sections"].copy()

        room_id_col = next((c for c in ["room_id", "room_code", "id"] if c in rooms.columns), None)
        room_capacity_col = next(
            (c for c in ["capacity", "max_capacity", "room_capacity", "seats"] if c in rooms.columns),
            None,
        )
        section_id_col = next((c for c in ["section_id", "section_code", "id"] if c in sections.columns), None)
        student_col = next(
            (c for c in ["student_count", "students", "strength", "student_strength", "num_students"]
             if c in sections.columns),
            None,
        )

        if all([room_id_col, room_capacity_col, section_id_col, student_col]):
            room_map = dict(zip(rooms[room_id_col].astype(str), rooms[room_capacity_col]))
            section_map = dict(zip(sections[section_id_col].astype(str), sections[student_col]))

            for _, row in scheduled.iterrows():
                room_capacity = room_map.get(str(row["room_id"]))
                section_size = section_map.get(str(row["section_id"]))
                try:
                    if pd.notna(room_capacity) and pd.notna(section_size):
                        if float(section_size) > float(room_capacity):
                            capacity_violations += 1
                except (TypeError, ValueError):
                    pass

    return {
        "scheduled": scheduled_count,
        "unscheduled": unscheduled,
        "coverage": coverage,
        **conflicts,
        "capacity_violations": capacity_violations,
    }


def validate_dataset(data):
    """Best-effort source-data diagnostics that do not reject harmless empty rows."""
    issues = []
    warnings = []

    id_candidates = {
        "courses": ["course_id", "course_code", "id"],
        "faculty": ["faculty_id", "faculty_code", "id"],
        "rooms": ["room_id", "room_code", "id"],
        "sections": ["section_id", "section_code", "id"],
        "timeslots": ["slot_id", "timeslot_id", "time_slot_id", "id"],
    }

    id_maps = {}

    for name, frame in data.items():
        if frame is None or len(frame) == 0:
            issues.append(f"{name}: dataset is empty.")
            continue

        # Ignore completely empty editor/CSV rows during validation.
        usable = frame.dropna(how="all").copy()

        if usable.empty:
            issues.append(f"{name}: dataset contains no usable records.")
            continue

        id_col = next(
            (c for c in id_candidates.get(name, []) if c in usable.columns),
            None,
        )

        if id_col:
            values = usable[id_col].fillna("").astype(str).str.strip()
            non_blank = values[values != ""]
            blank = int((values == "").sum())
            duplicate = int(non_blank.duplicated(keep=False).sum())

            id_maps[name] = (id_col, set(non_blank))

            # Only a populated record with a missing ID is a real problem.
            if blank:
                issues.append(
                    f"{name}: {blank} populated record(s) have a blank identifier in {id_col}."
                )

            if duplicate:
                issues.append(
                    f"{name}: duplicate identifier value(s) detected in {id_col}."
                )
        else:
            warnings.append(f"{name}: no standard identifier column detected.")

        empty_rows = int(frame.isna().all(axis=1).sum())
        if empty_rows:
            warnings.append(
                f"{name}: {empty_rows} completely empty row(s) will be ignored."
            )

    targets = {
        "course_id": "courses",
        "faculty_id": "faculty",
        "room_id": "rooms",
        "section_id": "sections",
        "slot_id": "timeslots",
        "timeslot_id": "timeslots",
        "time_slot_id": "timeslots",
    }

    for name, frame in data.items():
        usable = frame.dropna(how="all")
        for column in usable.columns:
            target = targets.get(str(column).lower())
            if not target or target not in id_maps:
                continue

            allowed = id_maps[target][1]
            values = usable[column].dropna().astype(str).str.strip()
            orphan = int((~values.isin(allowed) & (values != "")).sum())

            if orphan:
                issues.append(
                    f"{name}: {orphan} reference value(s) in {column} "
                    f"do not match {target} identifiers."
                )

    return {"issues": issues, "warnings": warnings}


def get_scheduled_frame(timetable):
    if timetable is None or len(timetable) == 0:
        return pd.DataFrame()
    if "day" not in timetable.columns:
        return timetable.copy()
    return timetable[
        timetable["day"].astype(str).str.upper() != "UNSCHEDULED"
    ].copy()


def get_analytics_summary(timetable):
    frame = get_scheduled_frame(timetable)
    if frame.empty:
        return {
            "busiest_day": "—", "busiest_time": "—",
            "top_faculty": "—", "top_room": "—",
            "avg_per_day": 0, "days_active": 0,
        }

    busiest_day = (
        frame["day"].astype(str).value_counts().index[0]
        if "day" in frame.columns else "—"
    )
    busiest_time = (
        frame["start_time"].astype(str).value_counts().index[0]
        if "start_time" in frame.columns else "—"
    )
    top_faculty = (
        frame["faculty_id"].astype(str).value_counts().index[0]
        if "faculty_id" in frame.columns else "—"
    )
    top_room = (
        frame["room_id"].astype(str).value_counts().index[0]
        if "room_id" in frame.columns else "—"
    )
    days_active = int(frame["day"].astype(str).nunique()) if "day" in frame.columns else 0

    return {
        "busiest_day": busiest_day,
        "busiest_time": busiest_time,
        "top_faculty": top_faculty,
        "top_room": top_room,
        "avg_per_day": round(len(frame) / days_active, 1) if days_active else 0,
        "days_active": days_active,
    }


def apply_timetable_filters(timetable):
    if timetable is None or len(timetable) == 0:
        return timetable

    scheduled = get_scheduled_frame(timetable)
    with st.expander("Filter & explore timetable", expanded=False):
        f1, f2, f3, f4, f5 = st.columns(5)

        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        present_days = [str(v) for v in scheduled["day"].dropna().unique()] if "day" in scheduled.columns else []
        ordered_days = [d for d in days_order if d in present_days] + [
            d for d in present_days if d not in days_order
        ]

        with f1:
            selected_day = st.selectbox("Day", ["All"] + ordered_days, key="filter_day")
        with f2:
            faculty_values = ["All"] + sorted(
                scheduled["faculty_id"].astype(str).dropna().unique().tolist()
            ) if "faculty_id" in scheduled.columns else ["All"]
            selected_faculty = st.selectbox("Faculty", faculty_values, key="filter_faculty")
        with f3:
            room_values = ["All"] + sorted(
                scheduled["room_id"].astype(str).dropna().unique().tolist()
            ) if "room_id" in scheduled.columns else ["All"]
            selected_room = st.selectbox("Room", room_values, key="filter_room")
        with f4:
            section_values = ["All"] + sorted(
                scheduled["section_id"].astype(str).dropna().unique().tolist()
            ) if "section_id" in scheduled.columns else ["All"]
            selected_section = st.selectbox("Section", section_values, key="filter_section")
        with f5:
            course_search = st.text_input(
                "Course search", placeholder="Course / ID", key="filter_course"
            )

        filtered = timetable.copy()

        if selected_day != "All" and "day" in filtered.columns:
            filtered = filtered[filtered["day"].astype(str) == selected_day]
        if selected_faculty != "All" and "faculty_id" in filtered.columns:
            filtered = filtered[filtered["faculty_id"].astype(str) == selected_faculty]
        if selected_room != "All" and "room_id" in filtered.columns:
            filtered = filtered[filtered["room_id"].astype(str) == selected_room]
        if selected_section != "All" and "section_id" in filtered.columns:
            filtered = filtered[filtered["section_id"].astype(str) == selected_section]
        if course_search.strip():
            q = course_search.strip().lower()
            mask = pd.Series(False, index=filtered.index)
            for column in ["course_id", "course_name"]:
                if column in filtered.columns:
                    mask |= filtered[column].astype(str).str.lower().str.contains(q, na=False)
            filtered = filtered[mask]

        st.caption(f"Showing {len(filtered)} of {len(timetable)} generated assignments.")
        return filtered

    return timetable


def make_excel_bytes(timetable):
    import xlsxwriter

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        timetable.to_excel(writer, sheet_name="Timetable", index=False)
        wb = writer.book
        header = wb.add_format({
            "bold": True, "font_color": "#FFFFFF", "bg_color": "#172333",
            "align": "center", "valign": "vcenter",
        })
        body = wb.add_format({"font_color": "#263443", "valign": "vcenter"})

        ws = writer.sheets["Timetable"]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(timetable)), max(0, len(timetable.columns) - 1))
        for idx, col in enumerate(timetable.columns):
            ws.write(0, idx, col, header)
            ws.set_column(idx, idx, min(max(len(str(col)) + 4, 12), 28), body)

        health = analyze_schedule_health(timetable)
        summary = pd.DataFrame([
            ["Generated", st.session_state.get("generated_at", "—")],
            ["Quality score", st.session_state.get("score", 0)],
            ["Scheduled", health["scheduled"]],
            ["Unscheduled", health["unscheduled"]],
            ["Coverage", f'{health["coverage"]}%'],
            ["Faculty overlaps", health["faculty_conflicts"]],
            ["Room overlaps", health["room_conflicts"]],
            ["Section overlaps", health["section_conflicts"]],
            ["Capacity violations", health["capacity_violations"]],
        ], columns=["Metric", "Value"])
        summary.to_excel(writer, sheet_name="Summary", index=False)
        sws = writer.sheets["Summary"]
        sws.set_column(0, 0, 24, body)
        sws.set_column(1, 1, 26, body)
        sws.write(0, 0, "Metric", header)
        sws.write(0, 1, "Value", header)

        for sheet_name, column in [
            ("By Section", "section_id"),
            ("By Faculty", "faculty_id"),
            ("By Room", "room_id"),
        ]:
            if column in timetable.columns:
                grouped = (
                    get_scheduled_frame(timetable)[column]
                    .astype(str)
                    .value_counts()
                    .rename_axis(column)
                    .reset_index(name="Classes")
                )
                grouped.to_excel(writer, sheet_name=sheet_name, index=False)
                nws = writer.sheets[sheet_name]
                for idx, col in enumerate(grouped.columns):
                    nws.write(0, idx, col, header)
                    nws.set_column(idx, idx, 22 if idx == 0 else 12, body)

    output.seek(0)
    return output.getvalue()


def make_pdf_bytes(timetable):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(A4),
        rightMargin=9 * mm, leftMargin=9 * mm,
        topMargin=10 * mm, bottomMargin=10 * mm,
        title="SmartSched AI Timetable",
    )

    styles = getSampleStyleSheet()
    title = styles["Title"]
    title.fontName = "Helvetica-Bold"
    title.fontSize = 17
    title.textColor = colors.HexColor("#172333")
    body = styles["Normal"]
    body.fontName = "Helvetica"
    body.fontSize = 7
    body.leading = 8
    body.textColor = colors.HexColor("#334455")

    story = [
        Paragraph("SmartSched AI · Optimized Timetable", title),
        Paragraph(
            f'Generated: {st.session_state.get("generated_at", "—")} · '
            f'Quality: {st.session_state.get("score", 0)}/100',
            body,
        ),
        Spacer(1, 5 * mm),
    ]

    fields = [
        ("Course", "course_name"), ("ID", "course_id"),
        ("Faculty", "faculty_id"), ("Section", "section_id"),
        ("Room", "room_id"), ("Day", "day"),
        ("Start", "start_time"), ("End", "end_time"),
    ]
    rows = [[name for name, _ in fields]]
    for _, row in timetable.iterrows():
        rows.append([str(row.get(key, "")) for _, key in fields])

    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#172333")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 6.7),
        ("TEXTCOLOR", (0,1), (-1,-1), colors.HexColor("#314153")),
        ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#CFD8E0")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [
            colors.HexColor("#F8FAFC"), colors.HexColor("#EDF2F6")
        ]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    doc.build(story)
    output.seek(0)
    return output.getvalue()


def make_print_html(timetable):
    columns = [
        c for c in [
            "course_name", "course_id", "faculty_id", "section_id",
            "room_id", "day", "start_time", "end_time"
        ] if c in timetable.columns
    ]
    labels = {
        "course_name": "Course", "course_id": "ID", "faculty_id": "Faculty",
        "section_id": "Section", "room_id": "Room", "day": "Day",
        "start_time": "Start", "end_time": "End",
    }

    head = "".join(
        f"<th>{html_lib.escape(labels.get(c, c))}</th>" for c in columns
    )
    body_rows = []
    for _, row in timetable.fillna("").iterrows():
        body_rows.append(
            "<tr>" + "".join(
                f'<td>{html_lib.escape(str(row.get(c, "")))}</td>'
                for c in columns
            ) + "</tr>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>SmartSched AI Timetable</title>
<style>
@page {{ size:A4 landscape; margin:12mm; }}
body {{ font-family:Arial,sans-serif; color:#172333; }}
.header {{ display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:16px; }}
h1 {{ margin:0; font-size:22px; }}
.meta {{ color:#64748b; font-size:10px; }}
table {{ width:100%; border-collapse:collapse; font-size:9px; }}
th {{ background:#172333; color:white; padding:7px; text-align:left; }}
td {{ border:1px solid #d8e0e8; padding:6px; }}
tr:nth-child(even) td {{ background:#f3f6f8; }}
@media print {{ .note {{ display:none; }} }}
</style></head><body>
<div class="header">
<div><h1>SmartSched AI · Optimized Timetable</h1>
<div class="meta">Generated: {html_lib.escape(st.session_state.get("generated_at", "—"))}</div></div>
<div class="meta">Quality: {st.session_state.get("score", 0)}/100</div>
</div>
<table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>
<div class="note" style="margin-top:12px;color:#64748b;font-size:9px;">
Open in a browser and use Print → Save as PDF.
</div>
</body></html>"""
    return html.encode("utf-8")


def safe_chart(label, fn, timetable):
    try:
        st.plotly_chart(fn(timetable), use_container_width=True)
    except Exception as error:
        st.warning(f"{label} is unavailable for the current data: {error}")


def render_timetable_grid(timetable):
    """Render a visual weekly timetable without changing scheduling logic."""
    if timetable is None or len(timetable) == 0:
        return

    required = {"day", "start_time", "end_time"}
    if not required.issubset(set(timetable.columns)):
        return

    scheduled = timetable[
        timetable["day"].astype(str).str.upper() != "UNSCHEDULED"
    ].copy()

    if scheduled.empty:
        render_html(
            """
            <div class="visual-empty">
                No scheduled classes are available for the visual grid.
            </div>
            """
        )
        return

    preferred_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    present_days = [str(x) for x in scheduled["day"].dropna().unique().tolist()]
    ordered_days = [d for d in preferred_days if d in present_days]
    ordered_days += [d for d in present_days if d not in ordered_days]

    present_times = [str(x) for x in scheduled["start_time"].dropna().unique().tolist()]
    ordered_times = sorted(present_times, key=_time_sort_key)

    # Build lookup: (day, start_time) -> list of classes.
    lookup = {}
    for _, row in scheduled.iterrows():
        day = str(row.get("day", ""))
        start = str(row.get("start_time", ""))
        lookup.setdefault((day, start), []).append(row)

    def esc(value):
        return html_lib.escape("" if value is None else str(value))

    grid_html = [
        '<div class="visual-timetable">',
        '<div class="visual-timetable-head">',
        '<div class="grid-title-block">',
        '<div class="table-kicker">WEEKLY VIEW</div>',
        '<div class="grid-title">Schedule at a glance</div>',
        '<div class="grid-subtitle">A visual view of every scheduled period. Hover a class for details.</div>',
        '</div>',
        f'<div class="grid-count">{len(scheduled)} CLASSES</div>',
        '</div>',
        '<div class="timetable-scroll">',
        '<div class="tt-grid" style="grid-template-columns:72px repeat('
        + str(len(ordered_days))
        + ', minmax(145px, 1fr));">',
        '<div class="tt-cell tt-corner">TIME</div>',
    ]

    for day in ordered_days:
        grid_html.append(
            f'<div class="tt-cell tt-day"><span>{esc(day[:3].upper())}</span><small>{esc(day)}</small></div>'
        )

    for time in ordered_times:
        grid_html.append(
            f'<div class="tt-cell tt-time">{esc(time)}</div>'
        )

        for day in ordered_days:
            rows = lookup.get((day, time), [])
            if not rows:
                grid_html.append('<div class="tt-cell tt-empty"><span>—</span></div>')
                continue

            cards = ['<div class="tt-cell tt-filled">']

            for row in rows:
                course_id = row.get("course_id", "")
                course_name = row.get("course_name", "")
                faculty_id = row.get("faculty_id", "")
                section_id = row.get("section_id", "")
                room_id = row.get("room_id", "")
                end_time = row.get("end_time", "")

                display_course = course_name if str(course_name).strip() else course_id
                title = " · ".join(
                    [
                        str(display_course),
                        str(faculty_id),
                        str(section_id),
                        str(room_id),
                        f"{time}–{end_time}",
                    ]
                )

                cards.append(
                    f"""
                    <div class="tt-card" title="{esc(title)}">
                        <div class="tt-course">{esc(display_course)}</div>
                        <div class="tt-code">{esc(course_id)}</div>
                        <div class="tt-meta">
                            <span>{esc(faculty_id)}</span>
                            <span>{esc(room_id)}</span>
                        </div>
                        <div class="tt-section">{esc(section_id)} · {esc(time)}–{esc(end_time)}</div>
                    </div>
                    """
                )

            cards.append("</div>")
            grid_html.append("\n".join(cards))

    grid_html.extend(["</div>", "</div>", "</div>"])

    render_html("\n".join(grid_html))


# =========================================================
# SIGNATURE UI SYSTEM
# Midnight Blueprint / Glass Academic OS
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #060913;
        --bg-2: #0a0f1b;
        --panel: rgba(12, 18, 31, .78);
        --panel-2: rgba(15, 22, 38, .86);
        --panel-3: rgba(10, 15, 27, .92);

        --line: rgba(135, 160, 190, .13);
        --line-strong: rgba(145, 173, 205, .23);

        --text: #f3f7fb;
        --text-2: #c8d3df;
        --muted: #8190a2;
        --dim: #5d6b7d;

        --accent: #8bd4ff;
        --accent-2: #8c7bff;
        --accent-soft: rgba(139, 212, 255, .12);

        --green: #68e0af;
        --amber: #f3c66b;
    }

    * { box-sizing: border-box; }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(900px 550px at 78% 8%, rgba(75, 87, 196, .18), transparent 62%),
            radial-gradient(780px 520px at 18% 24%, rgba(42, 168, 205, .12), transparent 64%),
            linear-gradient(180deg, #060913 0%, #070b14 48%, #050810 100%);
        color: var(--text);
    }

    body { overflow-x: hidden; }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: .22;
        background-image:
            linear-gradient(rgba(151,176,205,.055) 1px, transparent 1px),
            linear-gradient(90deg, rgba(151,176,205,.055) 1px, transparent 1px);
        background-size: 52px 52px;
        mask-image: linear-gradient(to bottom, rgba(0,0,0,.92), rgba(0,0,0,.2) 72%, transparent);
        -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,.92), rgba(0,0,0,.2) 72%, transparent);
    }

    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(350px 180px at 55% 22%, rgba(121, 88, 255, .08), transparent 70%),
            radial-gradient(280px 160px at 90% 70%, rgba(69, 198, 231, .06), transparent 70%);
        animation: ambientDrift 11s ease-in-out infinite alternate;
    }

    @keyframes ambientDrift {
        from { transform: translate3d(-10px, -5px, 0) scale(1); opacity: .82; }
        to   { transform: translate3d(15px, 8px, 0) scale(1.04); opacity: 1; }
    }

    [data-testid="stHeader"] {
        background: rgba(6, 9, 19, .66) !important;
        backdrop-filter: blur(16px);
    }

    #MainMenu, footer { visibility: hidden; }

    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 1480px;
        padding: 1.45rem 2.2rem 4rem;
    }

    /* ---------------- Sidebar ---------------- */

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(260px 190px at 20% 0%, rgba(87, 105, 197, .10), transparent 70%),
            linear-gradient(180deg, #080d17 0%, #070a12 100%);
        border-right: 1px solid rgba(130, 156, 185, .13);
    }

    section[data-testid="stSidebar"] > div {
        padding: 1rem .75rem 1rem;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(130, 156, 185, .10);
    }

    .brand-row {
        display:flex;
        align-items:center;
        gap:12px;
        padding:7px 8px 17px;
    }

    .brand-mark {
        position:relative;
        width:42px;
        height:42px;
        flex:0 0 42px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:13px;
        border:1px solid rgba(139,212,255,.24);
        background:
            radial-gradient(circle at 30% 25%, rgba(139,212,255,.18), transparent 45%),
            linear-gradient(145deg, rgba(23,35,61,.9), rgba(12,18,31,.95));
        color:#eaf7ff;
        font-size:12px;
        font-weight:900;
        letter-spacing:-.4px;
        box-shadow: 0 0 0 5px rgba(139,212,255,.025), 0 12px 30px rgba(0,0,0,.22);
    }

    .brand-mark::after {
        content:"";
        position:absolute;
        width:5px;
        height:5px;
        border-radius:50%;
        right:7px;
        top:7px;
        background:var(--green);
        box-shadow:0 0 12px rgba(104,224,175,.9);
    }

    .brand-name {
        color:#f4f8fc;
        font-size:17px;
        font-weight:850;
        letter-spacing:-.35px;
    }

    .brand-sub {
        color:#65758a;
        font-size:10px;
        margin-top:2px;
        letter-spacing:.2px;
    }

    .side-heading {
        color:#56667a;
        font-size:9px;
        font-weight:900;
        letter-spacing:1.7px;
        text-transform:uppercase;
        margin:16px 8px 8px;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap:5px;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        padding:9px 9px;
        border-radius:10px;
        border:1px solid transparent;
        transition:all .18s ease;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background:rgba(139,212,255,.045);
        border-color:rgba(139,212,255,.08);
    }

    section[data-testid="stSidebar"] .stRadio label {
        color:#8b9aab !important;
        font-size:12px;
        font-weight:700;
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background:
            linear-gradient(90deg, rgba(139,212,255,.11), rgba(139,212,255,.025));
        border-color:rgba(139,212,255,.16);
        box-shadow: inset 3px 0 0 var(--accent);
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) p,
    section[data-testid="stSidebar"] .stRadio label:has(input:checked) div {
        color:#f2f7fb !important;
    }

    .side-data-card {
        margin-top:2px;
        padding:11px 10px;
        border:1px solid rgba(132,157,188,.10);
        border-radius:12px;
        background:rgba(10,15,26,.56);
    }

    .side-data-row {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:5px 1px;
        color:#738296;
        font-size:10px;
        border-bottom:1px solid rgba(130,156,185,.055);
    }

    .side-data-row:last-child { border-bottom:none; }

    .side-data-row span:last-child {
        color:#d4deea;
        font-weight:850;
        font-variant-numeric:tabular-nums;
    }

    .side-status {
        margin:14px 3px 0;
        padding:10px 11px;
        border-radius:11px;
        background:linear-gradient(90deg, rgba(69,202,158,.08), rgba(69,202,158,.02));
        border:1px solid rgba(76,214,169,.13);
        color:#99dabe;
        font-size:10px;
    }

    .status-dot {
        display:inline-block;
        width:6px;
        height:6px;
        border-radius:50%;
        background:var(--green);
        margin-right:7px;
        vertical-align:1px;
        box-shadow:0 0 10px rgba(104,224,175,.7);
    }

    /* ---------------- Global typography ---------------- */

    .eyebrow {
        color:#607187;
        font-size:9px;
        font-weight:900;
        letter-spacing:1.8px;
        text-transform:uppercase;
    }

    .section-title {
        color:#f0f5f9;
        font-size:24px;
        font-weight:850;
        letter-spacing:-.75px;
        margin-top:3px;
    }

    .section-desc {
        color:#7e8da0;
        font-size:12px;
        margin-top:4px;
        line-height:1.6;
    }

    /* ---------------- Signature hero ---------------- */

    .hero {
        position:relative;
        overflow:hidden;
        min-height:300px;
        margin-bottom:25px;
        padding:34px 36px;
        border-radius:22px;
        border:1px solid rgba(137,164,196,.18);
        background:
            radial-gradient(430px 260px at 84% 48%, rgba(101,89,225,.17), transparent 67%),
            radial-gradient(420px 270px at 18% 88%, rgba(53,187,218,.10), transparent 72%),
            linear-gradient(135deg, rgba(14,21,36,.95), rgba(9,14,25,.88));
        box-shadow:
            0 26px 80px rgba(0,0,0,.28),
            inset 0 1px 0 rgba(255,255,255,.025);
    }

    .hero-grid {
        position:absolute;
        inset:0;
        opacity:.42;
        pointer-events:none;
        background-image:
            linear-gradient(rgba(154,180,210,.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(154,180,210,.07) 1px, transparent 1px);
        background-size:36px 36px;
        mask-image:linear-gradient(105deg, rgba(0,0,0,.95), transparent 66%);
        -webkit-mask-image:linear-gradient(105deg, rgba(0,0,0,.95), transparent 66%);
    }

    .hero-scan {
        position:absolute;
        left:-10%;
        right:-10%;
        height:1px;
        top:62%;
        background:linear-gradient(90deg, transparent, rgba(139,212,255,.20), transparent);
        box-shadow:0 0 22px rgba(139,212,255,.10);
        animation:scan 7s linear infinite;
        pointer-events:none;
    }

    @keyframes scan {
        0% { transform:translateY(-90px); opacity:0; }
        16% { opacity:1; }
        84% { opacity:1; }
        100% { transform:translateY(105px); opacity:0; }
    }

    .hero-ring {
        position:absolute;
        width:440px;
        height:440px;
        right:-165px;
        top:-170px;
        border-radius:50%;
        border:1px solid rgba(139,212,255,.11);
        box-shadow:
            0 0 0 32px rgba(139,212,255,.028),
            0 0 0 65px rgba(139,212,255,.018),
            0 0 0 102px rgba(139,212,255,.012);
        pointer-events:none;
    }

    .hero-content {
        position:relative;
        z-index:2;
        max-width:860px;
    }

    .hero-badge {
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding:7px 10px;
        border-radius:999px;
        color:#a9c2d4;
        background:rgba(139,212,255,.055);
        border:1px solid rgba(139,212,255,.13);
        font-size:9px;
        font-weight:900;
        letter-spacing:1.2px;
    }

    .hero-badge .dot {
        width:5px;
        height:5px;
        border-radius:50%;
        background:var(--accent);
        box-shadow:0 0 11px rgba(139,212,255,.85);
    }

    .hero-title {
        margin-top:16px;
        color:#f4f8fb;
        font-size:52px;
        line-height:.98;
        letter-spacing:-2.9px;
        font-weight:900;
    }

    .hero-title span {
        background:linear-gradient(90deg, #eaf8ff 0%, #91dbff 44%, #9f92ff 100%);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
    }

    .hero-description {
        max-width:760px;
        margin-top:14px;
        color:#91a0b2;
        font-size:14px;
        line-height:1.72;
    }

    .hero-meta {
        display:flex;
        flex-wrap:wrap;
        gap:10px;
        margin-top:20px;
    }

    .hero-meta-item {
        padding:8px 10px;
        border:1px solid rgba(139,157,183,.11);
        border-radius:9px;
        background:rgba(8,13,23,.44);
        color:#68798c;
        font-size:9px;
        text-transform:uppercase;
        letter-spacing:1px;
        font-weight:850;
    }

    .hero-meta-item b {
        color:#c8d6e4;
        font-weight:850;
        margin-left:4px;
    }

    /* Scheduler visualization on hero right */

    .schedule-art {
        position:absolute;
        width:350px;
        height:220px;
        right:35px;
        top:44px;
        z-index:2;
        opacity:.98;
    }

    .schedule-shell {
        position:absolute;
        inset:0;
        border:1px solid rgba(139,212,255,.13);
        border-radius:18px;
        background:linear-gradient(180deg, rgba(9,15,27,.52), rgba(8,12,21,.30));
        box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
        backdrop-filter:blur(5px);
    }

    .schedule-head {
        position:absolute;
        left:16px;
        right:16px;
        top:13px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        color:#6e8094;
        font-size:8px;
        font-weight:900;
        letter-spacing:1.2px;
        text-transform:uppercase;
    }

    .schedule-live {
        color:#7ce0b8;
        letter-spacing:.8px;
    }

    .schedule-grid {
        position:absolute;
        left:16px;
        right:16px;
        top:42px;
        bottom:14px;
        display:grid;
        grid-template-columns:56px repeat(4, 1fr);
        grid-template-rows:24px repeat(4, 1fr);
        gap:5px;
    }

    .sg {
        border:1px solid rgba(136,161,190,.075);
        border-radius:6px;
        background:rgba(255,255,255,.012);
    }

    .sg.head {
        display:flex;
        align-items:center;
        justify-content:center;
        color:#66788d;
        font-size:7px;
        font-weight:800;
        border-color:transparent;
        background:transparent;
    }

    .sg.time {
        display:flex;
        align-items:center;
        justify-content:center;
        color:#607084;
        font-size:7px;
        font-weight:800;
        background:rgba(255,255,255,.01);
    }

    .sg.block {
        position:relative;
        background:linear-gradient(135deg, rgba(139,212,255,.12), rgba(131,108,255,.10));
        border-color:rgba(139,212,255,.18);
        box-shadow:inset 0 0 18px rgba(139,212,255,.025);
    }

    .sg.block::after {
        content:"";
        position:absolute;
        left:7px;
        right:7px;
        bottom:6px;
        height:2px;
        border-radius:2px;
        background:linear-gradient(90deg, rgba(139,212,255,.8), rgba(140,123,255,.5));
        opacity:.8;
    }

    .sg.block.green {
        background:linear-gradient(135deg, rgba(104,224,175,.11), rgba(139,212,255,.055));
        border-color:rgba(104,224,175,.15);
    }

    .sg.block.amber {
        background:linear-gradient(135deg, rgba(243,198,107,.10), rgba(139,212,255,.04));
        border-color:rgba(243,198,107,.15);
    }

    .signal-line {
        position:absolute;
        width:108px;
        height:1px;
        right:175px;
        top:180px;
        background:linear-gradient(90deg, transparent, rgba(139,212,255,.42), transparent);
        transform:rotate(-24deg);
        transform-origin:right center;
        opacity:.42;
    }

    /* ---------------- Metrics ---------------- */

    div[data-testid="stMetric"] {
        position:relative;
        min-height:110px;
        padding:16px 17px;
        border-radius:15px;
        border:1px solid rgba(136,161,190,.12);
        background:
            linear-gradient(180deg, rgba(15,22,37,.82), rgba(9,14,24,.78));
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,.022),
            0 14px 35px rgba(0,0,0,.12);
        overflow:hidden;
        transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }

    div[data-testid="stMetric"]::before {
        content:"";
        position:absolute;
        left:0;
        top:0;
        bottom:0;
        width:2px;
        background:linear-gradient(180deg, rgba(139,212,255,.65), rgba(140,123,255,.18));
    }

    div[data-testid="stMetric"]:hover {
        transform:translateY(-3px);
        border-color:rgba(139,212,255,.20);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,.025),
            0 18px 40px rgba(0,0,0,.19);
    }

    div[data-testid="stMetricLabel"] {
        color:#6f8094 !important;
        font-size:10px !important;
        font-weight:750;
        letter-spacing:.4px;
    }

    div[data-testid="stMetricValue"] {
        color:#eef6fc !important;
        font-size:31px !important;
        font-weight:900;
        letter-spacing:-1px;
    }

    /* ---------------- Generator ---------------- */

    .generator-card {
        position:relative;
        overflow:hidden;
        display:flex;
        align-items:center;
        gap:16px;
        margin-top:20px;
        padding:20px 21px;
        border-radius:16px;
        border:1px solid rgba(139,159,186,.14);
        background:
            radial-gradient(220px 100px at 4% 50%, rgba(139,212,255,.055), transparent 75%),
            linear-gradient(180deg, rgba(14,21,36,.86), rgba(9,14,24,.84));
        box-shadow:0 15px 40px rgba(0,0,0,.12);
    }

    .generator-card::after {
        content:"";
        position:absolute;
        left:-18%;
        top:0;
        width:32%;
        height:100%;
        background:linear-gradient(90deg, transparent, rgba(139,212,255,.05), transparent);
        transform:skewX(-18deg);
        animation:generatorSweep 6.5s ease-in-out infinite;
        pointer-events:none;
    }

    @keyframes generatorSweep {
        0%, 58% { transform:translateX(0) skewX(-18deg); opacity:0; }
        72% { opacity:1; }
        100% { transform:translateX(410%) skewX(-18deg); opacity:0; }
    }

    .generator-mark {
        width:44px;
        height:44px;
        flex:0 0 44px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:12px;
        border:1px solid rgba(139,212,255,.17);
        background:linear-gradient(145deg, rgba(26,43,67,.72), rgba(11,18,31,.84));
        color:#c9efff;
        font-size:10px;
        font-weight:900;
        letter-spacing:.8px;
        box-shadow:0 0 22px rgba(139,212,255,.06);
    }

    .generator-copy { min-width:0; }

    .generator-title {
        color:#eef5fa;
        font-size:17px;
        font-weight:850;
        letter-spacing:-.25px;
    }

    .generator-text {
        color:#7e8ea0;
        font-size:11px;
        line-height:1.6;
        margin-top:4px;
        max-width:920px;
    }

    /* ---------------- Buttons ---------------- */

    .stButton > button, .stDownloadButton > button {
        min-height:42px;
        border-radius:11px;
        font-weight:800;
        letter-spacing:.05px;
        border:1px solid rgba(139,159,186,.15);
        background:linear-gradient(180deg, rgba(27,36,52,.9), rgba(15,22,34,.9));
        color:#e6eef5;
        box-shadow:0 8px 22px rgba(0,0,0,.11);
        transition:
            transform .18s ease,
            border-color .18s ease,
            background .18s ease,
            box-shadow .18s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform:translateY(-2px);
        border-color:rgba(139,212,255,.25);
        background:linear-gradient(180deg, rgba(32,44,64,.95), rgba(17,25,39,.95));
        box-shadow:0 12px 28px rgba(0,0,0,.18);
    }

    .stButton > button[kind="primary"] {
        position:relative;
        overflow:hidden;
        color:#06101a;
        border-color:#a9e4ff;
        background:linear-gradient(135deg, #dff7ff 0%, #9edfff 52%, #a8a0ff 100%);
        box-shadow:
            0 10px 28px rgba(74,184,235,.15),
            inset 0 1px 0 rgba(255,255,255,.55);
    }

    .stButton > button[kind="primary"]::after {
        content:"";
        position:absolute;
        top:-20%;
        bottom:-20%;
        width:20%;
        left:-30%;
        transform:skewX(-18deg);
        background:rgba(255,255,255,.32);
        animation:buttonShine 4.8s ease-in-out infinite;
    }

    @keyframes buttonShine {
        0%, 70% { left:-30%; opacity:0; }
        76% { opacity:1; }
        88%,100% { left:130%; opacity:0; }
    }

    .stButton > button p, .stDownloadButton > button p { font-size:12px; }

    /* ---------------- Result command center ---------------- */

    .result-wrap {
        position:relative;
        overflow:hidden;
        margin-top:19px;
        padding:19px 21px;
        border-radius:16px;
        border:1px solid rgba(139,212,255,.15);
        background:
            radial-gradient(300px 120px at 10% 0%, rgba(139,212,255,.055), transparent 75%),
            radial-gradient(280px 140px at 92% 100%, rgba(140,123,255,.065), transparent 75%),
            linear-gradient(180deg, rgba(13,20,34,.92), rgba(8,13,23,.90));
        box-shadow:0 15px 42px rgba(0,0,0,.14);
    }

    .result-top {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:16px;
    }

    .score-kicker { color:#63768b; font-size:9px; font-weight:900; letter-spacing:1.7px; text-transform:uppercase; }
    .score-main { display:flex; align-items:flex-end; gap:10px; margin-top:5px; }
    .score-big { color:#f4f9fd; font-size:38px; font-weight:900; line-height:1; letter-spacing:-1.8px; }
    .score-caption { color:#718196; font-size:10px; margin-bottom:4px; }

    .result-status {
        display:inline-flex;
        align-items:center;
        gap:7px;
        padding:7px 10px;
        border-radius:999px;
        color:#9ce7c7;
        background:rgba(104,224,175,.055);
        border:1px solid rgba(104,224,175,.15);
        font-size:9px;
        font-weight:900;
        letter-spacing:.9px;
    }

    .result-status-dot {
        width:5px;
        height:5px;
        border-radius:50%;
        background:var(--green);
        box-shadow:0 0 10px rgba(104,224,175,.8);
    }

    .result-meta {
        color:#74849a;
        font-size:10px;
        margin-top:8px;
    }

    /* ---------------- Optimization Intelligence ---------------- */

    .health-panel {
        position:relative;
        overflow:hidden;
        margin-top:14px;
        padding:18px;
        border-radius:15px;
        border:1px solid rgba(137,161,188,.13);
        background:
            radial-gradient(260px 130px at 3% 0%, rgba(104,224,175,.05), transparent 72%),
            radial-gradient(330px 150px at 96% 100%, rgba(139,212,255,.045), transparent 72%),
            linear-gradient(180deg, rgba(12,20,33,.88), rgba(8,13,23,.94));
        box-shadow:0 14px 36px rgba(0,0,0,.12);
    }

    .health-top {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:18px;
        margin-bottom:16px;
    }

    .health-title {
        color:#edf4f8;
        font-size:16px;
        font-weight:850;
        letter-spacing:-.25px;
    }

    .health-subtitle {
        color:#6f8093;
        font-size:10px;
        line-height:1.55;
        margin-top:3px;
    }

    .health-score {
        flex:0 0 auto;
        display:flex;
        align-items:center;
        gap:9px;
    }

    .health-score-ring {
        width:47px;
        height:47px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        border:1px solid rgba(104,224,175,.17);
        background:
            radial-gradient(circle, rgba(104,224,175,.07) 0 52%, transparent 53%),
            conic-gradient(#68e0af 0 100%, rgba(255,255,255,.04) 100%);
        box-shadow:0 0 26px rgba(104,224,175,.06);
    }

    .health-score-value {
        color:#eafff6;
        font-size:12px;
        font-weight:900;
    }

    .health-score-caption {
        color:#728296;
        font-size:8px;
        font-weight:900;
        letter-spacing:.85px;
        text-transform:uppercase;
    }

    .health-grid {
        display:grid;
        grid-template-columns:repeat(5, minmax(0,1fr));
        gap:7px;
    }

    .health-item {
        min-height:78px;
        padding:11px;
        border-radius:10px;
        border:1px solid rgba(136,161,190,.09);
        background:rgba(255,255,255,.012);
    }

    .health-item-label {
        color:#607187;
        font-size:8px;
        line-height:1.3;
        font-weight:850;
        letter-spacing:.35px;
        text-transform:uppercase;
    }

    .health-item-value {
        color:#eaf3f8;
        font-size:20px;
        font-weight:900;
        letter-spacing:-.5px;
        margin-top:8px;
    }

    .health-item-ok .health-item-value { color:#9fe2c4; }
    .health-item-warn .health-item-value { color:#f1ca75; }

    .health-item-note {
        color:#56687c;
        font-size:7px;
        margin-top:3px;
    }

    .health-bars {
        margin-top:12px;
        padding-top:13px;
        border-top:1px solid rgba(136,161,190,.075);
    }

    .health-bar-row {
        display:grid;
        grid-template-columns:145px 1fr 44px;
        align-items:center;
        gap:9px;
        margin-top:7px;
    }

    .health-bar-row:first-child { margin-top:0; }

    .health-bar-label {
        color:#718297;
        font-size:8px;
        font-weight:800;
    }

    .health-bar {
        height:5px;
        border-radius:999px;
        background:rgba(255,255,255,.045);
        overflow:hidden;
    }

    .health-bar-fill {
        height:100%;
        border-radius:999px;
        background:linear-gradient(90deg, #73d8b4, #8bd4ff);
        box-shadow:0 0 11px rgba(115,216,180,.12);
    }

    .health-bar-value {
        color:#93a4b6;
        text-align:right;
        font-size:8px;
        font-weight:850;
        font-variant-numeric:tabular-nums;
    }

    .health-foot {
        margin-top:11px;
        color:#55677b;
        font-size:8px;
        line-height:1.45;
    }

    @media (max-width: 900px) {
        .health-grid { grid-template-columns:repeat(2, minmax(0,1fr)); }
        .health-score { display:none; }
        .health-bar-row { grid-template-columns:115px 1fr 38px; }
    }

    @media (max-width: 560px) {
        .health-grid { grid-template-columns:1fr; }
    }

    /* ---------------- Timetable ---------------- */

    .table-toolbar {
        display:flex;
        align-items:flex-end;
        justify-content:space-between;
        gap:18px;
        margin-top:27px;
        margin-bottom:11px;
    }

    .table-kicker { color:#617287; font-size:9px; font-weight:900; letter-spacing:1.7px; text-transform:uppercase; }
    .table-title { color:#eef5fa; font-size:23px; font-weight:850; letter-spacing:-.65px; margin-top:3px; }
    .table-desc { color:#7b8a9d; font-size:11px; margin-top:4px; }
    .table-badge {
        padding:7px 9px;
        border-radius:8px;
        background:rgba(139,212,255,.05);
        border:1px solid rgba(139,212,255,.11);
        color:#9eb7c9;
        font-size:9px;
        font-weight:900;
        letter-spacing:.8px;
        white-space:nowrap;
    }

    div[data-testid="stDataFrame"] {
        border:1px solid rgba(137,161,188,.13);
        border-radius:15px;
        overflow:hidden;
        background:rgba(10,15,25,.85);
        box-shadow:0 16px 40px rgba(0,0,0,.14);
    }

    .export-note {
        color:#64758a;
        font-size:10px;
        margin:9px 0 9px;
    }

    /* ---------------- Visual Timetable ---------------- */

    .visual-timetable {
        position:relative;
        width:100%;
        max-width:100%;
        margin:0 0 16px;
        padding:17px;
        border:1px solid rgba(137,161,188,.13);
        border-radius:15px;
        background:
            radial-gradient(300px 130px at 10% 0%, rgba(139,212,255,.045), transparent 75%),
            linear-gradient(180deg, rgba(10,16,28,.84), rgba(7,11,19,.92));
        box-shadow:0 15px 38px rgba(0,0,0,.12);
        overflow:hidden;
    }

    .visual-timetable-head {
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        gap:16px;
        margin-bottom:13px;
    }

    .grid-title-block { min-width:0; }
    .grid-title {
        color:#eef5fa;
        font-size:18px;
        font-weight:850;
        letter-spacing:-.35px;
        margin-top:3px;
    }

    .grid-subtitle {
        color:#718297;
        font-size:10px;
        margin-top:3px;
    }

    .grid-count {
        flex:0 0 auto;
        padding:6px 9px;
        border-radius:8px;
        border:1px solid rgba(104,224,175,.13);
        background:rgba(104,224,175,.04);
        color:#91d8ba;
        font-size:9px;
        font-weight:900;
        letter-spacing:.75px;
    }

    .timetable-scroll {
        width:100%;
        max-width:100%;
        overflow-x:auto;
        padding-bottom:3px;
    }

    .tt-grid {
        display:grid;
        gap:5px;
        min-width:760px;
    }

    .tt-cell {
        min-height:76px;
        border-radius:8px;
        border:1px solid rgba(136,161,190,.075);
        background:rgba(255,255,255,.012);
        overflow:hidden;
    }

    .tt-corner {
        min-height:48px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#536479;
        font-size:8px;
        font-weight:900;
        letter-spacing:1.2px;
        background:rgba(255,255,255,.008);
    }

    .tt-day {
        min-height:48px;
        padding:8px 9px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        color:#d9e6ef;
        background:linear-gradient(180deg, rgba(25,37,54,.75), rgba(13,20,32,.72));
        border-color:rgba(139,212,255,.105);
    }

    .tt-day span {
        font-size:10px;
        font-weight:900;
        letter-spacing:1px;
    }

    .tt-day small {
        color:#66788d;
        font-size:8px;
        margin-top:3px;
    }

    .tt-time {
        min-height:76px;
        display:flex;
        align-items:center;
        justify-content:center;
        padding:7px;
        color:#6f8195;
        font-size:9px;
        font-weight:850;
        font-variant-numeric:tabular-nums;
        background:rgba(13,19,30,.58);
    }

    .tt-empty {
        min-height:76px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#354254;
        background:rgba(255,255,255,.007);
    }

    .tt-empty span {
        opacity:.55;
        font-size:10px;
    }

    .tt-filled {
        min-height:76px;
        padding:5px;
        display:flex;
        flex-direction:column;
        gap:5px;
        background:rgba(12,19,30,.62);
    }

    .tt-card {
        position:relative;
        padding:8px 8px 7px;
        min-width:0;
        border:1px solid rgba(139,212,255,.12);
        border-left:2px solid rgba(139,212,255,.55);
        border-radius:7px;
        background:
            linear-gradient(135deg, rgba(29,46,68,.72), rgba(13,21,34,.86));
        box-shadow:inset 0 1px 0 rgba(255,255,255,.018);
        transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }

    .tt-card:hover {
        transform:translateY(-2px);
        border-color:rgba(139,212,255,.28);
        box-shadow:
            0 8px 18px rgba(0,0,0,.19),
            inset 0 1px 0 rgba(255,255,255,.025);
    }

    .tt-course {
        color:#e8f4fa;
        font-size:10px;
        line-height:1.25;
        font-weight:850;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }

    .tt-code {
        color:#6fbedf;
        font-size:8px;
        margin-top:3px;
        font-weight:850;
        letter-spacing:.45px;
    }

    .tt-meta {
        display:flex;
        gap:8px;
        margin-top:7px;
        color:#8394a8;
        font-size:8px;
        font-weight:750;
    }

    .tt-section {
        color:#586a7e;
        font-size:7px;
        margin-top:4px;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }

    .visual-empty {
        padding:20px;
        border:1px solid rgba(137,161,188,.10);
        border-radius:12px;
        color:#6e7d90;
        background:rgba(10,15,25,.56);
        text-align:center;
        font-size:11px;
    }

    /* ---------------- Analytics ---------------- */

    .analytics-card {
        padding:14px 15px;
        border-radius:14px;
        border:1px solid rgba(137,161,188,.10);
        background:rgba(9,14,24,.55);
    }

    div[data-testid="stPlotlyChart"] {
        border:1px solid rgba(137,161,188,.10);
        border-radius:15px;
        background:rgba(10,15,25,.76);
        padding:3px;
        overflow:hidden;
        box-shadow:0 12px 32px rgba(0,0,0,.09);
    }

    /* ---------------- AI Assistant ---------------- */

    .ai-panel {
        position:relative;
        overflow:hidden;
        width:100%;
        max-width:100%;
        margin-top:29px;
        padding:22px 22px 20px;
        border-radius:18px;
        border:1px solid rgba(139,212,255,.16);
        background:
            radial-gradient(330px 150px at 7% 15%, rgba(139,212,255,.065), transparent 72%),
            radial-gradient(320px 150px at 93% 90%, rgba(140,123,255,.065), transparent 72%),
            linear-gradient(180deg, rgba(13,21,36,.94), rgba(8,13,23,.92));
        box-shadow:0 20px 52px rgba(0,0,0,.18);
        box-sizing:border-box;
    }

    .ai-panel::before {
        content:"";
        position:absolute;
        left:-40px;
        top:-40px;
        width:130px;
        height:130px;
        border-radius:50%;
        border:1px solid rgba(139,212,255,.08);
        box-shadow:0 0 0 18px rgba(139,212,255,.015), 0 0 0 36px rgba(139,212,255,.010);
        pointer-events:none;
    }

    .ai-top {
        display:flex;
        align-items:center;
        gap:12px;
        min-width:0;
    }

    .ai-icon {
        width:42px;
        height:42px;
        flex:0 0 42px;
        border-radius:12px;
        display:flex;
        align-items:center;
        justify-content:center;
        border:1px solid rgba(139,212,255,.18);
        background:linear-gradient(145deg, rgba(29,48,74,.86), rgba(12,19,32,.9));
        color:#dff8ff;
        font-size:12px;
        font-weight:900;
        box-shadow:0 0 24px rgba(139,212,255,.06);
    }

    .ai-title {
        color:#eef5fa;
        font-size:20px;
        font-weight:850;
        letter-spacing:-.4px;
    }

    .ai-desc {
        color:#78899d;
        font-size:11px;
        margin-top:3px;
    }

    .ai-status {
        margin-left:auto;
        flex:0 0 auto;
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding:6px 9px;
        border-radius:999px;
        color:#97dfbf;
        background:rgba(104,224,175,.045);
        border:1px solid rgba(104,224,175,.13);
        font-size:9px;
        font-weight:900;
        letter-spacing:.7px;
    }

    .ai-status-dot {
        width:5px;
        height:5px;
        border-radius:50%;
        background:var(--green);
        box-shadow:0 0 10px rgba(104,224,175,.85);
    }

    .ai-quick-label {
        color:#66788d;
        font-size:9px;
        font-weight:900;
        letter-spacing:1.65px;
        text-transform:uppercase;
        margin-top:17px;
        margin-bottom:6px;
    }

    .ai-hint {
        color:#6f8092;
        font-size:10px;
        margin-top:7px;
    }

    [data-testid="stForm"] {
        width:100%;
        max-width:100%;
        box-sizing:border-box;
        overflow:hidden;
    }

    .ai-response {
        width:100%;
        max-width:100%;
        box-sizing:border-box;
        margin-top:15px;
        padding:18px;
        border-radius:14px;
        border:1px solid rgba(139,212,255,.12);
        background:
            linear-gradient(135deg, rgba(18,29,46,.75), rgba(10,16,28,.88));
        overflow-wrap:anywhere;
        box-shadow:0 12px 30px rgba(0,0,0,.12);
    }

    .ai-response-title {
        color:#a8dff7;
        font-size:9px;
        font-weight:900;
        letter-spacing:1.35px;
        margin-bottom:8px;
    }

    .ai-response-text {
        color:#d7e1ea;
        font-size:13px;
        line-height:1.7;
        overflow-wrap:anywhere;
    }

    /* ---------------- Data management ---------------- */

    .data-note {
        margin:4px 0 16px;
        padding:10px 12px;
        border-radius:10px;
        color:#8191a4;
        background:rgba(12,18,30,.67);
        border:1px solid rgba(132,157,188,.10);
        font-size:10px;
    }

    .data-section-copy {
        color:#6f8092;
        font-size:10px;
        margin:4px 0 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap:5px;
        border-bottom:1px solid rgba(136,161,190,.10);
    }

    .stTabs [data-baseweb="tab"] {
        color:#718196;
        background:transparent;
        border:none;
        padding:9px 12px;
        font-size:10px;
        font-weight:800;
    }

    .stTabs [aria-selected="true"] {
        color:#eaf3f9 !important;
        border-bottom:2px solid var(--accent);
    }

    /* ---------------- Inputs / misc ---------------- */

    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background:rgba(8,13,23,.9) !important;
        color:#edf4f9 !important;
        border:1px solid rgba(136,161,190,.15) !important;
        border-radius:11px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color:rgba(139,212,255,.38) !important;
        box-shadow:0 0 0 1px rgba(139,212,255,.24), 0 0 0 5px rgba(139,212,255,.035) !important;
    }

    .stAlert {
        background:rgba(13,20,32,.88) !important;
        border:1px solid rgba(139,159,186,.12) !important;
        color:#dfe8ef !important;
    }

    details[data-testid="stExpander"] {
        background:rgba(10,15,26,.75);
        border:1px solid rgba(136,161,190,.10);
        border-radius:12px;
        margin-bottom:10px;
    }

    .streamlit-expanderHeader {
        color:#edf4f8 !important;
        font-weight:750;
    }

    hr { border-color:rgba(136,161,190,.09); }

    .footer {
        text-align:center;
        color:#4f6074;
        font-size:9px;
        padding:28px 0 5px;
        letter-spacing:.25px;
    }

    @media (max-width: 1100px) {
        .schedule-art { opacity:.42; right:-70px; }
        .hero-content { max-width:760px; }
    }

    @media (max-width: 900px) {
        .main .block-container { padding-left:1rem; padding-right:1rem; }
        .hero { padding:28px 24px; min-height:280px; }
        .hero-title { font-size:40px; letter-spacing:-2px; }
        .hero-description { font-size:13px; }
        .schedule-art { display:none; }
        .table-toolbar { align-items:flex-start; flex-direction:column; }
        .ai-top { align-items:flex-start; }
        .ai-status { margin-left:auto; }
    }

    @media (max-width: 650px) {
        .hero-meta-item { width:100%; }
        .hero-title { font-size:34px; }
        .ai-status { font-size:8px; padding:5px 7px; }
    }
    /* =====================================================
       COMPLETE PRODUCT POLISH
       ===================================================== */

    .analytics-stat {
        padding:14px;
        border:1px solid rgba(137,161,188,.10);
        border-radius:13px;
        background:rgba(10,15,25,.62);
        min-height:82px;
    }
    .analytics-stat-label {
        color:#65768a;
        font-size:8px;
        font-weight:900;
        letter-spacing:1.1px;
        text-transform:uppercase;
    }
    .analytics-stat-value {
        color:#edf5f9;
        font-size:18px;
        font-weight:900;
        margin-top:7px;
        letter-spacing:-.3px;
    }
    .analytics-stat-note { color:#56687c; font-size:8px; margin-top:3px; }

    .insight-panel {
        margin:13px 0 5px;
        padding:15px 16px;
        border:1px solid rgba(139,212,255,.10);
        border-radius:13px;
        background:linear-gradient(135deg,rgba(16,27,43,.72),rgba(9,14,24,.82));
    }
    .insight-title {
        color:#bfe8fa;
        font-size:9px;
        font-weight:900;
        letter-spacing:1.2px;
        text-transform:uppercase;
    }
    .insight-line { color:#899aab; font-size:10px; line-height:1.6; margin-top:6px; }
    .insight-line b { color:#d7e4ec; }

    .export-panel {
        padding:15px;
        margin-top:14px;
        margin-bottom:8px;
        border:1px solid rgba(137,161,188,.10);
        border-radius:14px;
        background:rgba(9,14,24,.54);
    }
    .export-title { color:#e8f1f6; font-size:13px; font-weight:850; }
    .export-desc { color:#66788c; font-size:9px; margin-top:3px; line-height:1.55; }

    .validation-panel {
        padding:14px 15px;
        margin:10px 0 15px;
        border-radius:13px;
        border:1px solid rgba(139,159,186,.10);
        background:rgba(10,15,25,.52);
    }
    .validation-title {
        color:#dce7ef;
        font-size:10px;
        font-weight:900;
        letter-spacing:1.1px;
        text-transform:uppercase;
    }
    .validation-item { color:#728398; font-size:9px; line-height:1.55; margin-top:6px; }
    .validation-item b { color:#c8d6e3; }

    .ai-history { margin-top:12px; display:flex; flex-direction:column; gap:8px; }
    .ai-msg-user {
        padding:10px 12px;
        border-radius:10px;
        border:1px solid rgba(139,212,255,.08);
        background:rgba(139,212,255,.035);
    }
    .ai-msg-ai {
        padding:11px 12px;
        border-radius:10px;
        border:1px solid rgba(139,212,255,.10);
        background:rgba(11,18,29,.74);
    }
    .ai-msg-label {
        color:#64768a;
        font-size:7px;
        font-weight:900;
        letter-spacing:1px;
        text-transform:uppercase;
    }
    .ai-msg-text { color:#d7e2ea; font-size:10px; line-height:1.55; margin-top:4px; }

    .project-feature {
        min-height:122px;
        padding:16px;
        border-radius:14px;
        border:1px solid rgba(137,161,188,.11);
        background:rgba(10,16,27,.67);
    }
    .project-feature-title { color:#e9f1f6; font-size:12px; font-weight:850; }
    .project-feature-copy { color:#6f8194; font-size:9px; line-height:1.55; margin-top:6px; }
    .stack-chip {
        display:inline-flex;
        padding:7px 9px;
        margin:4px 4px 0 0;
        border-radius:8px;
        color:#9cafc0;
        background:rgba(139,212,255,.035);
        border:1px solid rgba(139,212,255,.085);
        font-size:8px;
        font-weight:850;
    }
    .empty-state {
        padding:25px;
        border:1px dashed rgba(137,161,188,.15);
        border-radius:14px;
        text-align:center;
        background:rgba(9,14,24,.52);
    }
    .empty-state-title { color:#e7eff5; font-size:14px; font-weight:850; }
    .empty-state-copy { color:#6e8094; font-size:10px; line-height:1.6; margin-top:5px; }

    @media (max-width:900px) {
        .analytics-stat { min-height:75px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# AMBIENT BACKDROP DETAILS
# =========================================================

render_html(
    """
    <div style="
        position:fixed; inset:0; pointer-events:none; z-index:0;
        overflow:hidden;
    ">
        <div style="
            position:absolute; width:230px; height:230px; border-radius:50%;
            left:-110px; top:31vh;
            background:rgba(74,201,228,.055);
            filter:blur(40px);
            animation:ambientOrbOne 12s ease-in-out infinite alternate;
        "></div>
        <div style="
            position:absolute; width:300px; height:300px; border-radius:50%;
            right:-145px; top:7vh;
            background:rgba(124,101,255,.065);
            filter:blur(50px);
            animation:ambientOrbTwo 14s ease-in-out infinite alternate;
        "></div>
        <style>
            @keyframes ambientOrbOne {
                from { transform:translate3d(0,0,0); }
                to { transform:translate3d(55px,-25px,0); }
            }
            @keyframes ambientOrbTwo {
                from { transform:translate3d(0,0,0); }
                to { transform:translate3d(-45px,32px,0); }
            }
        </style>
    </div>
    """
)


# =========================================================
# LOAD DATA
# =========================================================

try:
    data = load_data()
except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    render_html(
        """
        <div class="brand-row">
            <div class="brand-mark">SS</div>
            <div>
                <div class="brand-name">SmartSched AI</div>
                <div class="brand-sub">Academic scheduling workspace</div>
            </div>
        </div>
        """
    )

    render_html('<div class="side-heading">Workspace</div>')

    page = st.radio(
        "Navigation",
        ["Dashboard", "Data Management", "Project"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    render_html('<div class="side-heading">Dataset</div>')

    render_html(
        f"""
        <div class="side-data-card">
            <div class="side-data-row"><span>Courses</span><span>{len(data["courses"])}</span></div>
            <div class="side-data-row"><span>Faculty</span><span>{len(data["faculty"])}</span></div>
            <div class="side-data-row"><span>Classrooms</span><span>{len(data["rooms"])}</span></div>
            <div class="side-data-row"><span>Sections</span><span>{len(data["sections"])}</span></div>
            <div class="side-data-row"><span>Time slots</span><span>{len(data["timeslots"])}</span></div>
        </div>
        """
    )

    render_html(
        """
        <div class="side-status">
            <span class="status-dot"></span>
            Engine ready · Local AI connected
        </div>
        """
    )


# =========================================================

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    render_html(
        """
        <div class="hero">
            <div class="hero-grid"></div>
            <div class="hero-scan"></div>
            <div class="hero-ring"></div>

            <div class="hero-content">
                <div class="hero-badge">
                    <span class="dot"></span>
                    SMART SCHEDULING OPERATING SPACE
                </div>

                <div class="hero-title">
                    Build the <span>perfect timetable.</span>
                </div>

                <div class="hero-description">
                    SmartSched AI combines constraint-aware scheduling, optimization,
                    timetable intelligence and a local AI assistant into one focused academic workspace.
                </div>

                <div class="hero-meta">
                    <div class="hero-meta-item">Engine <b>Constraint-aware</b></div>
                    <div class="hero-meta-item">Optimization <b>Automatic</b></div>
                    <div class="hero-meta-item">Assistant <b>Local AI</b></div>
                    <div class="hero-meta-item">Export <b>CSV · Excel · PDF</b></div>
                </div>
            </div>

            <div class="schedule-art">
                <div class="schedule-shell"></div>
                <div class="schedule-head">
                    <span>LIVE SCHEDULE MAP</span>
                    <span class="schedule-live">● READY</span>
                </div>

                <div class="schedule-grid">
                    <div class="sg head"></div>
                    <div class="sg head">09:00</div>
                    <div class="sg head">10:00</div>
                    <div class="sg head">11:00</div>
                    <div class="sg head">12:00</div>
                    <div class="sg time">MON</div>
                    <div class="sg block"></div>
                    <div class="sg"></div>
                    <div class="sg block green"></div>
                    <div class="sg"></div>
                    <div class="sg time">TUE</div>
                    <div class="sg"></div>
                    <div class="sg block amber"></div>
                    <div class="sg"></div>
                    <div class="sg block"></div>
                    <div class="sg time">WED</div>
                    <div class="sg block green"></div>
                    <div class="sg"></div>
                    <div class="sg block"></div>
                    <div class="sg"></div>
                    <div class="sg time">THU</div>
                    <div class="sg"></div>
                    <div class="sg block"></div>
                    <div class="sg"></div>
                    <div class="sg block green"></div>
                    <div class="sg time">FRI</div>
                    <div class="sg block"></div>
                    <div class="sg"></div>
                    <div class="sg block amber"></div>
                    <div class="sg"></div>
                </div>

                <div class="signal-line"></div>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="eyebrow">WORKSPACE SNAPSHOT</div>
        <div class="section-title">Scheduling environment</div>
        <div class="section-desc">
            Live counts from the datasets currently available to the scheduling engine.
        </div>
        <div style="height:13px"></div>
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Courses", len(data["courses"]))
    with col2:
        st.metric("Faculty", len(data["faculty"]))
    with col3:
        st.metric("Classrooms", len(data["rooms"]))
    with col4:
        st.metric("Sections", len(data["sections"]))

    render_html(
        """
        <div class="generator-card">
            <div class="generator-mark">RUN</div>
            <div class="generator-copy">
                <div class="generator-title">Timetable Generation Engine</div>
                <div class="generator-text">
                    Validate source data, build a conflict-aware schedule and optimize the result.
                </div>
            </div>
        </div>
        <div style="height:10px"></div>
        """
    )

    if st.button(
        "Generate Optimized Timetable",
        type="primary",
        use_container_width=True,
    ):
        validation = validate_dataset(data)

        # Harmless diagnostics are shown without blocking the existing
        # scheduler. The scheduler remains the final authority on whether
        # an assignment can actually be generated.
        if validation["warnings"]:
            with st.expander("Data diagnostics", expanded=False):
                for item in validation["warnings"][:8]:
                    st.info(item)

        if validation["issues"]:
            with st.expander("Data issues detected", expanded=False):
                for item in validation["issues"][:8]:
                    st.warning(item)

        try:
            with st.spinner("Analyzing constraints and optimizing your timetable..."):
                generate_current_timetable(data)
            st.toast("Optimized timetable generated.")
        except Exception as error:
            st.session_state.pop("timetable", None)
            st.session_state.pop("score", None)
            st.error(
                "Timetable generation could not complete with the current data. "
                f"Details: {error}"
            )

    if "timetable" in st.session_state:
        timetable = st.session_state["timetable"]
        score = int(st.session_state.get("score", 0))
        health = analyze_schedule_health(timetable, data)

        status_clear = (
            health["unscheduled"] == 0
            and health["faculty_conflicts"] == 0
            and health["room_conflicts"] == 0
            and health["section_conflicts"] == 0
            and health["capacity_violations"] == 0
        )
        status_text = "ALL CHECKS CLEAR" if status_clear else "REVIEW REQUIRED"

        render_html(
            f"""
            <div class="result-wrap">
                <div class="result-top">
                    <div>
                        <div class="score-kicker">OPTIMIZATION RESULT</div>
                        <div class="score-main">
                            <div class="score-big">{score}/100</div>
                            <div class="score-caption">timetable quality</div>
                        </div>
                        <div class="result-meta">
                            Generated {html_lib.escape(st.session_state.get("generated_at", "—"))}
                            · {health["scheduled"]} scheduled · {health["unscheduled"]} unscheduled
                        </div>
                    </div>
                    <div class="result-status">
                        <span class="result-status-dot"></span>
                        {status_text}
                    </div>
                </div>
            </div>
            """
        )

        if health["unscheduled"]:
            st.warning(
                f'{health["unscheduled"]} class(es) could not be placed. '
                "Review rooms, faculty availability, capacity and available time slots."
            )

        render_html(
            f"""
            <div class="health-panel">
                <div class="health-top">
                    <div>
                        <div class="score-kicker">SCHEDULE HEALTH</div>
                        <div class="health-title">Optimization Intelligence</div>
                        <div class="health-subtitle">
                            Transparent signals from the generated schedule.
                        </div>
                    </div>
                    <div class="health-score">
                        <div class="health-score-ring">
                            <div class="health-score-value">{max(0, min(100, score))}</div>
                        </div>
                        <div class="health-score-caption">Quality<br>score</div>
                    </div>
                </div>

                <div class="health-grid">
                    <div class="health-item health-item-ok">
                        <div class="health-item-label">Scheduled</div>
                        <div class="health-item-value">{health["scheduled"]}</div>
                        <div class="health-item-note">of {len(timetable)} classes</div>
                    </div>
                    <div class="health-item {'health-item-ok' if health["unscheduled"] == 0 else 'health-item-warn'}">
                        <div class="health-item-label">Unscheduled</div>
                        <div class="health-item-value">{health["unscheduled"]}</div>
                        <div class="health-item-note">remaining classes</div>
                    </div>
                    <div class="health-item {'health-item-ok' if health["faculty_conflicts"] == 0 else 'health-item-warn'}">
                        <div class="health-item-label">Faculty overlap</div>
                        <div class="health-item-value">{health["faculty_conflicts"]}</div>
                        <div class="health-item-note">interval clashes</div>
                    </div>
                    <div class="health-item {'health-item-ok' if health["room_conflicts"] == 0 else 'health-item-warn'}">
                        <div class="health-item-label">Room overlap</div>
                        <div class="health-item-value">{health["room_conflicts"]}</div>
                        <div class="health-item-note">interval clashes</div>
                    </div>
                    <div class="health-item {'health-item-ok' if health["capacity_violations"] == 0 else 'health-item-warn'}">
                        <div class="health-item-label">Capacity issues</div>
                        <div class="health-item-value">{health["capacity_violations"]}</div>
                        <div class="health-item-note">best-effort check</div>
                    </div>
                </div>

                <div class="health-bars">
                    <div class="health-bar-row">
                        <div class="health-bar-label">Schedule coverage</div>
                        <div class="health-bar">
                            <div class="health-bar-fill" style="width:{health["coverage"]}%"></div>
                        </div>
                        <div class="health-bar-value">{health["coverage"]}%</div>
                    </div>
                    <div class="health-bar-row">
                        <div class="health-bar-label">Optimization quality</div>
                        <div class="health-bar">
                            <div class="health-bar-fill" style="width:{max(0, min(100, score))}%"></div>
                        </div>
                        <div class="health-bar-value">{score}%</div>
                    </div>
                </div>
            </div>
            """
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Quality", f"{score}/100")
        with c2:
            st.metric("Coverage", f'{health["coverage"]}%')
        with c3:
            st.metric("Unscheduled", health["unscheduled"])
        with c4:
            st.metric(
                "Resource clashes",
                health["faculty_conflicts"] + health["room_conflicts"] + health["section_conflicts"],
            )

        render_html(
            """
            <div class="table-toolbar">
                <div>
                    <div class="table-kicker">SCHEDULE EXPLORER</div>
                    <div class="table-title">Optimized Timetable</div>
                    <div class="table-desc">
                        Filter the generated schedule or switch between the weekly visual and detailed table.
                    </div>
                </div>
                <div class="table-badge">LATEST RUN</div>
            </div>
            """
        )

        filtered_timetable = apply_timetable_filters(timetable)

        view_mode = st.radio(
            "Timetable view",
            ["Both", "Weekly Grid", "Detailed Table"],
            horizontal=True,
            label_visibility="collapsed",
            key="timetable_view_mode",
        )

        if view_mode in ["Both", "Weekly Grid"]:
            render_timetable_grid(filtered_timetable)

        display_columns = [
            "course_id", "course_name", "faculty_id", "section_id",
            "room_id", "day", "start_time", "end_time", "class_number",
        ]
        available_columns = [c for c in display_columns if c in filtered_timetable.columns]
        display_timetable = filtered_timetable[available_columns]

        if view_mode in ["Both", "Detailed Table"]:
            st.dataframe(
                display_timetable,
                use_container_width=True,
                hide_index=True,
                height=390,
                column_config={
                    "course_id": "Course ID",
                    "course_name": "Course",
                    "faculty_id": "Faculty",
                    "section_id": "Section",
                    "room_id": "Room",
                    "day": "Day",
                    "start_time": "Start",
                    "end_time": "End",
                    "class_number": "Class #",
                },
            )

        render_html(
            """
            <div class="export-panel">
                <div class="export-title">Export & presentation</div>
                <div class="export-desc">
                    Download the current filtered view or a full multi-sheet report pack.
                </div>
            </div>
            """
        )

        csv_data = display_timetable.to_csv(index=False).encode("utf-8")
        ex1, ex2, ex3, ex4 = st.columns(4)

        with ex1:
            st.download_button(
                "CSV",
                data=csv_data,
                file_name="smart_scheduled_timetable.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with ex2:
            try:
                excel_data = make_excel_bytes(timetable)
                st.download_button(
                    "Excel Workbook",
                    data=excel_data,
                    file_name="smart_scheduled_timetable.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as error:
                st.button("Excel unavailable", disabled=True, use_container_width=True)
                st.caption(str(error))

        with ex3:
            try:
                pdf_data = make_pdf_bytes(display_timetable)
                st.download_button(
                    "Print PDF",
                    data=pdf_data,
                    file_name="smart_scheduled_timetable.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as error:
                st.button("PDF unavailable", disabled=True, use_container_width=True)
                st.caption(str(error))

        with ex4:
            st.download_button(
                "Print HTML",
                data=make_print_html(display_timetable),
                file_name="smart_scheduled_timetable_print.html",
                mime="text/html",
                use_container_width=True,
            )

        # -------------------------------------------------
        # ANALYTICS INTELLIGENCE
        # -------------------------------------------------

        render_html(
            """
            <div style="height:28px"></div>
            <div class="eyebrow">ANALYTICS</div>
            <div class="section-title">Timetable Intelligence</div>
            <div class="section-desc">
                Turn scheduling output into concise operational signals.
            </div>
            <div style="height:11px"></div>
            """
        )

        analytics = get_analytics_summary(timetable)
        stats = [
            ("Busiest day", analytics["busiest_day"], "highest class count"),
            ("Peak period", analytics["busiest_time"], "most common start"),
            ("Top faculty", analytics["top_faculty"], "most assigned"),
            ("Top room", analytics["top_room"], "most assigned"),
            ("Average / day", analytics["avg_per_day"], f'{analytics["days_active"]} active days'),
        ]

        metric_cols = st.columns(5)
        for col, (label, value, note) in zip(metric_cols, stats):
            with col:
                render_html(
                    f"""
                    <div class="analytics-stat">
                        <div class="analytics-stat-label">{html_lib.escape(str(label))}</div>
                        <div class="analytics-stat-value">{html_lib.escape(str(value))}</div>
                        <div class="analytics-stat-note">{html_lib.escape(str(note))}</div>
                    </div>
                    """
                )

        scheduled_frame = get_scheduled_frame(timetable)
        if not scheduled_frame.empty:
            day_counts = scheduled_frame["day"].astype(str).value_counts() if "day" in scheduled_frame.columns else pd.Series(dtype="int64")
            busiest_count = int(day_counts.iloc[0]) if len(day_counts) else 0
            faculty_count = int(scheduled_frame["faculty_id"].astype(str).nunique()) if "faculty_id" in scheduled_frame.columns else 0
            room_count = int(scheduled_frame["room_id"].astype(str).nunique()) if "room_id" in scheduled_frame.columns else 0

            render_html(
                f"""
                <div class="insight-panel">
                    <div class="insight-title">Key findings</div>
                    <div class="insight-line">
                        <b>{html_lib.escape(str(analytics["busiest_day"]))}</b>
                        is currently the busiest day with <b>{busiest_count}</b> class assignment(s).
                    </div>
                    <div class="insight-line">
                        The schedule uses <b>{faculty_count}</b> faculty member(s)
                        across <b>{room_count}</b> classroom(s).
                    </div>
                    <div class="insight-line">
                        The most common starting period is <b>{html_lib.escape(str(analytics["busiest_time"]))}</b>.
                    </div>
                </div>
                """
            )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            safe_chart("Faculty workload", faculty_workload_chart, timetable)
        with chart_col2:
            safe_chart("Room utilization", room_utilization_chart, timetable)
        safe_chart("Daily distribution", daily_class_chart, timetable)

        # -------------------------------------------------
        # AI ASSISTANT
        # -------------------------------------------------

        render_html(
            """
            <div class="ai-panel">
                <div class="ai-top">
                    <div class="ai-icon">AI</div>
                    <div style="min-width:0;">
                        <div class="ai-title">SmartSched AI Assistant</div>
                        <div class="ai-desc">Ask natural-language questions about your generated timetable.</div>
                    </div>
                    <div class="ai-status">
                        <span class="ai-status-dot"></span>
                        ONLINE
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="ai-quick-label">Quick questions</div>
            <div class="ai-hint">Choose a shortcut or type your own question. Press Enter to submit.</div>
            """
        )

        quick_questions = [
            ("Most-used room", "Which room is used the most?"),
            ("Monday classes", "How many classes are scheduled on Monday?"),
            ("F001 courses", "Which courses are handled by F001?"),
            ("Conflicts", "Are there any timetable conflicts?"),
        ]

        quick_question = None
        quick_cols = st.columns(4, gap="small")

        for index, (button_label, question_text) in enumerate(quick_questions):
            with quick_cols[index]:
                if st.button(button_label, key=f"quick_ai_{index}", use_container_width=True):
                    quick_question = question_text

        with st.form("ai_question_form", clear_on_submit=False):
            question = st.text_input(
                "Ask SmartSched AI",
                placeholder="Ask anything about your timetable...",
                label_visibility="collapsed",
                key="ai_question_input",
            )
            submitted = st.form_submit_button("Ask AI", use_container_width=True)

        if "ai_history" not in st.session_state:
            st.session_state["ai_history"] = []

        question_to_ask = quick_question if quick_question else question.strip() if submitted else ""

        if question_to_ask:
            try:
                with st.spinner("AI is analyzing your timetable..."):
                    answer = ask_timetable_assistant(question_to_ask, timetable)
                st.session_state["ai_answer"] = answer
                st.session_state["ai_question"] = question_to_ask
                st.session_state["ai_history"].append(
                    {"question": question_to_ask, "answer": str(answer)}
                )
                st.session_state["ai_history"] = st.session_state["ai_history"][-6:]
            except Exception as error:
                st.error(f"AI assistant failed to respond: {error}")

        elif submitted and not question.strip():
            st.warning("Please enter a question.")

        if "ai_answer" in st.session_state:
            safe_answer = html_lib.escape(str(st.session_state["ai_answer"])).replace("\n", "<br>")
            safe_question = html_lib.escape(str(st.session_state["ai_question"]))
            render_html(
                f"""
                <div class="ai-response">
                    <div class="ai-response-title">AI RESPONSE</div>
                    <div class="ai-response-text">{safe_answer}</div>
                </div>
                <div class="ai-hint">Asked: {safe_question}</div>
                """
            )

        history = st.session_state.get("ai_history", [])
        if history:
            clear_col, _ = st.columns([1, 4])
            with clear_col:
                if st.button("Clear AI history", key="clear_ai_history"):
                    st.session_state["ai_history"] = []
                    st.session_state.pop("ai_answer", None)
                    st.session_state.pop("ai_question", None)
                    st.rerun()

            markup = ['<div class="ai-history">']
            for item in reversed(history):
                safe_q = html_lib.escape(item["question"])
                safe_a = html_lib.escape(item["answer"]).replace("\n", "<br>")
                markup.append(
                    f"""
                    <div class="ai-msg-user">
                        <div class="ai-msg-label">YOU</div>
                        <div class="ai-msg-text">{safe_q}</div>
                    </div>
                    <div class="ai-msg-ai">
                        <div class="ai-msg-label">SMARTSCHED AI</div>
                        <div class="ai-msg-text">{safe_a}</div>
                    </div>
                    """
                )
            markup.append("</div>")
            render_html("\n".join(markup))

    else:
        render_html(
            """
            <div style="height:20px"></div>
            <div class="empty-state">
                <div class="empty-state-title">No timetable generated yet</div>
                <div class="empty-state-copy">
                    Run the generation engine above to unlock the visual schedule,
                    optimization intelligence, analytics, exports and AI assistant.
                </div>
            </div>
            """
        )



# =========================================================
# DATA MANAGEMENT
# =========================================================

elif page == "Data Management":

    render_html(
        """
        <div class="hero">
            <div class="hero-grid"></div>
            <div class="hero-scan"></div>
            <div class="hero-ring"></div>

            <div class="hero-content">
                <div class="hero-badge">
                    <span class="dot"></span>
                    WORKSPACE DATA CONTROL
                </div>

                <div class="hero-title">
                    Shape your <span>source data.</span>
                </div>

                <div class="hero-description">
                    Search, inspect, validate and edit the records used by the scheduling engine.
                    Saved changes automatically invalidate the previous timetable so stale results are not reused.
                </div>

                <div class="hero-meta">
                    <div class="hero-meta-item">Storage <b>Local CSV</b></div>
                    <div class="hero-meta-item">Workflow <b>Edit · Validate · Save</b></div>
                    <div class="hero-meta-item">Data <b>Live on rerun</b></div>
                </div>
            </div>

            <div class="schedule-art">
                <div class="schedule-shell"></div>
                <div class="schedule-head">
                    <span>DATA PIPELINE</span>
                    <span class="schedule-live">● SYNC READY</span>
                </div>

                <div style="position:absolute;inset:52px 28px 25px;display:flex;flex-direction:column;gap:9px;">
                    <div style="padding:12px;border:1px solid rgba(139,212,255,.11);border-radius:10px;background:rgba(139,212,255,.035);color:#93a5b8;font-size:9px;font-weight:800;">
                        COURSES <span style="float:right;color:#d9eef8;">SOURCE</span>
                    </div>
                    <div style="padding:12px;border:1px solid rgba(139,212,255,.09);border-radius:10px;background:rgba(255,255,255,.012);color:#77899d;font-size:9px;font-weight:800;">
                        FACULTY <span style="float:right;color:#b6c4d2;">LINKED</span>
                    </div>
                    <div style="padding:12px;border:1px solid rgba(104,224,175,.11);border-radius:10px;background:rgba(104,224,175,.035);color:#7f988f;font-size:9px;font-weight:800;">
                        ROOMS + SECTIONS <span style="float:right;color:#9be0c5;">READY</span>
                    </div>
                    <div style="padding:12px;border:1px solid rgba(140,123,255,.11);border-radius:10px;background:rgba(140,123,255,.035);color:#7e819f;font-size:9px;font-weight:800;">
                        TIME SLOTS <span style="float:right;color:#b8b1f5;">MAPPED</span>
                    </div>
                </div>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="eyebrow">WORKSPACE DATA</div>
        <div class="section-title">Manage scheduling inputs</div>
        <div class="section-desc">
            The scheduler reads the latest saved CSV data every time you generate a timetable.
        </div>
        """
    )

    validation = validate_dataset(data)
    issue_count = len(validation["issues"])
    warning_count = len(validation["warnings"])

    if issue_count == 0 and warning_count == 0:
        render_html(
            """
            <div class="validation-panel">
                <div class="validation-title">Data health · All checks clear</div>
                <div class="validation-item">
                    No empty datasets, duplicate standard IDs or obvious orphan references were detected.
                </div>
            </div>
            """
        )
    else:
        items = []
        if issue_count:
            items.append(f'<div class="validation-item"><b>{issue_count}</b> issue(s) require attention.</div>')
        if warning_count:
            items.append(f'<div class="validation-item"><b>{warning_count}</b> warning(s) detected.</div>')
        for item in validation["issues"][:6]:
            items.append(f'<div class="validation-item">• {html_lib.escape(item)}</div>')
        for item in validation["warnings"][:4]:
            items.append(f'<div class="validation-item">• {html_lib.escape(item)}</div>')
        render_html(
            '<div class="validation-panel">'
            '<div class="validation-title">Data health</div>'
            + "".join(items)
            + "</div>"
        )

    tabs = st.tabs(["Courses", "Faculty", "Classrooms", "Sections", "Time Slots"])
    specs = [
        ("courses", "Courses", "data/courses.csv", tabs[0]),
        ("faculty", "Faculty", "data/faculty.csv", tabs[1]),
        ("rooms", "Classrooms", "data/rooms.csv", tabs[2]),
        ("sections", "Sections", "data/sections.csv", tabs[3]),
        ("timeslots", "Time Slots", "data/timeslots.csv", tabs[4]),
    ]

    for key, title, csv_path, tab in specs:
        with tab:
            frame = data[key].copy()

            search_col, sort_col = st.columns([1.4, 1])
            with search_col:
                query = st.text_input(
                    "Search",
                    placeholder=f"Search {title.lower()}...",
                    key=f"data_search_{key}",
                    label_visibility="collapsed",
                )
            with sort_col:
                sort_choices = ["Original order"] + list(frame.columns[:3])
                sort_choice = st.selectbox(
                    "Sort",
                    sort_choices,
                    key=f"data_sort_{key}",
                    label_visibility="collapsed",
                )

            preview = frame.copy()
            if query.strip():
                q = query.strip().lower()
                mask = pd.Series(False, index=preview.index)
                for column in preview.columns:
                    mask |= preview[column].astype(str).str.lower().str.contains(q, na=False)
                preview = preview[mask]
            if sort_choice != "Original order" and sort_choice in preview.columns:
                preview = preview.sort_values(sort_choice, kind="stable")

            st.caption(f"{len(preview)} matching record(s) · editor below contains the complete dataset")

            with st.expander("Search preview", expanded=bool(query.strip())):
                st.dataframe(
                    preview,
                    use_container_width=True,
                    hide_index=True,
                    height=190,
                )

            edited = st.data_editor(
                frame,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key=f"data_editor_{key}",
            )

            changed = not edited.equals(frame)
            if changed:
                render_html(
                    '<div class="data-note" style="color:#f2cb76;">Unsaved changes detected · validate before saving.</div>'
                )
            else:
                render_html(
                    '<div class="data-note" style="color:#79d5ae;">Dataset is in sync with the saved CSV.</div>'
                )

            action1, action2, action3 = st.columns([1, 1, 1])

            with action1:
                if st.button(
                    f"Validate {title}",
                    key=f"validate_button_{key}",
                    use_container_width=True,
                ):
                    temp = dict(data)
                    temp[key] = edited
                    result = validate_dataset(temp)
                    if not result["issues"] and not result["warnings"]:
                        st.success("No validation issues detected.")
                    else:
                        for item in result["issues"][:8]:
                            st.warning(item)
                        for item in result["warnings"][:5]:
                            st.info(item)

            with action2:
                if st.button(
                    f"Save {title}",
                    key=f"save_button_{key}",
                    type="primary" if changed else "secondary",
                    use_container_width=True,
                ):
                    try:
                        temp = dict(data)
                        temp[key] = edited
                        result = validate_dataset(temp)
                        blocking = result["issues"]

                        if blocking:
                            st.error(
                                "Save blocked because the edited dataset contains validation issues. "
                                "Use Validate to review them."
                            )
                            for item in blocking[:8]:
                                st.write(f"• {item}")
                        else:
                            edited.to_csv(csv_path, index=False)
                            for state_key in [
                                "timetable", "score", "generated_at",
                                "ai_answer", "ai_question", "ai_history",
                                f"data_editor_{key}",
                            ]:
                                st.session_state.pop(state_key, None)
                            st.toast(f"{title} saved. Generate a new timetable.")
                            st.rerun()
                    except Exception as error:
                        st.error(f"Could not save {title}: {error}")

            with action3:
                if st.button(
                    "Reload saved data",
                    key=f"reload_button_{key}",
                    use_container_width=True,
                ):
                    st.session_state.pop(f"data_editor_{key}", None)
                    st.rerun()


# =========================================================

# =========================================================
# PROJECT DOCUMENTATION
# =========================================================

elif page == "Project":

    render_html(
        """
        <div class="hero">
            <div class="hero-grid"></div>
            <div class="hero-scan"></div>
            <div class="hero-ring"></div>

            <div class="hero-content">
                <div class="hero-badge">
                    <span class="dot"></span>
                    PROJECT DOCUMENTATION
                </div>

                <div class="hero-title">
                    SmartSched <span>AI.</span>
                </div>

                <div class="hero-description">
                    An intelligent college timetable optimization system that combines
                    source-data management, constraint-aware scheduling, optimization,
                    visual timetable exploration, analytics and a local AI assistant.
                </div>

                <div class="hero-meta">
                    <div class="hero-meta-item">Platform <b>Streamlit</b></div>
                    <div class="hero-meta-item">Scheduler <b>Constraint-aware</b></div>
                    <div class="hero-meta-item">AI <b>Local Ollama</b></div>
                    <div class="hero-meta-item">Reports <b>CSV · Excel · PDF</b></div>
                </div>
            </div>

            <div class="schedule-art">
                <div class="schedule-shell"></div>
                <div class="schedule-head">
                    <span>SYSTEM FLOW</span>
                    <span class="schedule-live">● READY</span>
                </div>

                <div style="position:absolute;left:30px;right:30px;top:55px;">
                    <div style="padding:13px;border:1px solid rgba(139,212,255,.13);border-radius:10px;background:rgba(139,212,255,.035);color:#a6bdcf;font-size:8px;font-weight:900;letter-spacing:1px;text-align:center;">
                        SOURCE DATA
                    </div>
                    <div style="height:12px;border-left:1px dashed rgba(139,212,255,.14);margin-left:50%;"></div>
                    <div style="padding:13px;border:1px solid rgba(139,212,255,.13);border-radius:10px;background:rgba(139,212,255,.025);color:#a6bdcf;font-size:8px;font-weight:900;letter-spacing:1px;text-align:center;">
                        SCHEDULER → OPTIMIZER
                    </div>
                    <div style="height:12px;border-left:1px dashed rgba(104,224,175,.14);margin-left:50%;"></div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;">
                        <div style="padding:12px;border:1px solid rgba(104,224,175,.12);border-radius:10px;background:rgba(104,224,175,.03);color:#8ecfb3;font-size:8px;font-weight:900;text-align:center;">
                            TIMETABLE
                        </div>
                        <div style="padding:12px;border:1px solid rgba(140,123,255,.12);border-radius:10px;background:rgba(140,123,255,.03);color:#aaa3e9;font-size:8px;font-weight:900;text-align:center;">
                            AI ASSISTANT
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="eyebrow">01 · PROBLEM STATEMENT</div>
        <div class="section-title">Why SmartSched AI exists</div>
        <div class="section-desc">
            College timetable creation requires multiple resources to be placed together without
            violating scheduling constraints. SmartSched AI provides a single workspace for managing
            those inputs, generating a schedule, checking the result and understanding the output.
        </div>
        <div style="height:13px"></div>
        """
    )

    render_html(
        """
        <div class="project-feature" style="min-height:0;">
            <div class="project-feature-title">Problem</div>
            <div class="project-feature-copy">
                Manual timetable preparation can become difficult when courses, faculty, classrooms,
                sections, room capacities, availability and time slots must all be considered together.
                The system automates the scheduling workflow and presents the result in an understandable form.
            </div>
        </div>
        """
    )

    render_html(
        """
        <div style="height:18px"></div>
        <div class="eyebrow">02 · OBJECTIVES</div>
        <div class="section-title">System objectives</div>
        <div style="height:10px"></div>
        """
    )

    objectives = [
        ("Automate scheduling", "Generate a timetable from structured academic scheduling data."),
        ("Reduce conflicts", "Check faculty, room, section, capacity and availability constraints."),
        ("Improve usability", "Present the generated timetable as a weekly visual grid and detailed table."),
        ("Measure quality", "Show optimization score, coverage and schedule-health indicators."),
        ("Add intelligence", "Allow natural-language questions through the local AI assistant."),
        ("Enable reporting", "Export schedules in formats suitable for analysis, sharing and printing."),
    ]

    cols = st.columns(3)
    for idx, (title, copy) in enumerate(objectives):
        with cols[idx % 3]:
            render_html(
                f"""
                <div class="project-feature" style="margin-bottom:10px;min-height:105px;">
                    <div class="project-feature-title">{html_lib.escape(title)}</div>
                    <div class="project-feature-copy">{html_lib.escape(copy)}</div>
                </div>
                """
            )

    render_html(
        """
        <div style="height:18px"></div>
        <div class="eyebrow">03 · SYSTEM ARCHITECTURE</div>
        <div class="section-title">How the modules work together</div>
        <div class="section-desc">
            The application is organized into separate layers so the scheduling logic can remain independent
            from the user interface and reporting features.
        </div>
        <div style="height:10px"></div>
        """
    )

    arch1, arch2, arch3, arch4 = st.columns(4)
    architecture = [
        (arch1, "DATA LAYER", "CSV datasets", "Courses, faculty, rooms, sections and time slots."),
        (arch2, "SCHEDULING", "Constraint engine", "Builds valid assignments using the project's scheduling rules."),
        (arch3, "OPTIMIZATION", "Quality layer", "Returns the generated schedule and optimization score."),
        (arch4, "PRESENTATION", "UI + AI", "Visual timetable, analytics, exports and local AI assistant."),
    ]
    for col, kicker, title, copy in architecture:
        with col:
            render_html(
                f"""
                <div class="project-feature">
                    <div style="color:#6faed0;font-size:8px;font-weight:900;letter-spacing:1px;">{kicker}</div>
                    <div class="project-feature-title" style="margin-top:5px;">{title}</div>
                    <div class="project-feature-copy">{copy}</div>
                </div>
                """
            )

    render_html(
        """
        <div style="height:18px"></div>
        <div class="eyebrow">04 · SCHEDULING CONSTRAINTS</div>
        <div class="section-title">Rules the system considers</div>
        <div style="height:10px"></div>
        """
    )

    constraints = [
        ("Faculty conflict", "A faculty member should not be assigned to overlapping classes."),
        ("Room conflict", "A classroom should not hold multiple overlapping classes."),
        ("Section conflict", "A section should not have overlapping classes."),
        ("Room capacity", "The assigned room must be suitable for the section size."),
        ("Faculty availability", "Assignments must respect available faculty periods."),
    ]

    for title, copy in constraints:
        render_html(
            f"""
            <div style="display:flex;gap:10px;align-items:flex-start;padding:10px 12px;margin-bottom:6px;border:1px solid rgba(137,161,188,.08);border-radius:10px;background:rgba(9,14,24,.48);">
                <div style="color:#79d5ae;font-size:9px;font-weight:900;">✓</div>
                <div>
                    <div style="color:#dbe6ed;font-size:10px;font-weight:850;">{html_lib.escape(title)}</div>
                    <div style="color:#6e8094;font-size:9px;line-height:1.5;margin-top:2px;">{html_lib.escape(copy)}</div>
                </div>
            </div>
            """
        )

    render_html(
        """
        <div style="height:18px"></div>
        <div class="eyebrow">05 · AI INTEGRATION</div>
        <div class="section-title">Local timetable assistant</div>
        <div class="section-desc">
            The AI assistant is connected to the generated timetable through the existing
            <b>ask_timetable_assistant()</b> module and uses the local Ollama/Llama 3.2 setup.
        </div>
        <div style="height:10px"></div>
        """
    )

    render_html(
        """
        <div class="project-feature" style="min-height:0;">
            <div class="project-feature-title">Example questions</div>
            <div class="project-feature-copy">
                Which room is used the most? · How many classes are scheduled on Monday?
                · Which courses are handled by F001? · Are there any timetable conflicts?
            </div>
        </div>
        """
    )

    render_html(
        """
        <div style="height:18px"></div>
        <div class="eyebrow">06 · TECHNOLOGY STACK</div>
        <div class="section-title">Technologies used in the project</div>
        <div style="margin-top:9px;">
            <span class="stack-chip">Python</span>
            <span class="stack-chip">Streamlit</span>
            <span class="stack-chip">Pandas</span>
            <span class="stack-chip">Plotly</span>
            <span class="stack-chip">Ollama</span>
            <span class="stack-chip">Llama 3.2</span>
            <span class="stack-chip">CSV</span>
            <span class="stack-chip">XLSXWriter</span>
            <span class="stack-chip">ReportLab</span>
        </div>
        """
    )

    render_html(
        """
        <div style="height:18px"></div>
        <div class="eyebrow">07 · PROJECT WORKFLOW</div>
        <div class="section-title">End-to-end flow</div>
        <div style="height:10px"></div>
        """
    )

    flow = [
        ("01", "Load", "Read the latest source datasets."),
        ("02", "Validate", "Check identifiers and references."),
        ("03", "Generate", "Create assignments using the scheduler."),
        ("04", "Optimize", "Evaluate the generated schedule."),
        ("05", "Explore", "Inspect the visual grid, filters and analytics."),
        ("06", "Ask / Export", "Use the AI assistant or export the result."),
    ]

    flow_cols = st.columns(3)
    for idx, (number, title, copy) in enumerate(flow):
        with flow_cols[idx % 3]:
            render_html(
                f"""
                <div class="project-feature" style="margin-bottom:10px;min-height:102px;">
                    <div style="color:#6faed0;font-size:8px;font-weight:900;">STEP {number}</div>
                    <div class="project-feature-title" style="margin-top:5px;">{title}</div>
                    <div class="project-feature-copy">{copy}</div>
                </div>
                """
            )

    render_html(
        """
        <div style="height:18px"></div>
        <div class="eyebrow">08 · FUTURE SCOPE</div>
        <div class="section-title">Possible next-generation additions</div>
        <div class="section-desc">
            These are extension ideas, not requirements for the current implementation.
        </div>
        <div style="height:10px"></div>
        """
    )

    future = [
        ("Multi-department scheduling", "Combine multiple departments or academic programs in one planning run."),
        ("Advanced optimization", "Add weighted preferences for gaps, workload balance and room utilization."),
        ("Authentication", "Introduce user roles for administrators, faculty and viewers."),
        ("Database backend", "Move from CSV storage to SQLite or a production relational database."),
        ("Automated notifications", "Notify stakeholders when timetables are generated or changed."),
        ("Cloud deployment", "Host the application for shared access across a college network."),
    ]

    future_cols = st.columns(3)
    for idx, (title, copy) in enumerate(future):
        with future_cols[idx % 3]:
            render_html(
                f"""
                <div class="project-feature" style="margin-bottom:10px;min-height:105px;">
                    <div class="project-feature-title">{title}</div>
                    <div class="project-feature-copy">{copy}</div>
                </div>
                """
            )

    render_html(
        """
        <div style="height:21px"></div>
        <div class="validation-panel">
            <div class="validation-title">Presentation-ready summary</div>
            <div class="validation-item">
                <b>SmartSched AI</b> is a complete academic timetable workflow:
                editable source data → constraint-aware scheduling → optimization →
                visual timetable → analytics → AI-assisted understanding → export.
            </div>
        </div>
        """
    )


# FOOTER
# =========================================================

render_html(
    """
    <div class="footer">
        SmartSched AI · Intelligent College Timetable Optimization System
        · Constraint-aware scheduling · Analytics · Local AI
    </div>
    """
)
