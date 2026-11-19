import json, os, sqlite3
from datetime import datetime

DB_PATH = os.environ.get("WASTEWISE_DB", os.path.join(os.path.dirname(__file__), "wastewise.db"))


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT UNIQUE NOT NULL,
        classroom TEXT NOT NULL,
        severity TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Open',
        counts TEXT NOT NULL,
        labels TEXT NOT NULL,
        photo BLOB NOT NULL,
        created_at TEXT NOT NULL)""")
    return c


def save_report(token, classroom, severity, counts, photo_bytes):
    """Insert once per token; returns the report id (existing id if token repeats)."""
    labels = [k for k, v in counts.items() if v > 0]
    with _conn() as c:
        row = c.execute("SELECT id FROM reports WHERE token=?", (token,)).fetchone()
        if row:
            return row[0]
        c.execute("INSERT OR IGNORE INTO reports (token, classroom, severity, status, counts, labels, photo, created_at)"
                  " VALUES (?,?,?,?,?,?,?,?)",
                  (token, classroom, severity, "Open", json.dumps(counts), json.dumps(labels),
                   photo_bytes, datetime.now().isoformat(timespec="seconds")))
        return c.execute("SELECT id FROM reports WHERE token=?", (token,)).fetchone()[0]


SEV = {"Low": 1, "Medium": 2, "High": 3}


def open_reports():
    with _conn() as c:
        return c.execute("SELECT id, classroom, severity, counts, created_at FROM reports "
                         "WHERE status='Open' ORDER BY id DESC").fetchall()


def mark_duplicate(report_id):
    with _conn() as c:
        c.execute("UPDATE reports SET status='Duplicate' WHERE id=? AND status='Open'", (report_id,))


def ranking():
    """Open reports grouped by classroom: highest severity, then open count, then name."""
    rooms = {}
    for _id, room, sev, _c, _t in open_reports():
        rooms.setdefault(room, []).append(sev)
    rows = []
    for room, sevs in rooms.items():
        top = max(sevs, key=SEV.get)
        n = len(sevs)
        mix = ", ".join(f"{sevs.count(s)} {s}" for s in ("High", "Medium", "Low") if s in sevs)
        reason = f"Highest severity is {top}; {n} open incident{'s' if n != 1 else ''} ({mix})."
        rows.append((room, n, top, reason))
    rows.sort(key=lambda r: (-SEV[r[2]], -r[1], r[0]))
    return [{"Rank": i + 1, "Classroom": r[0], "Open incidents": r[1], "Highest severity": r[2], "Reason": r[3]}
            for i, r in enumerate(rows)]
