import uuid
import streamlit as st
from PIL import Image
from detector import detect, CLASSES
from db import save_report, ranking, open_reports, mark_duplicate

st.set_page_config(page_title="WasteWise AI", page_icon="♻️", layout="wide")
st.markdown("""<style>
h1,h2,h3{color:#1b5e20}
div[data-testid="stMetric"]{background:#eef6ee;border-radius:10px;padding:10px 14px;border:1px solid #c8e6c9}
.stButton>button,.stFormSubmitButton>button{border-radius:8px;font-weight:600}
</style>""", unsafe_allow_html=True)

SEV_LABEL = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}

st.title("♻️ WasteWise AI")
st.caption("Spot classroom waste with Grounding DINO Tiny, report it, and see which rooms to clean first.")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("1. Upload & detect")
    f = st.file_uploader("Upload a classroom photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f)
        if st.button("🔍 Detect waste", type="primary", width="stretch"):
            with st.spinner("Running Grounding DINO (first run downloads the model)..."):
                st.session_state["result"] = detect(img)
                st.session_state["token"] = uuid.uuid4().hex
                st.session_state["photo"] = f.getvalue()
        if "result" not in st.session_state:
            st.image(img, caption="Uploaded photo", width=int(img.width*min(1, 450/img.height)))
    if "result" in st.session_state:
        ann, counts, dets = st.session_state["result"]
        if sum(counts.values()) == 0:
            st.warning("No supported items (crumpled paper, plastic bottles, food packaging, aluminum cans) detected.")
        else:
            for col, (k, v) in zip(st.columns(len(CLASSES)), counts.items()):
                col.metric(k.title(), v)
        st.image(ann, caption="Detected items", width="content")

with right:
    st.subheader("2. Report incident")
    if "result" not in st.session_state:
        st.info("Detect waste in a photo first, then confirm the report here.")
    else:
        token = st.session_state["token"]
        with st.form("report_form"):
            room = st.selectbox("Classroom", ["Room 101", "Room 102", "Room 103", "Room 104", "Room 105"])
            sev = st.radio("Confirm severity", list(SEV_LABEL), format_func=SEV_LABEL.get, horizontal=True)
            if st.form_submit_button("Submit report", type="primary", width="stretch"):
                rid = save_report(token, room, sev, st.session_state["result"][1], st.session_state["photo"])
                st.session_state["saved"] = (token, rid)
        saved = st.session_state.get("saved")
        if saved and saved[0] == token:
            st.success(f"Report #{saved[1]} saved (Status: Open).")

st.divider()
st.header("🧹 Cleanup Priority")
rank = ranking()
if not rank:
    st.info("No open reports yet")
else:
    for r in rank:
        r["Highest severity"] = SEV_LABEL[r["Highest severity"]]
    st.dataframe(rank, hide_index=True, width="stretch")
    with st.expander("Open reports (mark repeat photos as Duplicate)"):
        for rid, room, sev, cnts, ts in open_reports():
            a, b = st.columns([5, 1])
            a.write(f"#{rid} · {room} · {SEV_LABEL[sev]} · {ts}")
            if b.button("Duplicate", key=f"dup{rid}"):
                mark_duplicate(rid)
                st.rerun()
