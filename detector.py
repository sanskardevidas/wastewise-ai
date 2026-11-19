import torch
from PIL import ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CLASSES = ["crumpled paper", "plastic bottle", "food packaging", "aluminum drink can"]
# (prompts, label map by substring, focused?) per pass; threshold applies to all passes
PASSES = {
    "main": (["crumpled paper", "plastic bottle", "snack wrapper", "aluminum drink can"],
             {"crumpled paper": "crumpled paper", "plastic bottle": "plastic bottle",
              "snack": "food packaging", "aluminum": "aluminum drink can"}, False),
    "food": (["paper food bag", "fast food wrapper", "takeout container"],
             {"food": "food packaging", "takeout": "food packaging", "wrapper": "food packaging",
              "bag": "food packaging"}, False),
    "can": (["crushed soda can", "aluminum beverage can"],
            {"can": "aluminum drink can", "aluminum": "aluminum drink can", "soda": "aluminum drink can",
             "beverage": "aluminum drink can"}, True),
    "bottle": (["plastic bottle"], {"bottle": "plastic bottle"}, True),
}
COLORS = {"crumpled paper": "#f59e0b", "plastic bottle": "#2563eb", "aluminum drink can": "#7c3aed", "food packaging": "#0d9488"}
_cache = {}


def _load():
    if not _cache:
        _cache["p"] = AutoProcessor.from_pretrained(MODEL_ID)
        _cache["m"] = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).eval()
    return _cache["p"], _cache["m"]


def _run(image, prompts, box_thr, text_thr=0.25):
    proc, model = _load()
    inputs = proc(images=image, text=". ".join(prompts) + ".", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[image.size[::-1]])[0]


def raw_detections(image, box_threshold=0.30, text_threshold=0.25):
    """All thresholded candidates from every pass (after the full-frame filter), tagged with their pass."""
    area = image.width * image.height
    raw = []
    for pname, (prompts, lmap, focused) in PASSES.items():
        res = _run(image, prompts, box_threshold, text_threshold)
        for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
            name = next((v for k, v in lmap.items() if k in label.strip().lower()), None)
            b = [round(float(v), 1) for v in box]
            if name and (b[2] - b[0]) * (b[3] - b[1]) < 0.85 * area:
                raw.append({"label": name, "score": round(float(score), 2), "box": b,
                            "pass": pname, "focused": focused})
    return raw


def detect(image, box_threshold=0.30, text_threshold=0.25):
    """Return (annotated_image, counts dict, detections list)."""
    image = image.convert("RGB")
    dets = resolve(raw_detections(image, box_threshold, text_threshold))
    # preview: max 450px tall, aspect preserved, whole photo kept
    scale = min(1.0, PREVIEW_H / image.height)
    annotated = image.resize((round(image.width * scale), round(image.height * scale))) if scale < 1 else image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _font(16)
    for d in dets:
        b = [v * scale for v in d["box"]]
        c = COLORS[d["label"]]
        draw.rectangle(b, outline=c, width=3)
        txt = f"{d['label']} {d['score']:.2f}"
        l, t, r, bt = draw.textbbox((0, 0), txt, font=font)
        w, h = r - l + 8, bt - t + 6
        y0 = b[1] - h if b[1] - h >= 0 else b[1]
        x0 = min(max(b[0], 0), max(annotated.width - w, 0))
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=c)
        draw.text((x0 + 4, y0 + 3 - t), txt, fill="white", font=font)
    counts = {k: sum(d["label"] == k for d in dets) for k in CLASSES}
    return annotated, counts, dets


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ar = lambda x: max(0, x[2] - x[0]) * max(0, x[3] - x[1])
    return inter / (ar(a) + ar(b) - inter + 1e-9)


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _contain(small, big):
    return _inter(small, big) / (_area(small) + 1e-9)


def _same_object(a, b):
    if _iou(a, b) >= 0.6:
        return True
    lo, hi = (a, b) if _area(a) <= _area(b) else (b, a)
    # containment, but only if the smaller box is a substantial part of the larger one
    return _contain(lo, hi) >= 0.85 and _area(lo) >= 0.4 * _area(hi)


BEVERAGE = {"plastic bottle", "aluminum drink can"}


def resolve(raw):
    """Cluster boxes that describe one object (overlap or containment), pick one label per cluster.
    Focused can/bottle passes are the class evidence: highest-scoring focused hit wins, unless a clearly
    tighter focused box of another class sits inside it (the larger box is then just a loose extension)."""
    n = len(raw)
    parent = list(range(n))
    find = lambda i: i if parent[i] == i else find(parent[i])
    for i in range(n):
        for j in range(i + 1, n):
            if _same_object(raw[i]["box"], raw[j]["box"]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(raw[i])
    out = []
    for g in groups.values():
        foc = [d for d in g if d["focused"]]
        gen = [d for d in g if not d["focused"]]
        # single-prompt focused passes fire on any object, so trust them only when a beverage-container
        # label is corroborated by a generic pass
        corroborated = any(d["label"] in BEVERAGE for d in gen)
        if not foc or not corroborated:
            if gen:
                out.append(max(gen, key=lambda d: d["score"]))
            continue
        best = max(foc, key=lambda d: d["score"])
        tight = [d for d in foc if d["label"] != best["label"] and _area(d["box"]) < 0.7 * _area(best["box"])
                 and _contain(d["box"], best["box"]) >= 0.85]
        out.append(max(tight, key=lambda d: d["score"]) if tight else best)
    return out


PREVIEW_H = 450


def _font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()
