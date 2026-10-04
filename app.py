import streamlit as st
import cv2
import numpy as np
import pandas as pd
import easyocr
import re
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Receipt Scanner",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Smart Receipt Scanner")
st.write("Upload a receipt to extract its items, quantities and prices.")

@st.cache_resource
def load_reader():
    return easyocr.Reader(["en"], gpu=False)

reader = load_reader()

def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def correct_receipt(image):
    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    resized = cv2.resize(
        image, (int(w * scale), int(h * scale))
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )

    contour = None
    for c in contours[:20]:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            contour = approx
            break

    if contour is None:
        return image, False

    corners = contour.reshape(4, 2).astype("float32") / scale
    rect = order_points(corners)
    tl, tr, br, bl = rect

    out_w = int(max(
        np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    ))
    out_h = int(max(
        np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    ))

    if out_w < 2 or out_h < 2:
        return image, False

    destination = np.array([
        [0, 0], [out_w - 1, 0],
        [out_w - 1, out_h - 1], [0, out_h - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    return warped, True

def parse_receipt(results):
    # Demo parser for the sample receipt layout.
    # Extend item recognition for other receipt formats.
    known_items = [
        "APPLE", "BANANA", "ORANGE", "PEAR", "GRAPES",
        "STRAWBERRY", "BLUEBERRY", "KIWI", "WATERMELON",
        "LEMON", "RASPBERRY", "MILK", "CHEESE", "YOGURT"
    ]

    entries = []
    for bbox, text, confidence in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        entries.append({
            "text": text.strip().upper().replace(" ", ""),
            "x": (min(xs) + max(xs)) / 2,
            "y": (min(ys) + max(ys)) / 2
        })

    items = [e for e in entries if e["text"] in known_items]
    prices = [
        e for e in entries
        if re.fullmatch(r"\d+\.\s*\d{2}", e["text"])
    ]
    quantities = [
        e for e in entries
        if re.fullmatch(r"\d+", e["text"])
        and e["x"] < 65
    ]

    parsed = []
    used_prices = set()

    for item in items:
        candidates = [
            (i, p) for i, p in enumerate(prices)
            if p["x"] > item["x"]
            and i not in used_prices
            and abs(p["y"] - item["y"]) <= 18
        ]

        if not candidates:
            parsed.append([1, item["text"], None])
            continue

        price_idx, price = min(
            candidates,
            key=lambda pair: abs(pair[1]["y"] - item["y"])
        )
        used_prices.add(price_idx)

        nearby_quantities = [
            q for q in quantities
            if q["x"] < item["x"]
            and abs(q["y"] - item["y"]) <= 8
        ]
        quantity = 1
        if nearby_quantities:
            quantity = int(min(
                nearby_quantities,
                key=lambda q: abs(q["y"] - item["y"])
            )["text"])

        parsed.append([
            quantity,
            item["text"],
            float(price["text"].replace(" ", ""))
        ])

    return pd.DataFrame(
        parsed, columns=["Quantity", "Item", "Price"]
    )

uploaded = st.file_uploader(
    "Choose a receipt image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    pil_image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
    rgb = np.array(pil_image)
    original = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Processing receipt..."):
        corrected, detected = correct_receipt(original)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        ocr_results = reader.readtext(corrected_rgb)
        df = parse_receipt(ocr_results)

    left, right = st.columns(2)
    with left:
        st.subheader("Original Receipt")
        st.image(rgb, use_container_width=True)
    with right:
        st.subheader("Processed Receipt")
        st.image(corrected_rgb, use_container_width=True)
        if detected:
            st.success("Receipt boundary detected and corrected.")
        else:
            st.info("No clear boundary detected. Used original image.")

    st.subheader("Extracted Text")
    if ocr_results:
        st.text("\n".join(text for _, text, _ in ocr_results))
    else:
        st.warning("No text detected. Try a clearer image.")

    st.subheader("Extracted Items")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)

        if df["Price"].notna().all():
            total = round(df["Price"].sum(), 2)
            st.metric("Calculated Total", f"{total:.2f}")
        else:
            st.warning("Some prices could not be matched. Review the table.")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download items as CSV",
            data=csv,
            file_name="receipt_items.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "No supported items were recognized by this demo parser. "
            "The current parser is tailored to the sample receipt."
        )
