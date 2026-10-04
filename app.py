import streamlit as st
import cv2
import numpy as np
import pandas as pd
import easyocr
import re
from PIL import Image, ImageOps
from io import BytesIO

st.set_page_config(page_title="Smart Receipt Scanner", page_icon="🧾", layout="wide")
st.title("🧾 Smart Receipt Scanner")
st.write("Upload a receipt image to extract item descriptions, quantities and prices.")

@st.cache_resource
def load_reader():
    return easyocr.Reader(["en"], gpu=False)

reader = load_reader()

def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(diff)],
        pts[np.argmax(s)], pts[np.argmax(diff)]
    ], dtype="float32")

def correct_receipt(image):
    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contour = None
    image_area = resized.shape[0] * resized.shape[1]
    for c in contours[:30]:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and image_area * 0.08 < area < image_area * 0.98:
            contour = approx
            break

    if contour is None:
        return image, False

    rect = order_points(contour.reshape(4, 2).astype("float32") / scale)
    tl, tr, br, bl = rect
    out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if out_w < 2 or out_h < 2:
        return image, False

    destination = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (out_w, out_h)), True

# Common totals/metadata labels are excluded from item rows.
SUMMARY_LABELS = {
    "total", "totalamount", "subtotal", "subtot", "tax", "salestax",
    "vat", "mwst", "balance", "balancedue", "change", "cash",
    "amountdue", "grandtotal", "tip", "servicecharge", "thankyou"
}

def money_value(text):
    """Return numeric value for a money-like OCR token, else None."""
    s = text.strip().replace(" ", "")
    s = re.sub(r"(?i)(CHF|USD|EUR|GBP|INR|Rs\.?|[$€£₹])", "", s)
    s = s.strip(":=*")
    # Require decimal cents or a currency marker was removed; avoid treating dates/IDs as prices.
    if not re.fullmatch(r"\d{1,7}[.,]\d{2}", s):
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None

def group_ocr_rows(results):
    entries = []
    heights = []
    for bbox, text, confidence in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        height = max(1, max(ys) - min(ys))
        heights.append(height)
        entries.append({
            "text": text.strip(),
            "x1": min(xs), "x2": max(xs),
            "y": (min(ys) + max(ys)) / 2,
            "height": height,
            "confidence": float(confidence)
        })

    if not entries:
        return []
    tolerance = max(8, float(np.median(heights)) * 0.65)
    entries.sort(key=lambda e: e["y"])
    rows = []
    for entry in entries:
        if not rows or abs(entry["y"] - rows[-1]["y"]) > tolerance:
            rows.append({"y": entry["y"], "entries": [entry]})
        else:
            row = rows[-1]
            row["entries"].append(entry)
            row["y"] = sum(e["y"] for e in row["entries"]) / len(row["entries"])
    for row in rows:
        row["entries"].sort(key=lambda e: e["x1"])
    return rows

def parse_receipt(results):
    rows = group_ocr_rows(results)
    parsed = []
    summary = []
    raw_lines = []

    for row in rows:
        parts = row["entries"]
        line = " ".join(p["text"] for p in parts).strip()
        if not line:
            continue
        raw_lines.append(line)
        normalized = re.sub(r"[^A-Z]", "", line.upper())

        # Summary rows are kept separately and never treated as purchased items.
        if any(label in normalized.lower() for label in SUMMARY_LABELS):
            amounts = [(p, money_value(p["text"])) for p in parts]
            amounts = [(p, v) for p, v in amounts if v is not None]
            if amounts:
                summary.append({"Label": line, "Amount": amounts[-1][1]})
            continue

        # Find price tokens, allowing currency symbols and decimal comma.
        price_parts = [(p, money_value(p["text"])) for p in parts]
        price_parts = [(p, v) for p, v in price_parts if v is not None]
        if not price_parts:
            continue

        # Usually the line amount is the rightmost money-like token.
        price_part, price = max(price_parts, key=lambda pv: pv[0]["x2"])
        description_parts = [p for p in parts if p is not price_part and money_value(p["text"]) is None]
        description = " ".join(p["text"] for p in description_parts).strip(" -:|")
        if not description:
            continue

        # Extract a quantity when written as "2 x item", "2x item", or "2 item".
        quantity = 1
        qty_match = re.match(r"^\s*(\d{1,3})\s*[xX×]\s*", description)
        if qty_match:
            quantity = int(qty_match.group(1))
            description = description[qty_match.end():].strip()
        else:
            qty_match = re.match(r"^\s*(\d{1,3})\s+(.+)$", description)
            if qty_match and not re.fullmatch(r"\d{1,3}", qty_match.group(2).strip()):
                quantity = int(qty_match.group(1))
                description = qty_match.group(2).strip()

        cleaned = re.sub(r"\s+", " ", description).strip()
        letters = sum(ch.isalpha() for ch in cleaned)
        if letters < 2 or cleaned.isdigit():
            continue
        parsed.append({"Quantity": quantity, "Item": cleaned, "Price": round(price, 2)})

    df = pd.DataFrame(parsed, columns=["Quantity", "Item", "Price"])
    summary_df = pd.DataFrame(summary, columns=["Label", "Amount"])
    return df, summary_df, raw_lines

uploaded = st.file_uploader("Choose a receipt image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        pil_image = ImageOps.exif_transpose(
            Image.open(BytesIO(uploaded.getvalue()))
        ).convert("RGB")
        rgb = np.array(pil_image)
        original = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Processing receipt..."):
            corrected, detected = correct_receipt(original)
            corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
            ocr_results = reader.readtext(corrected_rgb)
            df, summary_df, raw_lines = parse_receipt(ocr_results)

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
        if raw_lines:
            st.text("\n".join(raw_lines))
        else:
            st.warning("No text detected. Try a clearer, well-lit image.")

        st.subheader("Extracted Items")
        if not df.empty:
            st.caption("Review the extracted rows and edit any OCR mistakes before downloading.")
            edited_df = st.data_editor(
                df, use_container_width=True, hide_index=True,
                num_rows="dynamic",
                column_config={
                    "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1),
                    "Item": st.column_config.TextColumn("Item"),
                    "Price": st.column_config.NumberColumn("Line price", min_value=0.0, format="%.2f")
                }
            )
            valid_prices = pd.to_numeric(edited_df["Price"], errors="coerce")
            if valid_prices.notna().all() and len(edited_df):
                st.metric("Sum of extracted item prices", f"{valid_prices.sum():.2f}")
            else:
                st.warning("Some item prices are missing. Please review the table.")

            csv = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download items as CSV", data=csv,
                               file_name="receipt_items.csv", mime="text/csv")
        else:
            st.warning("No item rows could be confidently extracted from this receipt. Check the extracted text.")

        if not summary_df.empty:
            st.subheader("Receipt totals and charges")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.caption("These values are extracted separately and may need manual verification.")
    except Exception as e:
        st.error(f"Could not process this image: {e}")
        st.info("Try a PNG, JPG or JPEG image with readable text.")
