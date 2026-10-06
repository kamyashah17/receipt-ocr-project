import streamlit as st
import cv2
import numpy as np
import pandas as pd
import easyocr
import re
from PIL import Image, ImageOps
from io import BytesIO


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Receipt OCR",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Automatic Receipt OCR")
st.write(
    "Upload one or more receipts. The original receipt is preserved, "
    "while a processed copy is used internally for OCR."
)


# ============================================================
# EASY OCR MODEL
# ============================================================

@st.cache_resource
def load_reader():
    return easyocr.Reader(
        ['en'],
        gpu=False
    )


reader = load_reader()


# ============================================================
# RECEIPT CORNER ORDERING
# ============================================================

def order_points(points):
    """
    Safely order four receipt corner points as:
    top-left, top-right, bottom-right, bottom-left
    """

    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    # Sort points by Y coordinate
    y_sorted = points[np.argsort(points[:, 1])]

    # Top two and bottom two
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    # Sort top points by X
    top = top[np.argsort(top[:, 0])]

    # Sort bottom points by X
    bottom = bottom[np.argsort(bottom[:, 0])]

    top_left = top[0]
    top_right = top[1]

    bottom_left = bottom[0]
    bottom_right = bottom[1]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def correct_receipt(image):
    """
    Detect receipt boundary and straighten it.
    If no suitable boundary is found, return the original image.
    """

    original = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    height, width = image.shape[:2]

    image_area = height * width

    for contour in contours[:20]:

        area = cv2.contourArea(contour)

        if area < image_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        pts = np.asarray(
            approx,
            dtype=np.float32
        ).reshape(4, 2)

        rect = order_points(pts)

        tl, tr, br, bl = rect

        width_a = np.linalg.norm(
            br - bl
        )

        width_b = np.linalg.norm(
            tr - tl
        )

        max_width = int(
            max(width_a, width_b)
        )

        height_a = np.linalg.norm(
            tr - br
        )

        height_b = np.linalg.norm(
            tl - bl
        )

        max_height = int(
            max(height_a, height_b)
        )

        if max_width < 100 or max_height < 100:
            continue

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ],
            dtype=np.float32
        )

        matrix = cv2.getPerspectiveTransform(
            rect,
            dst
        )

        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height)
        )

        return warped

    return original


# ============================================================
# DESKEW
# ============================================================

def deskew_image(image):
    """
    Correct small rotations/tilts safely.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
        apertureSize=3
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=10
    )

    if lines is None:
        return image

    # Make sure lines always have shape (N, 4)
    lines = np.asarray(lines).reshape(-1, 4)

    angles = []

    for x1, y1, x2, y2 in lines:

        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        # Keep mostly horizontal lines
        if abs(angle) < 20:
            angles.append(angle)

    if not angles:
        return image

    angle = float(np.median(angles))

    # Don't rotate if already almost straight
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]

    center = (
        w // 2,
        h // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_for_ocr(image):
    """
    Improve OCR quality without changing the original image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Remove moderate noise
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        10,
        7,
        21
    )

    # Mild sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        denoised,
        -1,
        sharpen_kernel
    )

    return sharpened


# ============================================================
# OCR RESULT MERGING
# ============================================================

def run_ocr(image):
    """
    Run OCR on multiple versions of the processed receipt.
    """

    results = []

    # -----------------------------
    # Pass 1: normal processed image
    # -----------------------------

    processed = preprocess_for_ocr(image)

    result1 = reader.readtext(
        processed,
        detail=1,
        paragraph=False
    )

    results.extend(result1)

    # -----------------------------
    # Pass 2: adaptive threshold
    # -----------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    result2 = reader.readtext(
        adaptive,
        detail=1,
        paragraph=False
    )

    results.extend(result2)

    # -----------------------------
    # Pass 3: original corrected image
    # -----------------------------

    result3 = reader.readtext(
        image,
        detail=1,
        paragraph=False
    )

    results.extend(result3)

    return results


# ============================================================
# MONEY DETECTION
# ============================================================

MONEY_PATTERN = re.compile(
    r'(?<!\d)(\d{1,7}(?:[.,]\d{2}))(?!\d)'
)


def money_value(text):
    """
    Detect money-like values.
    """

    matches = MONEY_PATTERN.findall(text)

    if not matches:
        return None

    value = matches[-1]

    try:
        return float(
            value.replace(",", ".")
        )

    except:
        return None


# ============================================================
# CURRENCY DETECTION
# ============================================================

def detect_currency(text):
    """
    Detect currency without converting anything.
    """

    text_upper = text.upper()

    currency_symbols = {
        "₹": "INR",
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "CHF": "CHF",
        "CAD": "CAD",
        "AUD": "AUD",
        "¥": "JPY/CNY"
    }

    for symbol, currency in currency_symbols.items():

        if symbol in text:
            return currency

    currency_words = {
        "INR": ["INR", "RUPEE", "RUPEES"],
        "USD": ["USD", "DOLLAR", "DOLLARS"],
        "EUR": ["EUR", "EURO", "EUROS"],
        "GBP": ["GBP", "POUND", "POUNDS"],
        "CHF": ["CHF"],
        "CAD": ["CAD"],
        "AUD": ["AUD"]
    }

    for currency, words in currency_words.items():

        for word in words:

            if word in text_upper:
                return currency

    return ""


# ============================================================
# SUMMARY LABELS
# ============================================================

SUMMARY_LABELS = {
    "total",
    "totalamount",
    "subtotal",
    "subtot",
    "tax",
    "salestax",
    "vat",
    "gst",
    "mwst",
    "balance",
    "balancedue",
    "change",
    "cash",
    "amountdue",
    "grandtotal",
    "tip",
    "servicecharge",
    "discount"
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = text.strip()

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text


# ============================================================
# GROUP OCR RESULTS INTO ROWS
# ============================================================

def group_ocr_rows(results):

    data = []

    for result in results:

        if len(result) != 3:
            continue

        bbox, text, confidence = result

        if not text.strip():
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        x = min(xs)
        y = min(ys)

        data.append({
            "text": normalize_text(text),
            "confidence": confidence,
            "x": x,
            "y": y,
            "bbox": bbox
        })

    if not data:
        return []

    data.sort(
        key=lambda x: x["y"]
    )

    rows = []

    # Dynamic tolerance based on receipt size
    y_tolerance = 18

    for item in data:

        placed = False

        for row in rows:

            avg_y = np.mean(
                [r["y"] for r in row]
            )

            if abs(item["y"] - avg_y) <= y_tolerance:

                row.append(item)

                placed = True

                break

        if not placed:

            rows.append([item])

    # Sort each row left-to-right
    for row in rows:

        row.sort(
            key=lambda x: x["x"]
        )

    return rows


# ============================================================
# PARSE RECEIPT
# ============================================================

def parse_receipt(results):

    rows = group_ocr_rows(results)

    items = []
    summaries = []

    all_text = []

    for row in rows:

        row_text = " ".join(
            item["text"]
            for item in row
        )

        row_text = normalize_text(
            row_text
        )

        all_text.append(row_text)

        normalized = re.sub(
            r'[^a-zA-Z0-9]',
            '',
            row_text
        ).lower()

        # -----------------------------------------
        # SUMMARY ROW
        # -----------------------------------------

        is_summary = any(
            label in normalized
            for label in SUMMARY_LABELS
        )

        if is_summary:

            amount = money_value(
                row_text
            )

            if amount is not None:

                currency = detect_currency(
                    row_text
                )

                label = row_text

                # Remove amount from label
                label = MONEY_PATTERN.sub(
                    "",
                    label
                )

                label = normalize_text(
                    label
                )

                summaries.append({
                    "Label": label,
                    "Amount": amount,
                    "Currency": currency
                })

            continue

        # -----------------------------------------
        # MONEY VALUES
        # -----------------------------------------

        prices = []

        for element in row:

            value = money_value(
                element["text"]
            )

            if value is not None:

                prices.append({
                    "value": value,
                    "x": element["x"],
                    "text": element["text"]
                })

        if not prices:
            continue

        # Usually the right-most monetary value
        price_info = max(
            prices,
            key=lambda x: x["x"]
        )

        price = price_info["value"]

        # -----------------------------------------
        # REMOVE PRICE FROM TEXT
        # -----------------------------------------

        item_text = row_text

        item_text = MONEY_PATTERN.sub(
            "",
            item_text
        )

        item_text = normalize_text(
            item_text
        )

        if not item_text:
            continue

        # -----------------------------------------
        # QUANTITY DETECTION
        # -----------------------------------------

        quantity = 1

        tokens = item_text.split()

        if tokens:

            first = tokens[0]

            # Integer quantity
            if re.fullmatch(
                r'\d{1,3}',
                first
            ):

                quantity = int(first)

                tokens = tokens[1:]

                item_text = " ".join(
                    tokens
                )

        if not item_text:
            continue

        # Avoid obvious non-item lines
        lower_text = item_text.lower()

        if lower_text in {
            "receipt",
            "thank you",
            "thankyou",
            "www",
            "cashier"
        }:
            continue

        currency = detect_currency(
            row_text
        )

        items.append({
            "Quantity": quantity,
            "Item": item_text,
            "Price": price,
            "Currency": currency
        })

    return items, summaries, all_text


# ============================================================
# PROCESS ONE RECEIPT
# ============================================================

def process_receipt(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    # -----------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------

    original_pil = Image.open(
        BytesIO(file_bytes)
    )

    # Fix EXIF orientation
    original_pil = ImageOps.exif_transpose(
        original_pil
    ).convert("RGB")

    original_rgb = np.array(
        original_pil
    )

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )

    # -----------------------------------------
    # PROCESSED COPY
    # -----------------------------------------

    corrected = correct_receipt(
        original_bgr
    )

    corrected = deskew_image(
        corrected
    )

    # -----------------------------------------
    # OCR
    # -----------------------------------------

    results = run_ocr(
        corrected
    )

    # -----------------------------------------
    # PARSE
    # -----------------------------------------

    items, summaries, all_text = parse_receipt(
        results
    )

    return (
        original_pil,
        corrected,
        results,
        items,
        summaries,
        all_text
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload receipt image(s)",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    accept_multiple_files=True
)


# ============================================================
# MAIN APP
# ============================================================

if uploaded_files:

    combined_items = []
    combined_summaries = []

    for file_index, uploaded_file in enumerate(
        uploaded_files,
        start=1
    ):

        st.divider()

        st.header(
            f"🧾 Receipt {file_index}: {uploaded_file.name}"
        )

        try:

            (
                original_image,
                corrected_image,
                results,
                items,
                summaries,
                all_text
            ) = process_receipt(
                uploaded_file
            )

            # ==================================================
            # ORIGINAL RECEIPT
            # ==================================================

            st.subheader(
                "📷 Original Receipt"
            )

            st.image(
                original_image,
                caption="Original image — unchanged",
                use_container_width=True
            )

            # ==================================================
            # PROCESSED COPY
            # ==================================================

            with st.expander(
                "🔧 View processed image used for OCR"
            ):

                processed_rgb = cv2.cvtColor(
                    corrected_image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    processed_rgb,
                    caption="Processed copy used internally for OCR",
                    use_container_width=True
                )

            # ==================================================
            # OCR TEXT
            # ==================================================

            with st.expander(
                "🔍 View raw OCR output"
            ):

                if all_text:

                    for line in all_text:

                        st.write(line)

                else:

                    st.warning(
                        "No readable text was detected."
                    )

            # ==================================================
            # ITEM TABLE
            # ==================================================

            st.subheader(
                "🛒 Extracted Items"
            )

            if items:

                item_df = pd.DataFrame(
                    items
                )

                # Add receipt name
                item_df.insert(
                    0,
                    "Receipt",
                    uploaded_file.name
                )

                edited_df = st.data_editor(
                    item_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"items_{file_index}"
                )

                combined_items.append(
                    edited_df
                )

                # Calculate total of extracted item prices
                if "Price" in edited_df.columns:

                    numeric_prices = pd.to_numeric(
                        edited_df["Price"],
                        errors="coerce"
                    )

                    extracted_sum = numeric_prices.sum()

                    st.info(
                        f"💰 Sum of extracted item prices: "
                        f"{extracted_sum:.2f}"
                    )

            else:

                st.warning(
                    "No item rows were confidently extracted."
                )

            # ==================================================
            # SUMMARY
            # ==================================================

            st.subheader(
                "📌 Receipt Summary"
            )

            if summaries:

                summary_df = pd.DataFrame(
                    summaries
                )

                summary_df.insert(
                    0,
                    "Receipt",
                    uploaded_file.name
                )

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )

                combined_summaries.append(
                    summary_df
                )

            else:

                st.info(
                    "No summary fields such as Total, Tax, Cash or Change were detected."
                )

        except Exception as e:

            st.error(
                f"❌ Error processing {uploaded_file.name}: {e}"
            )


    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    st.divider()

    st.header(
        "📊 Combined Results"
    )

    # --------------------------------------------------------
    # Combined items
    # --------------------------------------------------------

    if combined_items:

        final_items = pd.concat(
            combined_items,
            ignore_index=True
        )

        st.subheader(
            "🛒 All Extracted Items"
        )

        st.dataframe(
            final_items,
            use_container_width=True,
            hide_index=True
        )

        csv_items = final_items.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Items CSV",
            data=csv_items,
            file_name="receipt_items.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------
    # Combined summaries
    # --------------------------------------------------------

    if combined_summaries:

        final_summary = pd.concat(
            combined_summaries,
            ignore_index=True
        )

        st.subheader(
            "📌 All Summary Fields"
        )

        st.dataframe(
            final_summary,
            use_container_width=True,
            hide_index=True
        )

        csv_summary = final_summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Summary CSV",
            data=csv_summary,
            file_name="receipt_summary.csv",
            mime="text/csv"
        )


else:

    st.info(
        "👆 Upload one or more receipt images to start."
    )

    st.markdown(
        """
        ### Supported formats
        - PNG
        - JPG
        - JPEG

        ### What the system does
        1. 📷 Keeps your original receipt unchanged
        2. 🔄 Creates a corrected copy for OCR
        3. ✨ Improves moderate blur/noise/shadows
        4. 🔍 Extracts text using EasyOCR
        5. 🧾 Identifies items, quantities and prices
        6. 📌 Identifies summary values such as Total, Tax, Cash and Change
        7. 💱 Keeps the **original currency**
        8. 📊 Displays editable structured data
        9. 📥 Allows CSV export

        **No currency conversion is performed.**
        """
    )
