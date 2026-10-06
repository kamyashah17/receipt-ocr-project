import streamlit as st
import cv2
import numpy as np
import pandas as pd
import easyocr
import re
import gc

from PIL import Image, ImageOps
from io import BytesIO


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Automatic Receipt OCR",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Automatic Receipt OCR")

st.write(
    "Upload one or more receipt images. "
    "The original receipt is preserved, while a processed copy "
    "is used internally for OCR."
)


# ============================================================
# EASY OCR MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading OCR model...")
def load_reader():

    return easyocr.Reader(
        ['en'],
        gpu=False
    )


reader = load_reader()


# ============================================================
# ORDER RECEIPT CORNERS
# ============================================================

def order_points(points):

    points = np.asarray(
        points,
        dtype=np.float32
    ).reshape(4, 2)

    # Sort according to Y coordinate
    y_sorted = points[
        np.argsort(points[:, 1])
    ]

    top = y_sorted[:2]
    bottom = y_sorted[2:]

    # Sort top points according to X
    top = top[
        np.argsort(top[:, 0])
    ]

    # Sort bottom points according to X
    bottom = bottom[
        np.argsort(bottom[:, 0])
    ]

    top_left = top[0]
    top_right = top[1]

    bottom_left = bottom[0]
    bottom_right = bottom[1]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ],
        dtype=np.float32
    )


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def correct_receipt(image):

    original = image.copy()

    try:

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

            area = cv2.contourArea(
                contour
            )

            # Ignore very small objects
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

            points = np.asarray(
                approx,
                dtype=np.float32
            ).reshape(4, 2)

            rect = order_points(
                points
            )

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

            if (
                max_width < 100
                or max_height < 100
            ):
                continue

            destination = np.array(
                [
                    [0, 0],
                    [max_width - 1, 0],
                    [
                        max_width - 1,
                        max_height - 1
                    ],
                    [0, max_height - 1]
                ],
                dtype=np.float32
            )

            matrix = cv2.getPerspectiveTransform(
                rect,
                destination
            )

            warped = cv2.warpPerspective(
                image,
                matrix,
                (
                    max_width,
                    max_height
                )
            )

            return warped

    except Exception:

        pass

    # If correction fails, use original
    return original


# ============================================================
# DESKEW IMAGE
# ============================================================

def deskew_image(image):

    try:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
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

        lines = np.asarray(
            lines
        ).reshape(-1, 4)

        angles = []

        for x1, y1, x2, y2 in lines:

            angle = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            # Only use almost-horizontal lines
            if abs(angle) < 20:

                angles.append(
                    angle
                )

        if not angles:
            return image

        angle = float(
            np.median(angles)
        )

        # Already straight
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

    except Exception:

        return image


# ============================================================
# PREPROCESS IMAGE FOR OCR
# ============================================================

def preprocess_for_ocr(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # Remove moderate noise
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        10,
        7,
        21
    )

    # Mild sharpening
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
    )

    sharpened = cv2.filter2D(
        denoised,
        -1,
        kernel
    )

    return sharpened


# ============================================================
# OCR
# ============================================================

def run_ocr(image):

    # --------------------------------------------------------
    # First OCR pass: enhanced image
    # --------------------------------------------------------

    processed = preprocess_for_ocr(
        image
    )

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False
    )

    # --------------------------------------------------------
    # If OCR found very little text, try threshold image
    # --------------------------------------------------------

    if len(results) < 3:

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

        second_results = reader.readtext(
            adaptive,
            detail=1,
            paragraph=False
        )

        results.extend(
            second_results
        )

        del gray
        del adaptive

    del processed

    return results


# ============================================================
# MONEY REGEX
# ============================================================

MONEY_PATTERN = re.compile(
    r'(?<!\d)(\d{1,7}(?:[.,]\d{2}))(?!\d)'
)


def money_value(text):

    matches = MONEY_PATTERN.findall(
        text
    )

    if not matches:
        return None

    value = matches[-1]

    try:

        return float(
            value.replace(
                ",",
                "."
            )
        )

    except Exception:

        return None


# ============================================================
# CURRENCY DETECTION
# ============================================================

def detect_currency(text):

    upper_text = text.upper()

    # Currency symbols
    currency_symbols = {

        "₹": "INR",

        "$": "USD",

        "€": "EUR",

        "£": "GBP",

        "CHF": "CHF",

        "¥": "JPY/CNY"
    }

    for symbol, currency in currency_symbols.items():

        if symbol in text:

            return currency

    # Currency words / codes
    currency_words = {

        "INR": [
            "INR",
            "RUPEE",
            "RUPEES"
        ],

        "USD": [
            "USD",
            "DOLLAR",
            "DOLLARS"
        ],

        "EUR": [
            "EUR",
            "EURO",
            "EUROS"
        ],

        "GBP": [
            "GBP",
            "POUND",
            "POUNDS"
        ],

        "CHF": [
            "CHF"
        ],

        "CAD": [
            "CAD"
        ],

        "AUD": [
            "AUD"
        ]
    }

    for currency, words in currency_words.items():

        for word in words:

            if word in upper_text:

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

        try:

            xs = [
                float(point[0])
                for point in bbox
            ]

            ys = [
                float(point[1])
                for point in bbox
            ]

            x = min(xs)
            y = min(ys)

        except Exception:

            continue

        data.append(
            {
                "text": normalize_text(text),
                "confidence": confidence,
                "x": x,
                "y": y,
                "bbox": bbox
            }
        )

    if not data:
        return []

    # Sort top-to-bottom
    data.sort(
        key=lambda item: item["y"]
    )

    rows = []

    for item in data:

        placed = False

        for row in rows:

            average_y = np.mean(
                [
                    r["y"]
                    for r in row
                ]
            )

            if abs(
                item["y"] - average_y
            ) <= 18:

                row.append(item)

                placed = True

                break

        if not placed:

            rows.append(
                [item]
            )

    # Sort each row left-to-right
    for row in rows:

        row.sort(
            key=lambda item: item["x"]
        )

    return rows


# ============================================================
# PARSE RECEIPT
# ============================================================

def parse_receipt(results):

    rows = group_ocr_rows(
        results
    )

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

        if not row_text:
            continue

        all_text.append(
            row_text
        )

        # ----------------------------------------------------
        # Normalize text for keyword matching
        # ----------------------------------------------------

        normalized = re.sub(
            r'[^a-zA-Z0-9]',
            '',
            row_text
        ).lower()

        # ----------------------------------------------------
        # Detect summary rows
        # ----------------------------------------------------

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

                label = MONEY_PATTERN.sub(
                    "",
                    row_text
                )

                label = normalize_text(
                    label
                )

                summaries.append(
                    {
                        "Label": label,
                        "Amount": amount,
                        "Currency": currency
                    }
                )

            continue

        # ----------------------------------------------------
        # Find monetary values
        # ----------------------------------------------------

        prices = []

        for element in row:

            value = money_value(
                element["text"]
            )

            if value is not None:

                prices.append(
                    {
                        "value": value,
                        "x": element["x"]
                    }
                )

        if not prices:
            continue

        # Use right-most money value
        price_info = max(
            prices,
            key=lambda item: item["x"]
        )

        price = price_info["value"]

        # ----------------------------------------------------
        # Remove price from item text
        # ----------------------------------------------------

        item_text = MONEY_PATTERN.sub(
            "",
            row_text
        )

        item_text = normalize_text(
            item_text
        )

        if not item_text:
            continue

        # ----------------------------------------------------
        # Quantity
        # ----------------------------------------------------

        quantity = 1

        tokens = item_text.split()

        if tokens:

            first_token = tokens[0]

            if re.fullmatch(
                r'\d{1,3}',
                first_token
            ):

                quantity = int(
                    first_token
                )

                tokens = tokens[1:]

                item_text = " ".join(
                    tokens
                )

        item_text = normalize_text(
            item_text
        )

        if not item_text:
            continue

        # ----------------------------------------------------
        # Ignore obvious non-item text
        # ----------------------------------------------------

        lower_text = item_text.lower()

        ignored = {

            "receipt",

            "thank you",

            "thankyou",

            "cashier",

            "invoice",

            "www",

            "total",

            "subtotal",

            "tax",

            "gst",

            "vat"
        }

        if lower_text in ignored:
            continue

        # ----------------------------------------------------
        # Currency
        # ----------------------------------------------------

        currency = detect_currency(
            row_text
        )

        items.append(
            {
                "Quantity": quantity,
                "Item": item_text,
                "Price": price,
                "Currency": currency
            }
        )

    return (
        items,
        summaries,
        all_text
    )


# ============================================================
# PROCESS ONE RECEIPT
# ============================================================

def process_receipt(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original_pil = Image.open(
        BytesIO(file_bytes)
    )

    # Correct EXIF orientation only
    original_pil = ImageOps.exif_transpose(
        original_pil
    ).convert("RGB")

    # This is kept as the original image
    original_rgb = np.array(
        original_pil
    )

    # ========================================================
    # OCR COPY
    # ========================================================

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Resize ONLY the OCR copy
    # --------------------------------------------------------

    max_dimension = 1800

    height, width = original_bgr.shape[:2]

    if max(
        height,
        width
    ) > max_dimension:

        scale = (
            max_dimension
            / max(height, width)
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        original_bgr = cv2.resize(
            original_bgr,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

    # ========================================================
    # PERSPECTIVE CORRECTION
    # ========================================================

    corrected = correct_receipt(
        original_bgr
    )

    # ========================================================
    # DESKEW
    # ========================================================

    corrected = deskew_image(
        corrected
    )

    # ========================================================
    # OCR
    # ========================================================

    results = run_ocr(
        corrected
    )

    # ========================================================
    # PARSE
    # ========================================================

    items, summaries, all_text = parse_receipt(
        results
    )

    # ========================================================
    # CLEAN TEMPORARY MEMORY
    # ========================================================

    del file_bytes
    del original_rgb
    del original_bgr
    del corrected
    del results

    gc.collect()

    return (
        original_pil,
        items,
        summaries,
        all_text
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "📤 Upload receipt image(s)",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    accept_multiple_files=True
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_files:

    combined_items = []

    combined_summaries = []

    # ========================================================
    # PROCESS EACH RECEIPT
    # ========================================================

    for receipt_number, uploaded_file in enumerate(
        uploaded_files,
        start=1
    ):

        st.divider()

        st.header(
            f"🧾 Receipt {receipt_number}: "
            f"{uploaded_file.name}"
        )

        try:

            (
                original_image,
                items,
                summaries,
                all_text
            ) = process_receipt(
                uploaded_file
            )

            # =================================================
            # ORIGINAL RECEIPT
            # =================================================

            st.subheader(
                "📷 Original Receipt"
            )

            st.image(
                original_image,
                caption="Original receipt — unchanged",
                use_container_width=True
            )

            # =================================================
            # RAW OCR TEXT
            # =================================================

            with st.expander(
                "🔍 View OCR Text"
            ):

                if all_text:

                    for line in all_text:

                        st.write(line)

                else:

                    st.warning(
                        "No readable text was detected."
                    )

            # =================================================
            # ITEMS
            # =================================================

            st.subheader(
                "🛒 Extracted Items"
            )

            if items:

                item_df = pd.DataFrame(
                    items
                )

                item_df.insert(
                    0,
                    "Receipt",
                    uploaded_file.name
                )

                edited_df = st.data_editor(
                    item_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"items_{receipt_number}"
                )

                combined_items.append(
                    edited_df
                )

                # ------------------------------------------------
                # Extracted item price total
                # ------------------------------------------------

                if "Price" in edited_df.columns:

                    numeric_prices = pd.to_numeric(
                        edited_df["Price"],
                        errors="coerce"
                    )

                    extracted_total = (
                        numeric_prices
                        .sum()
                    )

                    st.info(
                        "💰 Sum of extracted item prices: "
                        f"{extracted_total:.2f}"
                    )

            else:

                st.warning(
                    "⚠️ No item rows were extracted from this receipt."
                )

            # =================================================
            # SUMMARY
            # =================================================

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
                    "No Total / Tax / Cash / Change fields detected."
                )

            # =================================================
            # CLEAN MEMORY AFTER EACH RECEIPT
            # =================================================

            del original_image
            gc.collect()

        except Exception as error:

            st.error(
                f"❌ Error processing "
                f"{uploaded_file.name}: {error}"
            )

            # Continue to next receipt
            gc.collect()

    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    st.divider()

    st.header(
        "📊 Combined Results"
    )

    # ========================================================
    # ALL ITEMS
    # ========================================================

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

        items_csv = final_items.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Items CSV",
            data=items_csv,
            file_name="receipt_items.csv",
            mime="text/csv"
        )

    # ========================================================
    # ALL SUMMARY VALUES
    # ========================================================

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

        summary_csv = final_summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Summary CSV",
            data=summary_csv,
            file_name="receipt_summary.csv",
            mime="text/csv"
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    if combined_items or combined_summaries:

        st.success(
            "✅ All uploaded receipts have been processed."
        )

else:

    # ========================================================
    # INITIAL SCREEN
    # ========================================================

    st.info(
        "👆 Upload one or more receipt images to begin."
    )

    st.markdown(
        """
        ### 🧾 Supported formats

        - PNG
        - JPG
        - JPEG

        ### 🚀 Features

        - Multiple receipt uploads
        - Perspective correction
        - Deskewing
        - Blur/noise improvement
        - EasyOCR text recognition
        - Item extraction
        - Quantity extraction
        - Price extraction
        - Total extraction
        - Tax / GST extraction
        - Cash / Change extraction
        - Editable results
        - CSV export
        - Original currency preserved

        ### 💱 Currency

        **The receipt is scanned as it is.**

        There is **NO INR conversion**.

        For example:

        `CHF 54.50` → `54.50 CHF`

        `$131.08` → `131.08 USD`

        `€45.20` → `45.20 EUR`

        `₹500.00` → `500.00 INR`

        The application never changes the monetary value
        into another currency.
        """
    )
