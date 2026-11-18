# Snap-to-List — Smart Grocery Assistant

Snap-to-List is a desktop app that turns your webcam into a shopping-list scanner. Hold a product up to the camera: if it has been registered before, it is recognised and added to your shopping list automatically (quantity +1). If it is new, you register it once — **by blinking** — and it is remembered from then on.

The interface is themed for health & beauty shopping (skincare, hair care, vitamins, etc.), but the categories can be edited to suit any kind of product.

## Features

- **Webcam product recognition** — ORB feature matching with a RANSAC geometric check and a colour-histogram score.
- **Blink to register** — facial landmarks (MediaPipe Face Mesh + Eye Aspect Ratio) detect a deliberate blink and capture the product photo.
- **Item tracking zone** — background subtraction and motion detection decide when a product is present and held steady.
- **Image clean-up** — perspective flattening of boxes/labels, CLAHE brightening and sharpening for stored product photos.
- **Learning over time** — confirming an uncertain match adds the new photo as another reference for that product.
- **Shopping list** — quantity controls, bought/to-buy filter, search, aisle-ordered categories, estimated total price, and export to a PNG image.
- **Product manager** — view and delete registered products.
- **Manual entry** — products can be saved without a scan image if you prefer.
- **Local storage** — everything is stored in a SQLite database and image folder on your machine; nothing is uploaded.
- **Optional OCR** — with `pytesseract`, the product name is pre-filled from the label text.

## Requirements

- Python 3.9 – 3.11 (with Tkinter, which is included in the standard python.org installers)
- A webcam
- Windows is the primary target (the camera code tries `CAP_DSHOW` first), but it falls back to the default backend on macOS/Linux.
- *Optional:* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system if you want name suggestions.

## Setup

1. **Save the script** as `Snap_To_List.py` in a project folder, alongside `requirements.txt`.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Enable OCR name suggestions:**
   - Install the Tesseract engine for your OS (on Windows, make sure it is on your `PATH`).
   - Uncomment `pytesseract` in `requirements.txt` and run `pip install pytesseract`, or run it directly.

## Running the app

```bash
python Snap_To_List.py
```

On first launch the app creates a `grocery_data/` folder next to the script containing the database (`products.db`) and product images (`images/`).

## How to use

### 1. Scan a product
1. Open the **📷 Scan** tab.
2. Hold the product inside the box on the **right side** of the camera preview, label facing the camera.
3. Hold it steady until the progress bar fills.
4. If it is recognised, it is added to your list automatically. Remove the item from the box before scanning the next one.

### 2. Register a new product
1. Hold the unknown product steady in the box.
2. **Blink** to capture its photo (you can see your live eye-aspect-ratio, `EAR`, in the top-left of the preview).
3. Fill in the name, brand, size, price and category.
4. Leave **"Save this scanned image for future recognition"** checked so it can be auto-recognised next time (uncheck it to save the product details only).
5. Click **Save product & add to list**.

### 3. Uncertain matches
If the app is only fairly sure, it asks *"Is this …?"*:
- **Yes** — adds it to the list and saves the photo as an extra reference to improve future matching.
- **No** — takes you to the registration form.

### 4. Manage your list
Use the **📋 Shopping list** tab to change quantities, tick items as bought, search, filter, clear bought items, or export the to-buy list as a PNG. Use the **📦 Products** tab to delete incorrect registrations.

### Keyboard shortcut
- `Esc` — cancel the current confirm/register step.

## Configuration

Tunable constants are at the top of `Snap_To_List.py`:

| Setting | Purpose |
|---|---|
| `CURRENCY` | Currency symbol shown for prices |
| `CATEGORIES` / `CAT_ICON` | Product categories (also the aisle order of the list) |
| `EAR_THRESHOLD`, `BLINK_MIN`, `BLINK_MAX` | Blink detection sensitivity |
| `STEADY_SECONDS`, `MOTION_LIMIT`, `PRESENT_RATIO` | How steady / large an item must be to be scanned |
| `AUTO_INL`, `AUTO_CONF` | Thresholds for automatic add |
| `MAYBE_INL`, `MAYBE_CONF` | Thresholds for asking the user to confirm |
| `MIN_FEATS` | Minimum image features required to use a photo |

## Project structure

```
.
├── Snap_To_List.py        # the application
├── requirements.txt
├── README.md
└── grocery_data/          # created on first run
    ├── products.db        # SQLite: products, reference images, shopping list
    └── images/            # saved product photos
```

## Troubleshooting

- **"Camera not available"** — close other apps using the webcam, check the cable, then press **🎥 Retry camera**.
- **Product not recognised** — move closer, improve lighting, and keep the front label facing the camera. Registering a second photo of the same product (or confirming a "maybe" match) improves accuracy.
- **Blink not detected** — make sure your face is visible to the camera and blink deliberately; adjust `EAR_THRESHOLD` if needed.
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`** — install a compatible MediaPipe version: `pip install "mediapipe>=0.10.9,<0.10.21"`.
- **NumPy / OpenCV import errors** — use `numpy<2` with this MediaPipe range.
- **OCR does nothing** — confirm the Tesseract engine is installed and on your `PATH`.
- **Linux: Tkinter missing** — install it with your package manager (e.g. `sudo apt install python3-tk`).

## Notes

- Recognition works best on products with distinctive printed labels; plain or very shiny packaging provides fewer features.
- All processing happens locally on your computer.
- If a v1 `grocery_data/list.json` exists, it is imported automatically on first run.
