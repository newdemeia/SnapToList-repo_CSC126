"""
Snap-to-List -  Smart Grocery Assistant
===========================================
Show a product to the webcam. If it was registered before, it is recognised and added
to the shopping list automatically (quantity +1). Unknown products are registered once.

Topics:  Facial Landmarks (blink = register photo)  |  Motion/Object Tracking (item zone)
         Image Analysis/Transformation (crop, perspective, CLAHE, ORB features)

Install:  pip install opencv-python mediapipe customtkinter pillow numpy
Optional: pip install pytesseract   (auto-suggest names from label text)
Run:      python Snap_To_List.py
"""
import json
import queue
import sqlite3
import threading
import time
import tkinter as tk
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

try:
    import pytesseract
    HAS_OCR = True
except Exception:
    HAS_OCR = False

# =============================================================== settings
DATA_DIR = Path("grocery_data")
IMG_DIR = DATA_DIR / "images"
DB_FILE = DATA_DIR / "products.db"
CAM_SIZE = (640, 480)

EAR_THRESHOLD, BLINK_MIN, BLINK_MAX = 0.21, 2, 12      # blink detection
PRESENT_RATIO, MOTION_LIMIT, STEADY_SECONDS = 0.06, 2.5, 0.8
REMOVE_SECONDS = 1.0         # item must be gone this long before the next scan
RECOGNISE_EVERY = 0.4        # seconds between recognition attempts

MIN_FEATS = 15               # allow less detailed products while requiring usable descriptors
AUTO_INL, AUTO_CONF = 8, 0.30       # more tolerant matching; auto-add after 2 matching votes
MAYBE_INL, MAYBE_CONF = 5, 0.20     # weak matches are treated as candidates, not ignored
                                    # below: unknown product -> offer to register

CATEGORIES = ["Skincare", "Hair Care", "Makeup & Beauty", "Personal Care", "Oral Care",
              "Vitamins & Supplements", "Health & Medical", "Baby & Mum", "Feminine Care",
              "Men's Grooming", "Fragrance", "Snacks & Household", "Other"]   # also the aisle order of the list
CAT_ICON = {"Skincare": "🧴", "Hair Care": "💇", "Makeup & Beauty": "💄", "Personal Care": "🧼",
            "Oral Care": "🪥", "Vitamins & Supplements": "💊", "Health & Medical": "🩺", "Baby & Mum": "🍼",
            "Feminine Care": "🌸", "Men's Grooming": "🪒", "Fragrance": "✨", "Snacks & Household": "🛒",
            "Other": "📦"}
CURRENCY = "$"          # change to your currency symbol
GREEN, GREEN_DK, CARD, MUTED, GREY = "#2ecc71", "#27ae60", "#232a36", "#8b95a7", "#3a4254"
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =============================================================== image helpers
def eye_aspect_ratio(pts, idx):
    p = [np.array(pts[i]) for i in idx]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)


def flatten_label(img):
    """Perspective transform: flatten a large 4-corner shape (box / label) if one is found."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cnts, _ = cv2.findContours(cv2.dilate(cv2.Canny(gray, 50, 150), None),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4 or cv2.contourArea(approx) < 0.5 * img.shape[0] * img.shape[1]:
        return img
    pts = approx.reshape(4, 2).astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    rect = np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")
    w = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[3] - rect[2])))
    h = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
    if w < 60 or h < 60:
        return img
    M = cv2.getPerspectiveTransform(rect, np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32"))
    return cv2.warpPerspective(img, M, (w, h))


def enhance(img):
    """Display image: flatten, CLAHE brighten, sharpen, resize (NOT used for matching)."""
    img = flatten_label(img)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    img = cv2.cvtColor(cv2.merge((cv2.createCLAHE(2.0, (8, 8)).apply(l), a, b)), cv2.COLOR_LAB2BGR)
    img = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0)
    h, w = img.shape[:2]
    s = 420 / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# =============================================================== database (SQLite)
class Database:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS products(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        category TEXT NOT NULL,
        date_registered TEXT NOT NULL,
        brand TEXT NOT NULL DEFAULT '',
        size TEXT NOT NULL DEFAULT '',
        price REAL NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS product_images(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        image_path TEXT NOT NULL,
        keypoints BLOB NOT NULL, descriptors BLOB NOT NULL, hist BLOB NOT NULL);
    CREATE TABLE IF NOT EXISTS shopping_list(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL UNIQUE REFERENCES products(id) ON DELETE CASCADE,
        quantity INTEGER NOT NULL DEFAULT 1,
        date_added TEXT NOT NULL,
        bought INTEGER NOT NULL DEFAULT 0);
    """

    def __init__(self):
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(DB_FILE)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(self.SCHEMA)
        cols = [r["name"] for r in self.con.execute("PRAGMA table_info(products)")]
        for col, ddl in (("brand", "TEXT NOT NULL DEFAULT ''"), ("size", "TEXT NOT NULL DEFAULT ''"),
                         ("price", "REAL NOT NULL DEFAULT 0")):
            if col not in cols:                       # upgrade an older database in place
                self.con.execute(f"ALTER TABLE products ADD COLUMN {col} {ddl}")
        self.con.commit()

    @staticmethod
    def now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- products
    def register(self, name, category, display_img, feats, brand="", size="", price=0.0):
        """Create the product (or reuse it if the name exists) and attach a reference image."""
        row = self.con.execute("SELECT id FROM products WHERE name=?", (name,)).fetchone()
        is_new = row is None
        if is_new:
            cur = self.con.execute("INSERT INTO products(name,category,date_registered,brand,size,price) "
                                   "VALUES(?,?,?,?,?,?)", (name, category, self.now(), brand, size, price))
            pid = cur.lastrowid
        else:
            pid = row["id"]
            if price:
                self.con.execute("UPDATE products SET price=? WHERE id=?", (price, pid))
        ref = self.add_reference(pid, display_img, feats)
        return pid, is_new, ref

    def add_manual_product(self, name, category, brand="", size="", price=0.0):
        """Save a product entered manually; no camera photo is required."""
        row = self.con.execute("SELECT id FROM products WHERE name=?", (name,)).fetchone()
        if row:
            pid = row["id"]
            self.con.execute(
                "UPDATE products SET category=?, brand=?, size=?, price=? WHERE id=?",
                (category, brand, size, price, pid))
            is_new = False
        else:
            cur = self.con.execute(
                "INSERT INTO products(name,category,date_registered,brand,size,price) VALUES(?,?,?,?,?,?)",
                (name, category, self.now(), brand, size, price))
            pid = cur.lastrowid
            is_new = True
        self.con.commit()
        return pid, is_new

    def add_reference(self, pid, display_img, feats):
        pts, des, hist = feats
        path = IMG_DIR / f"{pid}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(path), display_img)
        self.con.execute("INSERT INTO product_images(product_id,image_path,keypoints,descriptors,hist) "
                         "VALUES(?,?,?,?,?)", (pid, str(path), pts.tobytes(), des.tobytes(), hist.tobytes()))
        self.con.commit()
        return (pid, pts, des, hist)

    def load_refs(self):
        refs = []
        for r in self.con.execute("SELECT product_id,keypoints,descriptors,hist FROM product_images"):
            refs.append((r["product_id"], np.frombuffer(r["keypoints"], np.float32).reshape(-1, 2),
                         np.frombuffer(r["descriptors"], np.uint8).reshape(-1, 32),
                         np.frombuffer(r["hist"], np.float32)))
        return refs

    def get_product(self, pid):
        return self.con.execute(
            "SELECT p.*, (SELECT image_path FROM product_images WHERE product_id=p.id ORDER BY id LIMIT 1) AS image "
            "FROM products p WHERE p.id=?", (pid,)).fetchone()

    def list_products(self):
        return self.con.execute(
            "SELECT p.*, (SELECT image_path FROM product_images WHERE product_id=p.id ORDER BY id LIMIT 1) AS image,"
            " (SELECT COUNT(*) FROM product_images WHERE product_id=p.id) AS refs "
            "FROM products p ORDER BY p.name").fetchall()

    def delete_product(self, pid):
        for r in self.con.execute("SELECT image_path FROM product_images WHERE product_id=?", (pid,)):
            Path(r["image_path"]).unlink(missing_ok=True)
        self.con.execute("DELETE FROM products WHERE id=?", (pid,))
        self.con.commit()

    # ---- shopping list
    def add_to_list(self, pid, qty=1, bought=False):
        row = self.con.execute("SELECT quantity,bought FROM shopping_list WHERE product_id=?", (pid,)).fetchone()
        if row is None:
            self.con.execute("INSERT INTO shopping_list(product_id,quantity,date_added,bought) VALUES(?,?,?,?)",
                             (pid, qty, self.now(), int(bought)))
            new_qty = qty
        elif row["bought"]:                      # bought before -> start a fresh entry
            new_qty = qty
            self.con.execute("UPDATE shopping_list SET quantity=?,bought=0,date_added=? WHERE product_id=?",
                             (new_qty, self.now(), pid))
        else:
            new_qty = row["quantity"] + qty
            self.con.execute("UPDATE shopping_list SET quantity=? WHERE product_id=?", (new_qty, pid))
        self.con.commit()
        return new_qty

    def list_items(self):
        return self.con.execute(
            "SELECT l.id AS item_id, p.id AS pid, p.name, p.brand, p.size, p.price, p.category, l.quantity, l.date_added, l.bought,"
            " (SELECT image_path FROM product_images WHERE product_id=p.id ORDER BY id LIMIT 1) AS image "
            "FROM shopping_list l JOIN products p ON p.id=l.product_id "
            "ORDER BY l.bought, l.date_added DESC").fetchall()

    def update_item(self, item_id, **fields):
        for k, v in fields.items():
            if k in ("quantity", "bought"):
                self.con.execute(f"UPDATE shopping_list SET {k}=? WHERE id=?", (v, item_id))
        self.con.commit()

    def remove_item(self, item_id):
        self.con.execute("DELETE FROM shopping_list WHERE id=?", (item_id,))
        self.con.commit()

    def clear_bought(self):
        self.con.execute("DELETE FROM shopping_list WHERE bought=1")
        self.con.commit()


# =============================================================== recognition (ORB)
class Recognizer:
    """ORB keypoints + ratio test + RANSAC homography (geometric check) + colour histogram."""

    def __init__(self):
        self.refs = []          # list of (product_id, points, descriptors, hist); replaced, never mutated

    @staticmethod
    def features(img):
        h, w = img.shape[:2]
        s = 400 / max(h, w)
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)))
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        kps, des = cv2.ORB_create(nfeatures=1000).detectAndCompute(gray, None)
        if des is None:
            return None
        hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return np.float32([k.pt for k in kps]), des, hist.flatten().astype(np.float32)

    @staticmethod
    def _inliers(bf, qp, qd, rp, rd):
        if len(rd) < 8:
            return 0
        good = [p[0] for p in bf.knnMatch(qd, rd, k=2) if len(p) == 2 and p[0].distance < 0.80 * p[1].distance]
        if len(good) < 8:
            return 0
        src = rp[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        dst = qp[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0

    def identify(self, img):
        out = {"pid": None, "conf": 0.0, "inliers": 0}
        refs, q = self.refs, self.features(img)
        if not refs or q is None or len(q[1]) < MIN_FEATS:
            return out
        qp, qd, qh = q
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        for pid, rp, rd, rh in refs:
            inl = self._inliers(bf, qp, qd, rp, rd)
            if inl == 0:
                continue
            corr = cv2.compareHist(qh.reshape(-1, 1), rh.reshape(-1, 1), cv2.HISTCMP_CORREL)
            conf = min(1.0, inl / 40) * 0.85 + 0.15 * max(corr, 0.0)
            if conf > out["conf"]:
                out = {"pid": pid, "conf": float(conf), "inliers": inl}
        return out


class RecognitionWorker(threading.Thread):
    """Runs recognition in the background so the camera preview never freezes."""

    def __init__(self, rec):
        super().__init__(daemon=True)
        self.rec, self.inq, self.outq = rec, queue.Queue(maxsize=1), queue.Queue()
        self.start()

    def submit(self, img):
        try:
            self.inq.put_nowait(img)
            return True
        except queue.Full:
            return False

    def run(self):
        while True:
            img = self.inq.get()
            if img is None:
                return
            try:
                self.outq.put(self.rec.identify(img))
            except Exception as e:                      # never kill the worker
                self.outq.put({"pid": None, "conf": 0.0, "inliers": 0, "error": str(e)})


# =============================================================== camera thread
class Camera(threading.Thread):
    """Continuously reads the webcam; everyone else just asks for the latest frame."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock, self.frame, self.cap = threading.Lock(), None, None
        self.running, self.restart_flag = True, False
        self.start()

    @staticmethod
    def _open():
        for idx in (0, 1):
            for api in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = cv2.VideoCapture(idx, api)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    return cap
                cap.release()
        return None

    def run(self):
        fails = 0
        while self.running:
            if self.restart_flag and self.cap:
                self.cap.release()
                self.cap, self.restart_flag = None, False
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    with self.lock:
                        self.frame = None
                    time.sleep(1)
                    continue
            ok, f = self.cap.read()
            if ok:
                fails = 0
                with self.lock:
                    self.frame = f
            else:
                fails += 1
                if fails > 30:                           # camera unplugged / lost
                    self.cap.release()
                    self.cap = None
                    with self.lock:
                        self.frame = None
                time.sleep(0.03)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# =============================================================== application
def db_guard(fn):
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except sqlite3.Error as e:
            self.set_status(f"⚠ Database error: {e}")
    return wrapper


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.title("Snap-to-List  •  Watsons Shopping Assistant")
        self.geometry("1180x760")
        self.minsize(1050, 700)

        self.db = Database()
        self.rec = Recognizer()
        self.rec.refs = self.db.load_refs()
        self.worker = RecognitionWorker(self.rec)
        self.cam = Camera()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.blank = ctk.CTkImage(Image.new("RGB", (2, 2), "#1a1f29"), size=(2, 2))
        self.cache, self.refs_keep = {}, []
        # tracking state
        self.bg = self.prev = self.steady_since = self.absent_since = None
        self.closed_frames, self.blink_event = 0, False
        # recognition state
        self.app_state = "SCAN"        # SCAN | CONFIRM | REGISTER | WAIT
        self.votes = deque(maxlen=3)
        self.rec_busy, self.last_submit = False, 0.0
        self.last_crop = self.pending_raw = self.pending_img = None
        self.candidate = None
        self.filter_mode = "All"

        self.build_ui()
        self.migrate_old_data()
        self.refresh_all()
        self.bind("<Escape>", lambda e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.loop()

    # ----------------------------------------------------------- UI
    def build_ui(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(16, 4))
        ctk.CTkLabel(top, text="💊  Snap-to-List  ·  Watsons", font=("Segoe UI", 26, "bold")).pack(side="left")
        self.count_lbl = ctk.CTkLabel(top, text="", text_color=MUTED, font=("Segoe UI", 14))
        self.count_lbl.pack(side="right")
        self.tabs = ctk.CTkTabview(self, fg_color="transparent", segmented_button_selected_color=GREEN_DK,
                                   segmented_button_selected_hover_color=GREEN_DK)
        self.tabs.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self.build_scan(self.tabs.add("📷  Scan"))
        self.build_list(self.tabs.add("📋  Shopping list"))
        self.build_products(self.tabs.add("📦  Products"))

    def build_scan(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(tab, fg_color=CARD, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=4)
        self.cam_lbl = ctk.CTkLabel(left, text="Starting camera…", width=CAM_SIZE[0], height=CAM_SIZE[1])
        self.cam_lbl.pack(padx=14, pady=14, expand=True)
        ctk.CTkLabel(left, text="Hold the product inside the box on the right side of the picture.",
                     text_color=MUTED).pack(pady=(0, 12))

        right = ctk.CTkScrollableFrame(tab, fg_color=CARD, corner_radius=16, width=340)
        right.grid(row=0, column=1, sticky="ns", pady=4)
        self.steps = []
        for t in ["1. Show the product", "2. Hold it steady", "3. Recognised"]:
            lbl = ctk.CTkLabel(right, text=t, anchor="w", text_color=MUTED, font=("Segoe UI", 14))
            lbl.pack(fill="x", padx=10, pady=1)
            self.steps.append((lbl, t))
        self.progress = ctk.CTkProgressBar(right, progress_color=GREEN)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=10, pady=(8, 4))
        self.status = ctk.CTkLabel(right, text="Starting…", font=("Segoe UI", 15, "bold"),
                                   wraplength=300, justify="left", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(2, 8))

        # result card
        card = ctk.CTkFrame(right, fg_color="#1a1f29", corner_radius=12)
        card.pack(fill="x", padx=10, pady=4)
        self.res_img = ctk.CTkLabel(card, text="Product photo", text_color=MUTED, image=self.blank,
                                    compound="center", width=300, height=150)
        self.res_img.pack(padx=8, pady=(8, 2))
        self.res_name = ctk.CTkLabel(card, text="—", font=("Segoe UI", 17, "bold"), wraplength=290)
        self.res_name.pack()
        self.conf_bar = ctk.CTkProgressBar(card, progress_color=GREEN)
        self.conf_bar.set(0)
        self.conf_bar.pack(fill="x", padx=16, pady=(6, 2))
        self.conf_lbl = ctk.CTkLabel(card, text="No match yet", text_color=MUTED)
        self.conf_lbl.pack(pady=(0, 8))

        # dynamic panels (one visible at a time)
        self.dyn = ctk.CTkFrame(right, fg_color="transparent", height=0)
        self.dyn.pack(fill="x", padx=10, pady=4)
        self.confirm_frame = ctk.CTkFrame(self.dyn, fg_color="transparent")
        self.confirm_q = ctk.CTkLabel(self.confirm_frame, text="", wraplength=290, font=("Segoe UI", 14))
        self.confirm_q.pack(pady=4)
        ctk.CTkButton(self.confirm_frame, text="✔  Yes, add it", height=40, command=self.confirm_yes).pack(fill="x", pady=2)
        ctk.CTkButton(self.confirm_frame, text="✖  No, register as new", height=34, fg_color=GREY,
                      hover_color="#4a546a", command=self.confirm_no).pack(fill="x", pady=2)

        self.register_frame = ctk.CTkFrame(self.dyn, fg_color="transparent")
        ctk.CTkLabel(self.register_frame, text="Register this product", font=("Segoe UI", 15, "bold"),
                     anchor="w").pack(fill="x")
        ctk.CTkLabel(
            self.register_frame,
            text="Example: Vaseline Intensive Care Lotion · Vaseline · 200 ml · 149.50",
            text_color=MUTED, wraplength=300, justify="left", anchor="w"
        ).pack(fill="x", pady=(2, 5))
        self.name_var, self.cat_var = tk.StringVar(), tk.StringVar(value="Personal Care")
        self.name_entry = ctk.CTkEntry(
            self.register_frame, textvariable=self.name_var, height=38,
            placeholder_text="Product name (e.g. Vaseline Intensive Care Lotion)"
        )
        self.name_entry.pack(fill="x", pady=4)
        self.name_entry.bind("<Return>", lambda e: self.save_registration())
        self.brand_var, self.size_var, self.price_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        row = ctk.CTkFrame(self.register_frame, fg_color="transparent")
        row.pack(fill="x", pady=2)
        row.grid_columnconfigure((0, 1, 2), weight=1)
        for c, (var, ph) in enumerate([
            (self.brand_var, "Brand (e.g. Vaseline)"),
            (self.size_var, "Size (e.g. 200 ml)"),
            (self.price_var, "Price (e.g. 149.50)")
        ]):
            ctk.CTkEntry(row, textvariable=var, placeholder_text=ph, height=34).grid(
                row=0, column=c, sticky="ew", padx=2)
        ctk.CTkLabel(self.register_frame, text="Category", text_color=MUTED,
                     anchor="w").pack(fill="x", pady=(4, 0))
        ctk.CTkOptionMenu(self.register_frame, variable=self.cat_var, values=CATEGORIES, height=36,
                          fg_color="#2f3848", button_color="#2f3848").pack(fill="x", pady=2)
        self.save_scan_reference_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self.register_frame, text="Save this scanned image for future recognition",
            variable=self.save_scan_reference_var, onvalue=True, offvalue=False,
            font=("Segoe UI", 12)
        ).pack(fill="x", pady=(6, 2))
        ctk.CTkLabel(
            self.register_frame,
            text="When checked, scanning this product again can recognise it and add it to your list automatically.",
            text_color=MUTED, wraplength=300, justify="left", anchor="w"
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkButton(self.register_frame, text="💾  Save product & add to list", height=40,
                      command=self.save_registration).pack(fill="x", pady=(6, 2))
        ctk.CTkButton(self.register_frame, text="Cancel (Esc)", height=32, fg_color=GREY, hover_color="#4a546a",
                      command=self.cancel).pack(fill="x", pady=2)

        for text, cmd in [("↻  Reset scan", self.reset_scan), ("🎥  Retry camera", self.cam_restart)]:
            ctk.CTkButton(right, text=text, height=34, fg_color="transparent", border_width=1,
                          command=cmd).pack(fill="x", padx=10, pady=2)

    def build_list(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", pady=(4, 8))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self.refresh_list())
        ctk.CTkEntry(bar, textvariable=self.search_var, placeholder_text="🔍  Search", height=38,
                     width=260).pack(side="left")
        self.seg = ctk.CTkSegmentedButton(bar, values=["All", "To buy", "Bought"], command=self.set_filter,
                                          selected_color=GREEN_DK, selected_hover_color=GREEN_DK)
        self.seg.set("All")
        self.seg.pack(side="left", padx=12)
        ctk.CTkButton(bar, text="🖼  Export", width=100, height=38, fg_color=GREY, command=self.export_image).pack(side="right")
        ctk.CTkButton(bar, text="🧹  Clear bought", width=130, height=38, fg_color=GREY,
                      command=self.clear_bought).pack(side="right", padx=8)
        self.list_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        self.list_frame.pack(fill="both", expand=True)

    def build_products(self, tab):
        ctk.CTkLabel(tab, text="Registered products the app can recognise. Delete wrong registrations here.",
                     text_color=MUTED).pack(anchor="w", pady=(4, 6))
        self.prod_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        self.prod_frame.pack(fill="both", expand=True)

    # ----------------------------------------------------------- helpers
    def set_status(self, text):
        self.status.configure(text=text)

    def ctk_image(self, pil, size=None):
        img = ctk.CTkImage(pil, size=size or pil.size)
        self.refs_keep = (self.refs_keep + [img])[-6:]       # keep transient images alive
        return img

    def thumb(self, path, box=84):
        key = (path, box)
        if key not in self.cache:
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                im = Image.new("RGB", (box, box), "#444")      # missing image placeholder
            im.thumbnail((box, box))
            self.cache[key] = ctk.CTkImage(im, size=im.size)
        return self.cache[key]

    def show_panel(self, which):
        self.confirm_frame.pack_forget()
        self.register_frame.pack_forget()
        if which == "confirm":
            self.confirm_frame.pack(fill="x")
        elif which == "register":
            self.register_frame.pack(fill="x")

    def show_product(self, pid, conf, name_override=None):
        p = self.db.get_product(pid) if pid else None
        if p:
            self.res_img.configure(image=self.thumb(p["image"], 150), text="")
            self.res_name.configure(text=name_override or p["name"])
        self.conf_bar.set(max(0.0, min(conf, 1.0)))
        self.conf_lbl.configure(text=f"{conf * 100:.0f}% match")

    def clear_result(self):
        self.res_img.configure(image=self.blank, text="Product photo")
        self.res_name.configure(text="—")
        self.conf_bar.set(0)
        self.conf_lbl.configure(text="No match yet")

    # ----------------------------------------------------------- main loop
    def loop(self):
        self.poll_results()
        frame = self.cam.latest()
        if frame is None:
            self.cam_lbl.configure(image=self.blank, text="⚠ Camera not available.\n"
                                   "Check the cable or press “Retry camera”.")
        else:
            self.process(frame)
        self.after(30, self.loop)

    def process(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(w * 0.52), int(h * 0.15), int(w * 0.97), int(h * 0.92)   # zone on the mirrored view
        roi = frame[y1:y2, w - x2:w - x1]                      # same zone on the un-mirrored frame (text stays readable)
        now = time.time()
        ear, face_found = self.detect_blink(frame)
        present, box, motion = self.track_item(roi)
        if present and motion < MOTION_LIMIT:
            self.steady_since = self.steady_since or now
        else:
            self.steady_since = None
        steady_for = now - self.steady_since if self.steady_since else 0.0
        ready = steady_for >= STEADY_SECONDS

        if self.app_state == "SCAN":
            self.scan_logic(roi, box, present, ready, steady_for, face_found, now)
        elif self.app_state == "WAIT":
            self.wait_logic(present, now)
        self.blink_event = False
        self.draw(frame, (x1, y1, x2, y2), box, roi.shape[1], present, ready, ear)

    # ---- facial landmarks
    def detect_blink(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (480, int(480 * h / w)))
        res = self.face_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            self.closed_frames = 0
            return None, False
        sh, sw = small.shape[:2]
        lm = [(p.x * sw, p.y * sh) for p in res.multi_face_landmarks[0].landmark]
        ear = (eye_aspect_ratio(lm, LEFT_EYE) + eye_aspect_ratio(lm, RIGHT_EYE)) / 2
        if ear < EAR_THRESHOLD:
            self.closed_frames += 1
        else:
            if BLINK_MIN <= self.closed_frames <= BLINK_MAX:
                self.blink_event = True
            self.closed_frames = 0
        return ear, True

    # ---- motion / object tracking
    def track_item(self, roi):
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if self.bg is None or self.bg.shape != gray.shape:
            self.bg, self.prev = gray.astype("float32"), gray
        mask = cv2.threshold(cv2.absdiff(gray, cv2.convertScaleAbs(self.bg)), 28, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), None, iterations=3)
        present = mask.mean() / 255 > PRESENT_RATIO
        motion = float(cv2.absdiff(gray, self.prev).mean())
        self.prev = gray
        box = None
        if present:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                box = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        else:
            cv2.accumulateWeighted(gray, self.bg, 0.05)       # keep learning the empty background
        return present, box, motion

    @staticmethod
    def crop_item(roi, box):
        if not box:
            return roi.copy()
        bx, by, bw, bh = box
        pad, (H, W) = int(0.08 * max(bw, bh)), roi.shape[:2]
        crop = roi[max(by - pad, 0):min(by + bh + pad, H), max(bx - pad, 0):min(bx + bw + pad, W)]
        return crop.copy() if crop.size else roi.copy()

    # ---- state: SCAN
    def scan_logic(self, roi, box, present, ready, steady_for, face_found, now):
        self.update_steps(present, ready, steady_for)
        if not present:
            self.votes.clear()
            self.set_status("Ready — show a product in the box.")
            return
        if not ready:
            self.set_status("Hold it steady…")
            return
        if self.blink_event:                                   # blink = register this photo
            self.begin_register(self.crop_item(roi, box))
            return
        self.set_status("Recognising…")
        if not self.rec_busy and now - self.last_submit > RECOGNISE_EVERY:
            crop = self.crop_item(roi, box)
            if self.worker.submit(crop):
                self.rec_busy, self.last_submit, self.last_crop = True, now, crop

    def poll_results(self):
        try:
            r = self.worker.outq.get_nowait()
        except queue.Empty:
            return
        self.rec_busy = False
        if self.app_state == "SCAN":
            self.handle_result(r)

    def handle_result(self, r):
        pid, conf, inl = r["pid"], r["conf"], r["inliers"]
        if pid and inl >= AUTO_INL and conf >= AUTO_CONF:
            level = "auto"
        elif pid and inl >= MAYBE_INL and conf >= MAYBE_CONF:
            level = "maybe"
        else:
            level = "unknown"
        self.votes.append((level, pid, conf))
        v = list(self.votes)
        if pid:
            self.show_product(pid, conf)
        if len(v) >= 2 and all(x[0] == "auto" and x[1] == pid for x in v[-2:]):
            self.auto_add(pid, conf)
        elif len(v) >= 3 and all(x[0] != "unknown" and x[1] == pid for x in v):
            self.begin_confirm(pid, conf)
        elif len(v) >= 3 and all(x[0] == "unknown" for x in v):
            self.set_status("Product not matched yet. Keep the label facing the camera, move closer, and improve the lighting. If it is already saved, try scanning its front label clearly.")
            # Do not force registration for every failed match; the product may already be saved.

    # ---- state: WAIT (item must leave before the next scan)
    def wait_logic(self, present, now):
        if present:
            self.absent_since = None
            return
        self.absent_since = self.absent_since or now
        if now - self.absent_since >= REMOVE_SECONDS:
            self.reset_scan(keep_bg=True)

    # ----------------------------------------------------------- actions
    @db_guard
    def auto_add(self, pid, conf):
        p = self.db.get_product(pid)
        qty = self.db.add_to_list(pid)
        self.show_product(pid, conf)
        self.app_state, self.absent_since = "WAIT", None
        self.votes.clear()
        self.update_steps(True, True, STEADY_SECONDS, done=True)
        self.set_status(f"✅ Product recognised and added to shopping list:\n{p['name']}  (qty {qty}).\n"
                        "Remove it to scan another.")
        self.refresh_all()

    def begin_confirm(self, pid, conf):
        p = self.db.get_product(pid)
        self.app_state, self.candidate = "CONFIRM", pid
        self.show_product(pid, conf)
        self.confirm_q.configure(text=f"Is this “{p['name']}”?")
        self.set_status("🤔 Not sure — please confirm.")
        self.show_panel("confirm")

    @db_guard
    def confirm_yes(self):
        if self.app_state != "CONFIRM":
            return
        pid = self.candidate
        feats = self.rec.features(self.last_crop)
        if feats is not None and len(feats[1]) >= MIN_FEATS:        # learn from the confirmed photo
            self.rec.refs = self.rec.refs + [self.db.add_reference(pid, enhance(self.last_crop), feats)]
        self.show_panel(None)
        self.auto_add(pid, 1.0)

    def confirm_no(self):
        self.begin_register(self.last_crop)

    def begin_register(self, crop):
        if crop is None or crop.size == 0:
            return
        self.app_state, self.pending_raw, self.pending_img = "REGISTER", crop, enhance(crop)
        pil = to_pil(self.pending_img)
        pil.thumbnail((290, 150))
        self.res_img.configure(image=self.ctk_image(pil), text="")
        self.res_name.configure(text="New product")
        self.conf_bar.set(0)
        self.conf_lbl.configure(text="Not in the database yet")
        self.set_status("📝 Register this product.")
        self.name_var.set(self.guess_name(crop))
        for v in (self.brand_var, self.size_var, self.price_var):
            v.set("")
        self.show_panel("register")
        self.name_entry.focus_set()

    def guess_name(self, img):
        if not HAS_OCR:
            return ""
        try:
            txt = pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), config="--psm 6")
            words = [ln.strip() for ln in txt.splitlines() if len(ln.strip()) > 3]
            return words[0][:30].title() if words else ""
        except Exception:
            return ""

    @db_guard
    def save_registration(self):
        if self.app_state != "REGISTER":
            return
        name = self.name_var.get().strip()
        if not name:
            self.set_status("⚠ Please enter a product name. Example: Vaseline Intensive Care Lotion.")
            self.name_entry.focus_set()
            return

        save_reference = self.save_scan_reference_var.get()
        feats = self.rec.features(self.pending_raw) if save_reference else None
        if save_reference and (feats is None or len(feats[1]) < MIN_FEATS):
            self.set_status("⚠ Not enough detail in this photo. Move closer, add light, show the label, or uncheck image saving.")
            return

        # Check for a possible duplicate only when this image will be saved for recognition.
        if save_reference:
            m = self.rec.identify(self.pending_raw)
            if m["pid"] and m["inliers"] >= AUTO_INL:
                existing = self.db.get_product(m["pid"])
                if existing["name"].lower() != name.lower() and messagebox.askyesno(
                        "Possible duplicate",
                        f"This looks like “{existing['name']}”.\\nAdd it as another photo of that product?"):
                    name = existing["name"]

        try:
            price = float(self.price_var.get().replace(",", "").strip() or 0)
            if price < 0:
                raise ValueError
        except ValueError:
            self.set_status("⚠ Price must be a valid non-negative number, e.g. 149.50")
            return

        if save_reference:
            pid, is_new, ref = self.db.register(
                name, self.cat_var.get(), self.pending_img, feats,
                self.brand_var.get().strip(), self.size_var.get().strip(), price)
            self.rec.refs = self.rec.refs + [ref]
            verb = "registered with a saved scan image" if is_new else "updated with a saved scan image"
        else:
            pid, is_new = self.db.add_manual_product(
                name, self.cat_var.get(), self.brand_var.get().strip(),
                self.size_var.get().strip(), price)
            verb = "saved without a scan image" if is_new else "updated (scan image not saved)"

        qty = self.db.add_to_list(pid)
        self.show_panel(None)
        self.app_state, self.absent_since = "WAIT", None
        extra = ("Scan it again later to auto-add it." if save_reference else
                 "To recognise it automatically later, scan it again and save its image.")
        self.set_status(f"✅ “{name}” {verb} and added to your list (qty {qty}).\\n{extra}\\nRemove it to scan another.")
        self.show_product(pid, 1.0)
        self.refresh_all()

    def cancel(self):
        if self.app_state in ("CONFIRM", "REGISTER"):
            self.show_panel(None)
            self.app_state, self.absent_since = "WAIT", None
            self.set_status("Cancelled. Remove the item to scan another.")

    def reset_scan(self, keep_bg=False):
        """Restart recognition without closing the app."""
        self.show_panel(None)
        self.app_state, self.votes = "SCAN", deque(maxlen=3)
        self.steady_since = self.absent_since = None
        if not keep_bg:
            self.bg = None
        self.clear_result()
        self.set_status("Ready — show a product in the box.")

    def cam_restart(self):
        self.cam.restart_flag = True
        self.set_status("Restarting camera…")

    def update_steps(self, present, ready, steady_for, done=False):
        for (lbl, text), d in zip(self.steps, [present, ready, done]):
            lbl.configure(text=text, text_color=GREEN if d else MUTED)
        self.progress.set(min(steady_for / STEADY_SECONDS, 1.0))

    def draw(self, frame, zone, box, roi_w, present, ready, ear):
        disp = cv2.flip(frame, 1)
        x1, y1, x2, y2 = zone
        col = (113, 204, 46) if ready else ((0, 200, 255) if present else (230, 230, 230))
        cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
        if box and self.app_state == "SCAN":
            bx, by, bw, bh = box
            dx = x1 + roi_w - (bx + bw)                         # mirror the box for display
            cv2.rectangle(disp, (dx, y1 + by), (dx + bw, y1 + by + bh), col, 3)
        if ear is not None:
            cv2.putText(disp, f"EAR {ear:.2f}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 80, 255) if ear < EAR_THRESHOLD else (113, 204, 46), 2)
        self.cam_img = ctk.CTkImage(to_pil(cv2.resize(disp, CAM_SIZE)), size=CAM_SIZE)
        self.cam_lbl.configure(image=self.cam_img, text="")

    # ----------------------------------------------------------- lists
    def refresh_all(self):
        self.refresh_list()
        self.refresh_products()

    def set_filter(self, v):
        self.filter_mode = v
        self.refresh_list()

    def refresh_list(self):
        for w in self.list_frame.winfo_children():
            w.destroy()
        q = self.search_var.get().lower().strip()
        rows = sorted(self.db.list_items(), key=lambda r: (
            r["bought"], CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else 99, r["name"]))
        todo = sum(not r["bought"] for r in rows)
        total = sum(r["price"] * r["quantity"] for r in rows if not r["bought"])
        est = f"  •  est. {CURRENCY}{total:,.2f}" if total else ""
        self.count_lbl.configure(text=f"{todo} to buy  •  {len(rows) - todo} bought{est}")
        rows = [r for r in rows if q in r["name"].lower()
                and (self.filter_mode == "All" or (self.filter_mode == "Bought") == bool(r["bought"]))]
        if not rows:
            ctk.CTkLabel(self.list_frame, text="Your list is empty.\nScan a product to add it to your shopping list.",
                         text_color=MUTED, font=("Segoe UI", 16)).pack(pady=80)
            return
        for r in rows:
            card = ctk.CTkFrame(self.list_frame, fg_color=CARD, corner_radius=14)
            card.pack(fill="x", pady=5, padx=4)
            ctk.CTkLabel(card, image=self.thumb(r["image"]), text="").pack(side="left", padx=10, pady=10)
            info = ctk.CTkFrame(card, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(info, text=r["name"], anchor="w", font=("Segoe UI", 17, "bold"),
                         text_color=MUTED if r["bought"] else "white").pack(fill="x")
            ctk.CTkLabel(info, anchor="w", text_color=MUTED,
                         text=f"{CAT_ICON.get(r['category'], '🛒')} {r['category']}"
                              + (f"  •  {r['brand']}" if r["brand"] else "")
                              + (f"  •  {r['size']}" if r["size"] else "")
                              + (f"  •  {CURRENCY}{r['price'] * r['quantity']:,.2f}" if r["price"] else "")
                         ).pack(fill="x")
            ctk.CTkButton(card, text="🗑", width=38, height=38, fg_color=GREY, hover_color="#c0392b",
                          command=lambda i=r["item_id"]: self.remove_item(i)).pack(side="right", padx=10)
            cb = ctk.CTkCheckBox(card, text="Bought", checkbox_width=24, checkbox_height=24,
                                 command=lambda i=r["item_id"], b=r["bought"]: self.set_bought(i, not b))
            cb.select() if r["bought"] else cb.deselect()
            cb.pack(side="right", padx=6)
            qf = ctk.CTkFrame(card, fg_color="transparent")
            qf.pack(side="right", padx=8)
            ctk.CTkButton(qf, text="−", width=30, height=30, fg_color=GREY,
                          command=lambda i=r["item_id"], n=r["quantity"] - 1: self.set_qty(i, n)).pack(side="left")
            ctk.CTkLabel(qf, text=str(r["quantity"]), width=34, font=("Segoe UI", 16, "bold")).pack(side="left")
            ctk.CTkButton(qf, text="+", width=30, height=30, fg_color=GREY,
                          command=lambda i=r["item_id"], n=r["quantity"] + 1: self.set_qty(i, n)).pack(side="left")

    def refresh_products(self):
        for w in self.prod_frame.winfo_children():
            w.destroy()
        for p in self.db.list_products():
            card = ctk.CTkFrame(self.prod_frame, fg_color=CARD, corner_radius=12)
            card.pack(fill="x", pady=4, padx=4)
            ctk.CTkLabel(card, image=self.thumb(p["image"], 64), text="").pack(side="left", padx=10, pady=8)
            ctk.CTkLabel(card, anchor="w", justify="left", text_color=MUTED,
                         text=f"{p['name']}\nID {p['id']}  •  {p['category']}  •  {p['brand'] or 'no brand'} {p['size']}  •  {p['refs']} photo(s)  •  "
                              f"{p['date_registered'][:10]}").pack(side="left")
            ctk.CTkButton(card, text="🗑", width=38, height=36, fg_color=GREY, hover_color="#c0392b",
                          command=lambda i=p["id"], n=p["name"]: self.delete_product(i, n)).pack(side="right", padx=10)

    @db_guard
    def set_qty(self, item_id, qty):
        if qty >= 1:
            self.db.update_item(item_id, quantity=qty)
            self.refresh_list()

    @db_guard
    def set_bought(self, item_id, value):
        self.db.update_item(item_id, bought=int(value))
        self.refresh_list()

    @db_guard
    def remove_item(self, item_id):
        self.db.remove_item(item_id)
        self.refresh_list()

    @db_guard
    def clear_bought(self):
        if messagebox.askyesno("Clear bought", "Remove all bought items from the list?"):
            self.db.clear_bought()
            self.refresh_list()

    @db_guard
    def delete_product(self, pid, name):
        if messagebox.askyesno("Delete product", f"Delete “{name}” and its photos?\nIt will no longer be recognised."):
            self.db.delete_product(pid)
            self.rec.refs = [r for r in self.rec.refs if r[0] != pid]
            self.refresh_all()

    def export_image(self):
        rows = [r for r in self.db.list_items() if not r["bought"]]
        if not rows:
            messagebox.showinfo("Export", "Nothing to buy yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="shopping_list.png",
                                            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        W, H = 700, 110
        sheet = Image.new("RGB", (W, 80 + H * len(rows)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((24, 26), f"Shopping list - {datetime.now():%d %b %Y}", fill="black")
        for n, r in enumerate(rows):
            y = 80 + n * H
            try:
                im = Image.open(r["image"]).convert("RGB")
                im.thumbnail((90, 90))
                sheet.paste(im, (24, y + 10))
            except Exception:
                pass
            d.text((140, y + 30), f"{r['name']}   x{r['quantity']}", fill="black")
            d.text((140, y + 55), r["category"], fill="gray")
            d.rectangle((W - 60, y + 35, W - 36, y + 59), outline="black", width=2)
        sheet.save(path)
        messagebox.showinfo("Export", "Saved!")

    # ----------------------------------------------------------- migration / exit
    def migrate_old_data(self):
        """Import the v1 list.json (if present) so earlier items are not lost."""
        old = DATA_DIR / "list.json"
        try:
            if not old.exists() or self.db.list_products():
                return
            for it in json.loads(old.read_text(encoding="utf-8")):
                img = cv2.imread(it["image"])
                feats = self.rec.features(img) if img is not None else None
                if feats is None:
                    continue
                pid, _, ref = self.db.register(it["name"], it.get("category", "Other"), img, feats)
                self.rec.refs = self.rec.refs + [ref]
                qty = int(it["qty"]) if str(it.get("qty", "1")).isdigit() else 1
                self.db.add_to_list(pid, qty, it.get("bought", False))
            old.rename(old.with_suffix(".json.migrated"))
        except Exception as e:
            print("Migration skipped:", e)

    def on_close(self):
        self.cam.stop()
        self.worker.inq.put(None) if not self.worker.inq.full() else None
        self.face_mesh.close()
        self.db.con.close()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
