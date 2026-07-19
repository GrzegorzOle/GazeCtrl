#!/usr/bin/env python3
"""
gaze_grid.py - Gaze-Grid: 12-polowy system sterowania wzrokiem dla wsparcia
niepelnosprawnosci, oparty WYLACZNIE na zwyklej kamerze RGB (bez sprzetu Tobii,
bez ograniczen licencyjnych).

Caly stos jest w pelni open source, wolny od restrykcji licencyjnych sprzetu:
    - MediaPipe Face Landmarker   (Apache 2.0)
    - OpenCV                      (Apache 2.0 / BSD)
    - scikit-learn                (BSD)

INSTALACJA
    pip install mediapipe opencv-python scikit-learn numpy

MODEL (pobierz raz, dziala lokalnie / offline, bez chmury):
    wget -O face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

UZYCIE
    python gaze_grid.py --list-cameras   # 0) sprawdz, ktora kamera oddaje obraz
    python gaze_grid.py --debug          #    podglad tesczowek (weryfikacja indeksow)
    python gaze_grid.py --calibrate      # 1) patrz kolejno na 12 podswietlanych pol
    python gaze_grid.py --run            # 2) praca: wykrywanie pola + dwell-activation

Kazdy tryb przyjmuje --camera N (domyslnie 0), jesli laptop ma kilka urzadzen
wideo (czeste: kamera IR lub urzadzenie metadata obok wlasciwej kamery RGB).

Skalibrowany klasyfikator zapisywany jest do calibration_model.pkl i wczytywany
automatycznie przy --run. Kalibracje warto powtorzyc po zmianie pozycji
kamery/oswietlenia - klasyfikator jest per-uzytkownik i per-ustawienie.
"""

import argparse
import os
import pickle
import time
from collections import deque, Counter

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------------
# KONFIGURACJA
# ----------------------------------------------------------------------------
GRID_ROWS, GRID_COLS = 3, 4                 # 3x4 = 12 pol
N_ZONES = GRID_ROWS * GRID_COLS

DWELL_TIME_S = 0.7                          # czas patrzenia -> aktywacja pola
SMOOTH_WINDOW = 7                           # glosowanie wiekszosciowe (klatki)
MIN_CONFIDENCE = 0.55                       # prog pewnosci klasyfikatora
SAMPLES_PER_ZONE = 40                       # probek kalibracyjnych na pole

MAX_READ_FAILURES = 30                      # kolejne nieudane klatki -> przerwij

MODEL_TASK_PATH = "face_landmarker.task"
CALIB_MODEL_PATH = "calibration_model.pkl"

# Indeksy landmarkow MediaPipe (478 pkt, tesczowki = 468-477).
# UWAGA: przypisanie oko<->tesczowka jest zgodne z powszechna konwencja
# tutoriali MediaPipe, ale zweryfikuj wizualnie na wlasnej kamerze (patrz
# debug_draw_landmarks) - w razie potrzeby po prostu zamien pary miejscami.
EYE_A_CORNERS = (33, 133)
EYE_B_CORNERS = (362, 263)
IRIS_A = [469, 470, 471, 472]
IRIS_B = [474, 475, 476, 477]


# ----------------------------------------------------------------------------
# EKSTRAKCJA CECH Z TWARZY
# ----------------------------------------------------------------------------
def rotation_to_yaw_pitch(matrix_4x4: np.ndarray):
    """Przyblizone yaw/pitch (radiany) z macierzy transformacji glowy.
    Sluzy jako cecha pomocnicza kompensujaca ruch glowy - dokladna konwencja
    kata nie jest krytyczna, liczy sie tylko spojna korelacja z rotacja."""
    r = matrix_4x4[:3, :3]
    yaw = np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    pitch = np.arctan2(r[2, 1], r[2, 2])
    return yaw, pitch


def extract_features(landmarks, transform_matrix, frame_w, frame_h) -> np.ndarray:
    """Wektor cech: wzgledna pozycja tesczowki w kazdym oku (poziomo/pionowo)
    + przyblizona orientacja glowy (yaw/pitch)."""
    pts = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in landmarks])

    def iris_ratio(iris_idx, corner_idx):
        iris_c = pts[iris_idx].mean(axis=0)
        c1, c2 = pts[corner_idx[0]], pts[corner_idx[1]]
        eye_vec = c2 - c1
        eye_len = np.linalg.norm(eye_vec) + 1e-6
        t = np.dot(iris_c - c1, eye_vec) / (eye_len ** 2)      # poziomo: ~0..1
        eye_mid = (c1 + c2) / 2
        v = (iris_c[1] - eye_mid[1]) / eye_len                  # pionowo
        return t, v

    ax, ay = iris_ratio(IRIS_A, EYE_A_CORNERS)
    bx, by = iris_ratio(IRIS_B, EYE_B_CORNERS)
    yaw, pitch = rotation_to_yaw_pitch(transform_matrix)

    return np.array([ax, ay, bx, by, yaw, pitch], dtype=np.float32)


# ----------------------------------------------------------------------------
# KAMERA
# ----------------------------------------------------------------------------
def open_camera(index):
    """Otwiera kamere i weryfikuje, ze faktycznie oddaje klatki.

    Sam VideoCapture.isOpened() nie wystarcza: urzadzenia typu kamera IR czy
    metadata-only (na laptopach czesto /dev/video1..3) potrafia sie 'otworzyc',
    a nastepnie nie zwrocic ani jednej klatki - stad probny odczyt."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None, f"Nie udalo sie otworzyc kamery o indeksie {index}."
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None, (f"Kamera {index} otwarta, ale nie zwraca klatek "
                      f"(moze to urzadzenie IR/metadata?).")
    return cap, None


def list_cameras(max_index=10):
    """Zwraca indeksy kamer, ktore realnie oddaja obraz - do --list-cameras.

    Sondowanie nieistniejacych indeksow zawsze generuje halas na stderr z
    backendow V4L2/FFMPEG - wyciszamy go, bo tutaj bledy sa spodziewane."""
    prev_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    try:
        found = []
        for i in range(max_index):
            cap, err = open_camera(i)
            if cap is not None:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                found.append((i, w, h))
                cap.release()
        return found
    finally:
        cv2.utils.logging.setLogLevel(prev_level)


# ----------------------------------------------------------------------------
# SIATKA EKRANU (12 POL)
# ----------------------------------------------------------------------------
def zone_rects(screen_w, screen_h):
    cw, ch = screen_w // GRID_COLS, screen_h // GRID_ROWS
    rects = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            rects.append((col * cw, row * ch, (col + 1) * cw, (row + 1) * ch))
    return rects


def draw_grid(canvas, rects, active_zone=None, progress=0.0, labels=None):
    for i, (x1, y1, x2, y2) in enumerate(rects):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 1)
        text = labels[i] if labels else str(i + 1)
        cv2.putText(canvas, text, (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        if active_zone == i:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), -1)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            fill_w = int((x2 - x1) * progress)
            cv2.rectangle(canvas, (x1, y2 - 8), (x1 + fill_w, y2), (0, 220, 0), -1)


# ----------------------------------------------------------------------------
# DETEKTOR TWARZY (MediaPipe Face Landmarker, tryb VIDEO)
# ----------------------------------------------------------------------------
class FaceLandmarker:
    def __init__(self, model_path=MODEL_TASK_PATH):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
        return landmarks, matrix


# ----------------------------------------------------------------------------
# KLASYFIKATOR POLA (trenowany per-uzytkownik podczas kalibracji)
# ----------------------------------------------------------------------------
class ZoneClassifier:
    def __init__(self):
        self.clf = MLPClassifier(hidden_layer_sizes=(24, 16), max_iter=2000)
        self.trained = False

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.trained = True

    def fit_with_report(self, X, y):
        """Trenuje na czesci danych i zwraca trafnosc na odlozonym zbiorze,
        zeby uzytkownik od razu wiedzial, czy kalibracja sie udala. Finalny
        model uczony jest ponownie na calosci - odlozone probki tez sa cenne."""
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0)
        self.clf.fit(X_tr, y_tr)
        acc = float(self.clf.score(X_te, y_te))
        per_zone = {}
        pred = self.clf.predict(X_te)
        for z in range(N_ZONES):
            mask = y_te == z
            if mask.any():
                per_zone[z] = float((pred[mask] == z).mean())
        self.clf.fit(X, y)          # finalny model: cale dane
        self.trained = True
        return acc, per_zone

    def predict(self, feat_vec):
        probs = self.clf.predict_proba([feat_vec])[0]
        zone = int(np.argmax(probs))
        return zone, float(probs[zone])

    def save(self, path=CALIB_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)

    def load(self, path=CALIB_MODEL_PATH):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)
        self.trained = True


# ----------------------------------------------------------------------------
# WYGLADZANIE CZASOWE + DWELL ACTIVATION (warstwa "AI-predykcji")
# ----------------------------------------------------------------------------
class DwellTracker:
    """Zamienia zaszumiony strumien predykcji na pojedyncze, pewne zdarzenia
    aktywacji: glosowanie wiekszosciowe w oknie czasowym + prog pewnosci +
    wymagany czas patrzenia (dwell time), zeby uniknac falszywych trafien."""

    def __init__(self, on_activate=print):
        self.history = deque(maxlen=SMOOTH_WINDOW)
        self.current_zone = None
        self.zone_since = None
        self.fired = False
        self.on_activate = on_activate

    def update(self, zone, confidence):
        if confidence < MIN_CONFIDENCE:
            zone = None
        self.history.append(zone)
        votes = Counter([z for z in self.history if z is not None])
        stable_zone = votes.most_common(1)[0][0] if votes else None

        if stable_zone != self.current_zone:
            self.current_zone = stable_zone
            self.zone_since = time.time()
            self.fired = False

        progress = 0.0
        if self.current_zone is not None and self.zone_since is not None:
            progress = min((time.time() - self.zone_since) / DWELL_TIME_S, 1.0)
            if progress >= 1.0 and not self.fired:
                self.fired = True
                self.on_activate(self.current_zone)

        return self.current_zone, progress


# ----------------------------------------------------------------------------
# KALIBRACJA
# ----------------------------------------------------------------------------
def run_calibration(cap, landmarker, screen_w, screen_h):
    rects = zone_rects(screen_w, screen_h)
    X, y = [], []

    read_failures = 0

    for zone_idx in range(N_ZONES):
        collected = 0
        while collected < SAMPLES_PER_ZONE:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= MAX_READ_FAILURES:
                    cv2.destroyAllWindows()
                    print(f"Kamera przestala zwracac klatki "
                          f"({MAX_READ_FAILURES} nieudanych odczytow) - przerywam.")
                    return None, None
                continue
            read_failures = 0

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            draw_grid(canvas, rects, active_zone=zone_idx,
                      progress=collected / SAMPLES_PER_ZONE)
            cv2.putText(canvas, "Patrz na podswietlone pole (ESC = przerwij)",
                        (30, screen_h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (200, 200, 200), 2)
            cv2.imshow("Kalibracja", canvas)

            result = landmarker.process(frame)
            if result is not None:
                landmarks, matrix = result
                feat = extract_features(landmarks, matrix, frame.shape[1], frame.shape[0])
                X.append(feat)
                y.append(zone_idx)
                collected += 1

            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                return None, None

    cv2.destroyAllWindows()
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------------
# PODGLAD DEBUG - weryfikacja przypisania indeksow tesczowek
# ----------------------------------------------------------------------------
def run_debug(cap, landmarker):
    """Rysuje na obrazie z kamery wykryte tesczowki i kaciki oczu.

    Sluzy do wizualnej weryfikacji stalych IRIS_A/EYE_A_CORNERS (grupa A, na
    zolto) oraz IRIS_B/EYE_B_CORNERS (grupa B, na niebiesko) - jesli zolte
    punkty laduja na oku po drugiej stronie niz zolte kaciki, zamien pary
    miejscami w sekcji KONFIGURACJA."""
    read_failures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            read_failures += 1
            if read_failures >= MAX_READ_FAILURES:
                print("Kamera przestala zwracac klatki - koncze.")
                break
            continue
        read_failures = 0

        frame = cv2.flip(frame, 1)          # lustro: naturalniejszy podglad
        result = landmarker.process(frame)

        if result is None:
            cv2.putText(frame, "brak twarzy", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            landmarks, matrix = result
            h, w = frame.shape[:2]
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

            for iris_idx, corner_idx, color, name in (
                    (IRIS_A, EYE_A_CORNERS, (0, 255, 255), "A"),
                    (IRIS_B, EYE_B_CORNERS, (255, 200, 0), "B")):
                for i in iris_idx:
                    cv2.circle(frame, tuple(pts[i].astype(int)), 2, color, -1)
                for i in corner_idx:
                    cv2.drawMarker(frame, tuple(pts[i].astype(int)), color,
                                   cv2.MARKER_TILTED_CROSS, 10, 2)
                c = pts[iris_idx].mean(axis=0).astype(int)
                cv2.putText(frame, name, (c[0] + 8, c[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            feat = extract_features(landmarks, matrix, w, h)
            cv2.putText(frame,
                        f"A: {feat[0]:+.2f},{feat[1]:+.2f}  "
                        f"B: {feat[2]:+.2f},{feat[3]:+.2f}  "
                        f"yaw/pitch: {feat[4]:+.2f},{feat[5]:+.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        cv2.putText(frame, "ESC = wyjscie", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        cv2.imshow("Gaze-Grid debug", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# GLOWNA PETLA PRACY
# ----------------------------------------------------------------------------
def on_zone_activated(zone_idx):
    # TODO: podepnij tu docelowa akcje - np. odtworzenie slowa/dzwieku,
    # wyslanie klikniecia, zdarzenie do aplikacji AAC, itp.
    print(f"[AKTYWACJA] pole {zone_idx + 1}")


def run_live(cap, landmarker, classifier, screen_w, screen_h):
    rects = zone_rects(screen_w, screen_h)
    tracker = DwellTracker(on_activate=on_zone_activated)
    read_failures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            read_failures += 1
            if read_failures >= MAX_READ_FAILURES:
                print(f"Kamera przestala zwracac klatki "
                      f"({MAX_READ_FAILURES} nieudanych odczytow) - koncze.")
                break
            continue
        read_failures = 0

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        result = landmarker.process(frame)

        zone, conf = None, 0.0
        if result is not None:
            landmarks, matrix = result
            feat = extract_features(landmarks, matrix, frame.shape[1], frame.shape[0])
            zone, conf = classifier.predict(feat)

        active_zone, progress = tracker.update(zone, conf)
        draw_grid(canvas, rects, active_zone=active_zone, progress=progress)
        cv2.putText(canvas, f"conf: {conf:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.imshow("Gaze-Grid", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gaze-Grid: 12-polowy gaze tracking z kamery RGB")
    parser.add_argument("--calibrate", action="store_true", help="uruchom kalibracje")
    parser.add_argument("--run", action="store_true", help="uruchom detekcje na zywo")
    parser.add_argument("--debug", action="store_true",
                        help="podglad z kamery z zaznaczonymi tesczowkami")
    parser.add_argument("--list-cameras", action="store_true",
                        help="wypisz kamery, ktore realnie oddaja obraz")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamery (domyslnie 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        if cams:
            print("Dostepne kamery:")
            for i, w, h in cams:
                print(f"  --camera {i}   ({w}x{h})")
        else:
            print("Nie znaleziono dzialajacej kamery.")
        return

    if not (args.calibrate or args.run or args.debug):
        parser.print_help()
        return

    if not os.path.exists(MODEL_TASK_PATH):
        print(f"Brak pliku modelu: {MODEL_TASK_PATH}\nPobierz go poleceniem:\n"
              "  wget -O face_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
        return

    if args.run and not os.path.exists(CALIB_MODEL_PATH):
        print("Brak pliku kalibracji - uruchom najpierw: python gaze_grid.py --calibrate")
        return

    cap, err = open_camera(args.camera)
    if cap is None:
        print(f"{err}\nSprawdz dostepne urzadzenia: python gaze_grid.py --list-cameras")
        return

    try:
        landmarker = FaceLandmarker()

        if args.debug:
            run_debug(cap, landmarker)

        elif args.calibrate:
            X, y = run_calibration(cap, landmarker, args.width, args.height)
            if X is not None:
                clf = ZoneClassifier()
                acc, per_zone = clf.fit_with_report(X, y)
                clf.save()
                print(f"\nKalibracja zapisana do {CALIB_MODEL_PATH} ({len(X)} probek).")
                print(f"Trafnosc na odlozonych probkach: {acc:.1%}")
                weak = [z for z, a in per_zone.items() if a < 0.6]
                if weak:
                    print("Slabo rozpoznawane pola: "
                          + ", ".join(str(z + 1) for z in sorted(weak)))
                if acc < 0.7:
                    print("UWAGA: niska trafnosc. Sprobuj poprawic oswietlenie, "
                          "ustabilizowac pozycje glowy i powtorzyc kalibracje.\n"
                          "Sprawdz tez podglad: python gaze_grid.py --debug")

        elif args.run:
            clf = ZoneClassifier()
            clf.load()
            run_live(cap, landmarker, clf, args.width, args.height)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
