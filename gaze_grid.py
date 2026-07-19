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
import glob
import os
import pickle
import random
import time
from collections import deque, Counter
from typing import NamedTuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------------
# KONFIGURACJA
# ----------------------------------------------------------------------------
GRID_ROWS, GRID_COLS = 3, 4                 # 3x4 = 12 pol
N_ZONES = GRID_ROWS * GRID_COLS

FEATURE_DIM = 4                             # ax, ay, bx, by (patrz extract_features)

WEAK_ZONE_ACC = 0.8                         # ponizej tego pole uznajemy za slabe
                                            # i szukamy wzorca w pomylkach

ENSEMBLE_SEEDS = 5                          # ile sieci tworzy zespol klasyfikatora.
                                            # Pojedynczy MLP na 480 probkach zalezy
                                            # od ziarna az do 12 pkt trafnosci

# O ile korekta dryfu glowy musi poprawic trafnosc na odlozonych rundach, zeby
# w ogole ja wlaczyc. Korekta jest ZAWODNA i decyzja musi zapadac per sesja:
#
#     kalibracja        bez korekty      z korekta      zmiana
#     2026-07-19 pop.   82.5% +-10.9     86.9% +-7.4     +4.4
#     2026-07-19 wiecz. 72.1%  +-8.8     80.8% +-0.6     +8.7
#     2026-07-19 noc    55.6% +-10.4     43.8% +-22.9   -11.8
#
# W sesji nocnej dryf miedzyrundowy nie byl napedzany ruchem glowy - iloraz
# "przesuniecie przewidziane / faktyczne" wyszedl +0.02, -0.20, -0.75, +0.35,
# czyli w dwoch rundach ZLY ZNAK. Korekta odejmowala wtedy wielkosc niezwiazana
# z problemem. Nie da sie tego zalozyc z gory, wiec sprawdzamy za kazdym razem.
DRIFT_MIN_GAIN = 0.02                       # 2 pkt proc.

# O ile inny wariant musi pobic domyslny (klasyfikator bez korekty), zeby go
# zastapic. Drugi wybor obok korekty dryfu to TYP GLOWICY:
#
#   klasyfikator - 12 nieporownywalnych etykiet, nie wie ze pole 5 lezy
#                  miedzy 1 a 9
#   regresor     - przewiduje (kolumna, wiersz) jako liczby ciagle i dopiero
#                  potem przypisuje do pola, wiec korzysta z uporzadkowania
#
# Zmierzone na trzech kalibracjach z 19.07 (bez korekty dryfu, LORO):
#
#     sesja        klasyfikator   regresor    caly zysk siedzi w PIONIE
#     popoludnie      82.5%        79.2%      -3.3   (wiersz -2.9)
#     wieczor         72.1%        75.6%      +3.5   (wiersz +8.1)
#     noc             55.6%        63.1%      +7.5   (wiersz +2.9)
#
# Regresor pomaga tam, gdzie sygnalu brakuje, i lekko szkodzi tam, gdzie go
# starcza - typowy podpis regularyzacji. Zaden z wariantow nie wygrywa zawsze,
# stad wybor per sesja zamiast decyzji raz na zawsze.
VARIANT_MIN_GAIN = 0.02                     # 2 pkt proc.

DWELL_TIME_S = 0.7                          # czas patrzenia -> aktywacja pola
SMOOTH_WINDOW = 7                           # glosowanie wiekszosciowe (klatki)
MIN_CONFIDENCE = 0.55                       # prog pewnosci klasyfikatora
SAMPLES_PER_ZONE = 40                       # probek kalibracyjnych na pole
CALIB_ROUNDS = 4                            # rundy kalibracji; w kazdej pola w innej
                                            # losowej kolejnosci. Rozrywa korelacje
                                            # miedzy numerem pola a dryfem w czasie
                                            # (osuwanie glowy, zmiana swiatla) i daje
                                            # bloki do uczciwego odlozenia probek.
SETTLE_TIME_S = 1.5                         # czas na przeniesienie wzroku po zmianie
                                            # pola; klatki z sakady sa odrzucane.
                                            # Musi wystarczyc nie tylko na sakade
                                            # (~50 ms), ale na zauwazenie zmiany i
                                            # spokojne skupienie wzroku.

MAX_READ_FAILURES = 30                      # kolejne nieudane klatki -> przerwij

MODEL_TASK_PATH = "face_landmarker.task"
CALIB_MODEL_PATH = "calibration_model.pkl"
CALIB_DATA_PATH = "calibration_data.npz"    # surowe probki - pozwalaja analizowac
                                            # nieudana kalibracje bez powtarzania jej

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
    """Wektor cech: wzgledna pozycja tesczowki w kazdym oku (poziomo/pionowo).

    `landmarks` to albo obiekty MediaPipe (.x/.y), albo tablica (N, 2+) ze
    znormalizowanymi wspolrzednymi - ta druga postac siedzi w calibration_data.npz
    i pozwala przeliczyc nowy wektor cech bez powtarzania kalibracji."""
    if isinstance(landmarks, np.ndarray):
        pts = landmarks[:, :2] * np.array([frame_w, frame_h])
    else:
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

    # Probowane i odrzucone, zeby nie wracac do tego w kolko:
    # - yaw/pitch glowy: koduja dryf miedzy rundami, model sie ich chwyta
    #   i przestaje uogolniac na nowa runde;
    # - cechy powiek (polozenie tesczowki wzgledem szpary + rozwarcie):
    #   wygladaly na +9 pkt, ale to bylo przeuczenie doboru cech na tym samym
    #   zbiorze, na ktorym mierzylem. Na swiezej kalibracji daja +0.9 pkt,
    #   czyli tyle, ile wynosi rozrzut miedzy ziarnami MLP.
    # Glowne zrodlo bledu to dryf glowy w trakcie kalibracji: srednie ax
    # przesuwa sie miedzy runda pierwsza a ostatnia o 18% calego sygnalu
    # odrozniajacego kolumny. Tam sa punkty do odzyskania, nie w cechach.
    return np.array([ax, ay, bx, by], dtype=np.float32)


# ----------------------------------------------------------------------------
# KAMERA
# ----------------------------------------------------------------------------
# Przy domyslnych 640x480 oko ma tylko ~30 px szerokosci, a teczowka ~14 px -
# model teczowki dostaje wycinek, w ktorym nie ma czego powiekszac. Prosimy o
# 720p; kamera moze odmowic i zejsc do swojego maksimum, dlatego czytamy
# faktyczny rozmiar z powrotem zamiast zakladac, ze set() zadzialal.
REQUESTED_WIDTH, REQUESTED_HEIGHT = 1280, 720


def open_camera(index, width=REQUESTED_WIDTH, height=REQUESTED_HEIGHT):
    """Otwiera kamere i weryfikuje, ze faktycznie oddaje klatki.

    Sam VideoCapture.isOpened() nie wystarcza: urzadzenia typu kamera IR czy
    metadata-only (na laptopach czesto /dev/video1..3) potrafia sie 'otworzyc',
    a nastepnie nie zwrocic ani jednej klatki - stad probny odczyt."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None, f"Nie udalo sie otworzyc kamery o indeksie {index}."
    if width and height:
        # MJPG przed rozmiarem: przy nieskompresowanym YUYV pasmo USB zwykle
        # ogranicza 720p do ~10 fps, a mniej klatek to mniej probek na runde
        # kalibracji. Kolejnosc ma znaczenie - fourcc ustawiamy pierwszy.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
def detect_screen_size():
    """Rozdzielczosc podlaczonego ekranu z /sys/class/drm, albo (0, 0).

    Czytane wprost z jadra, wiec dziala tak samo pod Wayland i X11 i nie
    wymaga dodatkowej zaleznosci. Przy kilku ekranach bierze pierwszy
    podlaczony - siatka i tak jest sensowna tylko na jednym."""
    try:
        for status in sorted(glob.glob("/sys/class/drm/card*/status")):
            if open(status).read().strip() != "connected":
                continue
            modes = os.path.join(os.path.dirname(status), "modes")
            first = open(modes).readline().strip()
            if "x" in first:
                w, h = first.split("x")[:2]
                return int(w), int(h)
    except (OSError, ValueError):
        pass
    return 0, 0


def open_grid_window(name, fallback_w, fallback_h, fullscreen=True):
    """Otwiera okno siatki i zwraca jego realny rozmiar w pikselach.

    Plotno musi miec rozmiar okna, bo pola siatki wyznaczaja katy spojrzenia:
    kalibracja przeprowadzona w malym oknie uczy klasyfikator innego rozkladu
    niz ten, ktory wystapi przy pracy na pelnym ekranie.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(name, fallback_w, fallback_h)
        return fallback_w, fallback_h

    # Plotno zasiewajace musi miec proporcje ekranu. OpenCV wpasowuje obraz
    # w okno z zachowaniem proporcji, wiec obraz 16:9 na ekranie 16:10 zostaje
    # otoczony pasami tla, a getWindowImageRect zwraca wtedy rozmiar samego
    # wpasowanego obrazu - siatka utknelaby w zlych proporcjach na stale.
    seed_w, seed_h = detect_screen_size()
    if seed_w <= 0:
        seed_w, seed_h = fallback_w, fallback_h

    # okno musi sie raz wyrenderowac, zanim poda swoj rozmiar
    cv2.imshow(name, np.zeros((seed_h, seed_w, 3), dtype=np.uint8))
    cv2.waitKey(200)
    try:
        _, _, w, h = cv2.getWindowImageRect(name)
    except cv2.error:
        w = h = 0

    if w <= 0 or h <= 0:
        print(f"Nie udalo sie odczytac rozmiaru okna - uzywam "
              f"{seed_w}x{seed_h}. Mozesz podac wlasny: --width/--height")
        return seed_w, seed_h
    return w, h


def zone_rects(screen_w, screen_h):
    cw, ch = screen_w // GRID_COLS, screen_h // GRID_ROWS
    rects = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            rects.append((col * cw, row * ch, (col + 1) * cw, (row + 1) * ch))
    return rects


def describe_confusion_pattern(weak, confusions):
    """Podpowiedz, w ktora strone szukac przyczyny slabych pol.

    Kierunek pomylek niesie inna informacje niz sama trafnosc: mylenie pol
    lezacych nad soba wskazuje na pion (kompensacja pitch, kamera patrzaca
    pod katem), a obok siebie - na poziom (yaw, zbyt waskie kolumny).
    To tylko heurystyka na podstawie dominujacego kierunku, nie diagnoza.
    """
    # przy jednym-dwoch slabych polach "dominujacy kierunek" to pojedyncza
    # pomylka, a nie wzorzec - sugerowanie na tej podstawie przebudowy siatki
    # byloby myleniem szumu z geometria
    if len(weak) < 3:
        return ("\nZa malo slabych pol, zeby mowic o wzorcu - to raczej lokalny\n"
                "szum niz blad geometrii siatki.")

    vertical = horizontal = diagonal = 0
    for z in weak:
        if z not in confusions:
            continue
        other = confusions[z][0]
        same_row = z // GRID_COLS == other // GRID_COLS
        same_col = z % GRID_COLS == other % GRID_COLS
        if same_col:
            vertical += 1
        elif same_row:
            horizontal += 1
        else:
            diagonal += 1

    if not (vertical or horizontal or diagonal):
        return ""
    if vertical > horizontal and vertical >= diagonal:
        return ("\nPomylki sa glownie w pionie (pole mylone z tym nad/pod nim).\n"
                "Sprawdz kat kamery i czy podczas kalibracji nie zmieniala sie\n"
                "wysokosc glowy - pionowy sygnal jest slabszy niz poziomy.")
    if horizontal > vertical and horizontal >= diagonal:
        return ("\nPomylki sa glownie w poziomie (pole mylone z sasiadem obok).\n"
                "To zwykle znaczy, ze kolumny sa za waskie jak na dokladnosc\n"
                "z kamery RGB - rozwaz siatke 3x3 zamiast 3x4.")
    return ("\nPomylki nie ukladaja sie w jeden kierunek - to raczej ogolny szum\n"
            "(oswietlenie, odbicia w okularach, ruchy glowy) niz blad geometrii.")


def draw_fixation_point(canvas, rect, settling, progress, remaining):
    """Rysuje punkt fiksacji w srodku pola kalibrowanego.

    Podswietlony kwadrat pozwala wzrokowi bladzic po calym polu, wiec probki
    jednej klasy rozjezdzaja sie po duzym kacie i klasy zachodza na siebie.
    Punkt sciaga wzrok w jedno miejsce i zaciesnia klaster.
    """
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = max(10, canvas.shape[0] // 60)

    if settling:
        # cyfra dokladnie na srodku pola, a nie obok kropki: czytanie jej
        # trzyma wzrok w miejscu, w ktorym ma byc, gdy ruszy zbieranie
        text = str(int(remaining) + 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
        cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (130, 130, 130), 3)
        # kurczacy sie pierscien - sygnal peryferyjny, widac go bez
        # odrywania wzroku od srodka
        cv2.circle(canvas, (cx, cy), int(r + 3 * r * min(remaining, 1.5) / 1.5),
                   (70, 70, 70), 2)
        return

    cv2.circle(canvas, (cx, cy), r, (255, 255, 255), -1)
    cv2.circle(canvas, (cx, cy), r + 4, (0, 220, 0), 2)
    # pierscien postepu - widac, ile jeszcze trzeba patrzec, bez odrywania
    # wzroku od punktu
    cv2.ellipse(canvas, (cx, cy), (2 * r, 2 * r), -90, 0,
                int(360 * progress), (0, 220, 0), 4)


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
MLP_HIDDEN = (24, 16)

# Srodki wszystkich pol we wspolrzednych siatki - uklad odniesienia regresora
ZONE_CENTERS = np.array([[z % GRID_COLS, z // GRID_COLS]
                         for z in range(N_ZONES)], dtype=float)

# Szerokosc jadra zamieniajacego przewidziany punkt na rozklad po polach.
# Dobrana tak, zeby pewnosc regresora miala podobna skale co prawdopodobienstwa
# klasyfikatora - inaczej prog MIN_CONFIDENCE znaczylby co innego dla kazdego
# wariantu i regresor po cichu przestalby aktywowac pola.
REGRESSOR_SIGMA = 0.45


def _build_mlp(seed, kind):
    """Jedno miejsce na architekture - raport i model finalny musza uczyc sie
    tak samo, inaczej raportowana liczba przestaje dotyczyc zapisanego modelu."""
    if kind == "klasyfikator":
        return MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=2000,
                             random_state=seed)
    # regresor przewiduje (kolumna, wiersz) jako liczby ciagle, wiec wymaga
    # skalowania wejscia - inaczej nie schodzi z plaskiego minimum
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=MLP_HIDDEN,
                                      max_iter=2000, random_state=seed))


def _fit_ensemble(X, y, kind):
    if kind == "klasyfikator":
        return [_build_mlp(sd, kind).fit(X, y) for sd in range(ENSEMBLE_SEEDS)]
    target = np.column_stack([y % GRID_COLS, y // GRID_COLS]).astype(float)
    return [_build_mlp(sd, kind).fit(X, target) for sd in range(ENSEMBLE_SEEDS)]


def _ensemble_proba(models, kind, X):
    """Rozklad po WSZYSTKICH polach, w jednakowej postaci dla obu wariantow."""
    if kind == "klasyfikator":
        p = np.mean([m.predict_proba(X) for m in models], axis=0)
        # runda treningowa moze nie zawierac ktoregos pola - rozklad musi i tak
        # miec pelna szerokosc, bo indeksy kolumn to numery pol
        pelne = np.zeros((len(p), N_ZONES))
        pelne[:, models[0].classes_] = p
        return pelne
    # regresja: pole tym bardziej prawdopodobne, im blizej przewidzianego punktu
    P = np.mean([m.predict(X) for m in models], axis=0)
    d2 = ((P[:, None, :] - ZONE_CENTERS[None, :, :]) ** 2).sum(axis=2)
    p = np.exp(-d2 / (2 * REGRESSOR_SIGMA ** 2))
    return p / p.sum(axis=1, keepdims=True)


class ZoneClassifier:
    def __init__(self):
        # Zespol ENSEMBLE_SEEDS sieci roznacych sie tylko inicjalizacja, laczony
        # przez usrednienie prawdopodobienstw. Pojedynczy MLP na tak malym
        # zbiorze (480 probek) zalezy od ziarna az do 12 pkt trafnosci - zespol
        # zbija te wariancje i, co rownie wazne, sprawia ze raportowana po
        # kalibracji liczba dotyczy DOKLADNIE tego modelu, ktory zostaje
        # zapisany. Ziarna sa ustalone, wiec kalibracja pozostaje powtarzalna.
        self.models = []
        self.head_kind = "klasyfikator"
        self.trained = False
        self.last_acc_spread = 0.0
        self.variants = {}          # wyniki wszystkich wariantow z ostatniej kalibracji
        self.drift_k = None         # (2, FEATURE_DIM) wplyw yaw/pitch na cechy
        self.drift_ref = None       # (2,) pozycja glowy odniesienia z kalibracji
        self.drift_gain = None      # ile korekta dala na odlozonych rundach
        self.chosen_variant = ("klasyfikator", False)

    # -- korekta dryfu glowy --------------------------------------------------
    # Glowa dryfuje przez cala kalibracje i przesuwa cechy o ok. 10-18% calego
    # sygnalu odrozniajacego kolumny. Odejmujemy ten wplyw, szacujac go z
    # wariacji WEWNATRZ pola: przy ustalonym polu spojrzenie jest stale, wiec
    # to, co zostaje, pochodzi od samej glowy - dzieki temu korekta z definicji
    # nie moze zjesc sygnalu odrozniajacego pola.
    #
    # Korygujemy WYLACZNIE ax. Zmierzone na dwoch niezaleznych kalibracjach:
    #
    #                      2026-07-19 pop.   2026-07-19 wiecz.
    #     bez korekty        82.1% +-10.9      60.3% +-9.8
    #     sam ax             86.4%  +-7.8      75.3% +-3.0
    #     wszystkie 4        71.2% +-20.0      73.7% +-9.3
    #
    # Na cechach pionowych (ay, by) korekta przewiduje ZLY ZNAK przesuniecia
    # w 3 rundach na 4 i potrafi zabrac 11 pkt - stad maska. bx wypada tak samo
    # jak sam ax w granicach szumu, wiec zostawiamy wezszy wariant.
    DRIFT_FEATURES = (0,)                   # indeksy w wektorze cech: ax

    @staticmethod
    def _fit_drift(X, H, y):
        """Wspolczynniki k: jak cecha reaguje na glowe przy ustalonym polu."""
        Xc, Hc = X.copy(), H.copy()
        for z in np.unique(y):                # centrujemy obie strony w polu,
            m = y == z                        # wiec miedzypolowy sygnal wypada
            Xc[m] -= X[m].mean(axis=0)
            Hc[m] -= H[m].mean(axis=0)
        k, *_ = np.linalg.lstsq(Hc, Xc, rcond=None)
        mask = np.ones(k.shape[1], bool)
        mask[list(ZoneClassifier.DRIFT_FEATURES)] = False
        k[:, mask] = 0.0                      # reszty cech nie ruszamy
        return k

    def _apply_drift(self, X, H):
        if self.drift_k is None:
            return X
        return X - (H - self.drift_ref) @ self.drift_k

    def fit(self, X, y, H=None, kind="klasyfikator"):
        self.drift_k = self.drift_ref = None
        if H is not None:
            self.drift_k = self._fit_drift(X, H, y)
            self.drift_ref = H.mean(axis=0)
            X = self._apply_drift(X, H)
        self.head_kind = kind
        self.models = _fit_ensemble(X, y, kind)
        self.trained = True

    def fit_with_report(self, X, y, groups=None, H=None):
        """Trenuje na czesci danych i zwraca trafnosc na odlozonym zbiorze,
        zeby uzytkownik od razu wiedzial, czy kalibracja sie udala. Finalny
        model uczony jest ponownie na calosci - odlozone probki tez sa cenne.

        Odklada cale rundy kalibracji, a nie losowe probki. Probki w obrebie
        jednej rundy to kolejne, niemal identyczne klatki - losowy podzial
        rozdzielilby duplikaty na oba zbiory i zawyzal trafnosc.

        Kazda runda sluzy po kolei jako zbior testowy, a wynik jest srednia.
        Pojedyncza odlozona runda daje wynik obarczony duzym rozrzutem
        (na zapisanych probkach +-11 pkt proc.), wiec latwo wziac szczesliwa
        runde za poprawe.

        W kazdym foldzie oceniany jest caly zespol ENSEMBLE_SEEDS sieci, czyli
        dokladnie ta konstrukcja, ktora potem zostaje zapisana. Raportowana
        liczba dotyczy wiec tego modelu, ktorego uzywasz - nie sredniej po
        wariantach, ktorych nikt nie uruchomi.

        Dwie decyzje zapadaja WARUNKOWO, osobno dla kazdej sesji: korekta dryfu
        glowy i typ glowicy (klasyfikator albo regresor). Obie okazaly sie
        pomagac w jednych sesjach i szkodzic w innych, wiec sprawdzamy je za
        kazdym razem - patrz komentarze przy DRIFT_MIN_GAIN i VARIANT_MIN_GAIN."""
        if groups is not None and len(np.unique(groups)) > 1:
            folds = [(groups != h, groups == h) for h in np.unique(groups)]
        else:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                                      stratify=y, random_state=0)
            mask_tr = np.zeros(len(y), bool); mask_tr[tr] = True
            folds = [(mask_tr, ~mask_tr)]

        # Wariant domyslny jest uprzywilejowany: zeby go zmienic, kandydat musi
        # wygrac o VARIANT_MIN_GAIN. Bez tego progu wybieralibysmy zwyciezce
        # szumu - cztery warianty na czterech foldach to sporo okazji do pomylki.
        DOMYSLNY = ("klasyfikator", False)
        self.variants = {}
        for kind in ("klasyfikator", "regresor"):
            for z_korekta in (False, True):
                if z_korekta and H is None:
                    continue
                wynik = self._loro(X, y, folds, H if z_korekta else None, kind)
                self.variants[(kind, z_korekta)] = wynik

        acc_domyslny = self.variants[DOMYSLNY][0]
        najlepszy = max(self.variants, key=lambda v: self.variants[v][0])
        if self.variants[najlepszy][0] - acc_domyslny > VARIANT_MIN_GAIN:
            wybrany = najlepszy
        else:
            wybrany = DOMYSLNY
        self.chosen_variant = wybrany
        kind, z_korekta = wybrany
        use_H = H if z_korekta else None
        # ile dala korekta przy WYBRANYM typie glowicy - to jest liczba, ktora
        # mowi cos o tej sesji; porownywanie w poprzek typow mieszaloby efekty
        if H is not None:
            self.drift_gain = (self.variants[(kind, True)][0]
                               - self.variants[(kind, False)][0])

        acc, accs, y_te, pred = self.variants[wybrany]
        # rozrzut MIEDZY RUNDAMI - to on mowi o powtarzalnosci kalibracji
        self.last_acc_spread = float(np.std(accs))

        per_zone, confusions = {}, {}
        for z in range(N_ZONES):
            mask = y_te == z
            if mask.any():
                per_zone[z] = float((pred[mask] == z).mean())
                # z czym mylone jest pole - kierunek pomylki wskazuje przyczyne
                # inaczej niz sama trafnosc (pomylki w pionie sugeruja
                # kompensacje pitch, w poziomie - yaw albo uklad kolumn)
                wrong = pred[mask][pred[mask] != z]
                if len(wrong):
                    top = Counter(wrong.tolist()).most_common(1)[0]
                    confusions[z] = (int(top[0]), top[1] / int(mask.sum()))
        self.fit(X, y, use_H, kind)  # finalny model: cale dane, wybrany wariant
        return acc, per_zone, confusions

    def _loro(self, X, y, folds, H, kind):
        """Jeden przebieg walidacji. Zwraca (trafnosc, per runda, y, predykcje)."""
        accs, y_true, y_pred = [], [], []
        for train_mask, test_mask in folds:
            Xtr, Xte = X[train_mask], X[test_mask]
            if H is not None:
                # k i punkt odniesienia WYLACZNIE z rund treningowych - liczone
                # na calosci bylyby przeciekiem i zawyzalyby raportowana liczbe
                k = self._fit_drift(Xtr, H[train_mask], y[train_mask])
                ref = H[train_mask].mean(axis=0)
                Xtr = Xtr - (H[train_mask] - ref) @ k
                Xte = Xte - (H[test_mask] - ref) @ k
            # oceniamy CALY zespol, tak samo jak potem dziala predict()
            fold_models = _fit_ensemble(Xtr, y[train_mask], kind)
            pred_fold = _ensemble_proba(fold_models, kind, Xte).argmax(axis=1)
            accs.append(float((pred_fold == y[test_mask]).mean()))
            y_true.append(y[test_mask])
            y_pred.append(pred_fold)
        return (float(np.mean(accs)), accs,
                np.concatenate(y_true), np.concatenate(y_pred))

    def predict(self, feat_vec, head=None):
        """`head` to (yaw, pitch) biezacej klatki - bez tego korekta dryfu jest
        pomijana i model dostaje cechy w innej postaci niz te, na ktorych sie
        uczyl, wiec trafnosc cicho spada."""
        feat_vec = np.asarray(feat_vec, dtype=np.float32)
        if self.drift_k is not None and head is not None:
            feat_vec = self._apply_drift(feat_vec[None, :],
                                         np.asarray(head, float)[None, :])[0]
        probs = _ensemble_proba(self.models, self.head_kind, [feat_vec])[0]
        zone = int(np.argmax(probs))
        return zone, float(probs[zone])

    def save(self, path=CALIB_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "head_kind": self.head_kind,
                         "drift_k": self.drift_k,
                         "drift_ref": self.drift_ref}, f)

    def load(self, path=CALIB_MODEL_PATH):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        # starsze modele to goly MLPClassifier albo slownik z pojedynczym "clf" -
        # bez zespolu i bez wspolczynnikow korekty dzialalyby dalej, ale gorzej
        # i po cichu, wiec odmawiamy zamiast po cichu tracic punkty
        if not isinstance(blob, dict) or "head_kind" not in blob:
            raise ValueError(
                f"{path} pochodzi ze starszej wersji (brak zespolu sieci, typu "
                f"glowicy albo wspolczynnikow korekty dryfu glowy).\n"
                f"Uruchom kalibracje ponownie: python gaze_grid.py --calibrate")
        self.models = blob["models"]
        self.head_kind = blob["head_kind"]
        self.drift_k = blob["drift_k"]
        self.drift_ref = blob["drift_ref"]
        # model zapisany przed zmiana wektora cech dalby tu bardzo mylacy blad
        # gdzies w glebi sklearn, zamiast powiedziec wprost, co jest nie tak
        n = getattr(self.models[0], "n_features_in_", FEATURE_DIM)
        if n != FEATURE_DIM:
            raise ValueError(
                f"{path} pochodzi ze starszej wersji (model oczekuje {n} cech, "
                f"program wylicza {FEATURE_DIM}).\n"
                f"Uruchom kalibracje ponownie: python gaze_grid.py --calibrate")
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
class CalibrationSamples(NamedTuple):
    """Plon kalibracji: gotowe cechy + surowiec do ich ponownego wyliczenia."""
    X: np.ndarray            # (n, FEATURE_DIM) cechy wg biezacego extract_features
    y: np.ndarray            # (n,) indeks pola
    groups: np.ndarray       # (n,) numer rundy - do walidacji leave-one-round-out
    landmarks: np.ndarray    # (n, 478, 3) znormalizowane wspolrzedne twarzy
    matrices: np.ndarray     # (n, 4, 4) macierze transformacji glowy
    frame_size: np.ndarray   # (2,) szerokosc, wysokosc klatki z kamery


def run_calibration(cap, landmarker, screen_w, screen_h, fullscreen=True,
                    settle_time=SETTLE_TIME_S):
    screen_w, screen_h = open_grid_window("Kalibracja", screen_w, screen_h, fullscreen)
    rects = zone_rects(screen_w, screen_h)
    X, y, groups = [], [], []
    # Surowe landmarki i macierze glowy odkladane obok gotowych cech. Dzieki nim
    # nowy pomysl na wektor cech da sie przeliczyc na tych samych probkach,
    # zamiast powtarzac kalibracje przed kamera przy kazdej zmianie.
    raw_lm, raw_mat = [], []
    frame_size = None

    per_round = max(1, SAMPLES_PER_ZONE // CALIB_ROUNDS)
    read_failures = 0

    for round_idx in range(CALIB_ROUNDS):
        order = list(range(N_ZONES))
        random.shuffle(order)

        for zone_idx in order:
            collected = 0
            settle_until = time.time() + settle_time

            while collected < per_round:
                ok, frame = cap.read()
                if not ok:
                    read_failures += 1
                    if read_failures >= MAX_READ_FAILURES:
                        cv2.destroyAllWindows()
                        print(f"Kamera przestala zwracac klatki "
                              f"({MAX_READ_FAILURES} nieudanych odczytow) - przerywam.")
                        return None
                    continue
                read_failures = 0

                remaining = settle_until - time.time()
                settling = remaining > 0

                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                # sama siatka, bez podswietlenia pola i bez numerow - wzrok ma
                # trzymac sie punktu, wiec nic innego nie powinno go przyciagac
                draw_grid(canvas, rects, labels=[""] * N_ZONES)
                draw_fixation_point(canvas, rects[zone_idx], settling,
                                    collected / per_round, remaining)

                status = ("Przenies wzrok na kropke..." if settling
                          else "PATRZ W KROPKE - zbieram probki (ESC = przerwij)")
                cv2.putText(canvas, status, (30, screen_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(canvas, f"runda {round_idx + 1}/{CALIB_ROUNDS}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (150, 150, 150), 2)
                cv2.imshow("Kalibracja", canvas)

                # klatki z okresu przenoszenia wzroku lapia oko w trakcie sakady,
                # wiec nie odpowiadaja jeszcze podswietlonemu polu
                if not settling:
                    result = landmarker.process(frame)
                    if result is not None:
                        landmarks, matrix = result
                        feat = extract_features(landmarks, matrix,
                                                frame.shape[1], frame.shape[0])
                        X.append(feat)
                        y.append(zone_idx)
                        groups.append(round_idx)
                        # wspolrzedne znormalizowane (0..1) - razem z frame_size
                        # pozwalaja odtworzyc wejscie extract_features co do piksela
                        raw_lm.append([(lm.x, lm.y, lm.z) for lm in landmarks])
                        raw_mat.append(matrix)
                        frame_size = (frame.shape[1], frame.shape[0])
                        collected += 1

                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    return None

    cv2.destroyAllWindows()
    return CalibrationSamples(
        X=np.array(X), y=np.array(y), groups=np.array(groups),
        landmarks=np.array(raw_lm, dtype=np.float32),
        matrices=np.array(raw_mat, dtype=np.float32),
        frame_size=np.array(frame_size or (0, 0), dtype=np.int32))


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
            # yaw/pitch NIE sa czescia wektora cech (zakodowalyby dryf miedzy
            # rundami) - bierzemy je wprost z macierzy, bo w podgladzie chodzi
            # o obserwowanie tego dryfu na zywo.
            yaw, pitch = rotation_to_yaw_pitch(matrix)[:2]
            cv2.putText(frame,
                        f"A: {feat[0]:+.2f},{feat[1]:+.2f}  "
                        f"B: {feat[2]:+.2f},{feat[3]:+.2f}  "
                        f"yaw/pitch: {yaw:+.3f},{pitch:+.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            # szerokosc oka w px - kontrola, czy kamera faktycznie dala 720p
            eye_px = np.linalg.norm(pts[EYE_A_CORNERS[1]] - pts[EYE_A_CORNERS[0]])
            cv2.putText(frame, f"klatka: {w}x{h}   oko: {eye_px:.0f} px",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

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


def run_live(cap, landmarker, classifier, screen_w, screen_h, fullscreen=True):
    screen_w, screen_h = open_grid_window("Gaze-Grid", screen_w, screen_h, fullscreen)
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
            zone, conf = classifier.predict(feat, rotation_to_yaw_pitch(matrix))

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
    parser.add_argument("--settle", type=float, default=SETTLE_TIME_S,
                        help=f"czas na przeniesienie wzroku po zmianie pola, "
                             f"w sekundach (domyslnie {SETTLE_TIME_S}); zwieksz, "
                             f"jesli nie nadazasz za zmianami")
    parser.add_argument("--windowed", action="store_true",
                        help="siatka w oknie zamiast na pelnym ekranie "
                             "(kalibracja i praca musza uzywac tego samego trybu)")
    parser.add_argument("--width", type=int, default=1280,
                        help="szerokosc okna dla --windowed (domyslnie 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="wysokosc okna dla --windowed (domyslnie 720)")
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
            samples = run_calibration(cap, landmarker, args.width, args.height,
                                      fullscreen=not args.windowed,
                                      settle_time=args.settle)
            if samples is not None:
                X, y, groups = samples.X, samples.y, samples.groups
                H = np.array([rotation_to_yaw_pitch(m) for m in samples.matrices])
                clf = ZoneClassifier()
                acc, per_zone, confusions = clf.fit_with_report(X, y, groups, H)
                clf.save()
                np.savez_compressed(CALIB_DATA_PATH, **samples._asdict())
                print(f"\nKalibracja zapisana do {CALIB_MODEL_PATH} ({len(X)} probek).")
                print(f"Probki zapisane do {CALIB_DATA_PATH} (cechy + surowe "
                      f"landmarki - nowy wektor cech przeliczysz bez powtarzania "
                      f"kalibracji).")
                print(f"Trafnosc (srednia z {CALIB_ROUNDS} odlozonych rund): "
                      f"{acc:.1%} +-{clf.last_acc_spread:.1%}")
                # Oba wybory bywaja zawodne i zapadaja per sesja - bez tego
                # wydruku nie wiadomo, co model faktycznie w sobie ma
                kind, z_korekta = clf.chosen_variant
                print(f"Wybrany wariant: {kind}, korekta dryfu glowy "
                      f"{'WLACZONA' if z_korekta else 'wylaczona'}")
                print("  wszystkie sprawdzone warianty:")
                for (k, kor), wynik in sorted(clf.variants.items(),
                                              key=lambda t: -t[1][0]):
                    znacznik = " <- wybrany" if (k, kor) == clf.chosen_variant else ""
                    print(f"    {k:13s} korekta {'tak' if kor else 'nie'}: "
                          f"{wynik[0]:5.1%}{znacznik}")
                # Zawsze pokazujemy trzy najslabsze pola, nawet gdy sa przyzwoite.
                # Przy samym progu raport milczal przy trafnosci 68% i nie bylo
                # widac, ktore pola ciagna wynik w dol.
                ranking = sorted(per_zone.items(), key=lambda t: t[1])[:3]
                print("\nNajslabsze pola (pole: trafnosc -> najczestsza pomylka):")
                for z, a in ranking:
                    line = f"  pole {z + 1}: {a:.0%}"
                    if z in confusions:
                        other, share = confusions[z]
                        line += f" -> mylone z polem {other + 1} ({share:.0%} probek)"
                    print(line)
                weak = sorted(z for z, a in per_zone.items() if a < WEAK_ZONE_ACC)
                if weak:
                    print(describe_confusion_pattern(weak, confusions))
                if acc < 0.7:
                    print("UWAGA: niska trafnosc. Sprobuj poprawic oswietlenie, "
                          "ustabilizowac pozycje glowy i powtorzyc kalibracje.\n"
                          "Sprawdz tez podglad: python gaze_grid.py --debug")

        elif args.run:
            clf = ZoneClassifier()
            clf.load()
            run_live(cap, landmarker, clf, args.width, args.height,
                     fullscreen=not args.windowed)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
