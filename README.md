# GazeCtrl / Gaze-Grid

Sterowanie komputerem wzrokiem przy użyciu **zwykłej kamery RGB laptopa** — bez
sprzętu IR (Tobii i podobne) i bez ograniczeń licencyjnych. Kamera obserwuje
użytkownika i rozpoznaje, na które z **12 pól ekranu** (siatka 3×4) patrzy;
zatrzymanie wzroku na polu przez zadany czas aktywuje je.

Projekt powstaje z myślą o wsparciu osób niepełnosprawnych — docelowo jako
warstwa wejściowa dla aplikacji typu AAC.

Cały stos jest open source: MediaPipe (Apache 2.0), OpenCV (Apache 2.0 / BSD),
scikit-learn (BSD). Działa **lokalnie i offline** — obraz z kamery nie opuszcza
komputera.

## Stan projektu

Wczesny prototyp. Pipeline jest kompletny i uruchamialny, ale **nie został
jeszcze przetestowany na żywo z prawdziwą twarzą** — nie ma więc zmierzonej
skuteczności. Kolejny krok to weryfikacja opisana w
[Pierwsze uruchomienie](#pierwsze-uruchomienie).

| Element | Stan |
|---|---|
| Detekcja twarzy i tęczówek (MediaPipe) | działa, zweryfikowane |
| Ekstrakcja cech | zaimplementowane, do weryfikacji na żywo |
| Kalibracja 12 pól + raport trafności | zaimplementowane, logika przetestowana na danych syntetycznych |
| Klasyfikacja pola + dwell activation | zaimplementowane, do weryfikacji na żywo |
| Akcja po aktywacji pola | **nie zaimplementowane** — `on_zone_activated()` tylko wypisuje numer pola |

## Jak to działa

```
kamera RGB
   ↓
MediaPipe Face Landmarker      478 punktów twarzy, w tym tęczówki (468–477)
   ↓
ekstrakcja cech                pozycja tęczówki względem kącików oka (oba oczy)
                               + yaw/pitch głowy jako kompensacja ruchu
   ↓
klasyfikator MLP               trenowany per-użytkownik podczas kalibracji,
                               12 klas = 12 pól siatki
   ↓
wygładzanie czasowe            głosowanie większościowe w oknie 7 klatek
                               + próg pewności 0,55
   ↓
dwell activation               0,7 s stabilnego patrzenia → zdarzenie aktywacji
```

### Dlaczego klasyfikacja pól, a nie współrzędne kursora

Appearance-based gaze estimation ze zwykłej kamery RGB daje błąd rzędu kilku
stopni kąta, co przy typowej odległości od ekranu przekłada się na kilka
centymetrów. To wystarcza, żeby rozpoznać **obszar** ekranu, ale nie do
wskazania pojedynczego piksela — do tego potrzebna byłaby kamera IR z
emiterami (dokładność <1°).

Dlatego model uczy się bezpośrednio klasyfikacji 12 dużych pól, zamiast
regresji współrzędnych x/y. Pomija to pośredni krok szacowania wektora
spojrzenia w stopniach (jak w L2CS-Net czy ETH-XGaze), który i tak wymagałby
kalibracyjnego mapowania na ekran.

Z tego samego powodu wygładzanie to głosowanie większościowe, a nie filtr
Kalmana — przy dyskretnych polach nie ma ciągłej trajektorii do wygładzania.
Kalman miałby sens dopiero przy przejściu na predykcję współrzędnych.

## Instalacja

Wymagany Python **3.12** — MediaPipe nie publikuje wheeli dla 3.14, więc
instalacja na systemowym Pythonie Fedory się nie powiedzie.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Bez `uv` — dowolny Python 3.12 w systemie:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model MediaPipe (3,6 MB, pobierany raz, potem działa offline):

```bash
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Pierwsze uruchomienie

Kolejność ma znaczenie — każdy krok weryfikuje założenie następnego.

**1. Znajdź właściwą kamerę.** Laptopy często mają kilka urządzeń wideo
(kamera IR, urządzenie metadata), z których część otwiera się, ale nie oddaje
obrazu. Poniższe polecenie pokazuje tylko te, które realnie zwracają klatki:

```bash
.venv/bin/python gaze_grid.py --list-cameras
```

**2. Sprawdź, czy tęczówki są wykrywane poprawnie.** Ten krok jest istotny:
przypisanie oko↔tęczówka w stałych `IRIS_A`/`IRIS_B` opiera się na powszechnej
konwencji tutoriali MediaPipe, ale warto potwierdzić je wizualnie.

```bash
.venv/bin/python gaze_grid.py --debug
```

Grupa A rysowana jest na żółto, B na niebiesko — punkty tęczówki jako kropki,
kąciki oka jako krzyżyki. **Jeśli żółte kropki są na innym oku niż żółte
krzyżyki, zamień `IRIS_A` z `IRIS_B`** w sekcji KONFIGURACJA w `gaze_grid.py`.
Bez tego kalibracja uczy się na przemieszanych cechach i wychodzi słabo bez
widocznej przyczyny.

**3. Kalibracja** — 12 pól × 40 próbek. Patrz kolejno na podświetlane pole:

```bash
.venv/bin/python gaze_grid.py --calibrate
```

Na koniec wypisywana jest trafność na odłożonych próbkach i lista słabo
rozpoznawanych pól. Poniżej ~70% warto poprawić oświetlenie, ustabilizować
pozycję głowy i powtórzyć.

**4. Praca:**

```bash
.venv/bin/python gaze_grid.py --run
```

ESC kończy każdy tryb. Wszystkie tryby przyjmują `--camera N`.

## Ograniczenia

- **Kalibracja jest per-użytkownik i per-ustawienie.** Zmiana pozycji względem
  kamery, oświetlenia albo samej kamery wymaga powtórzenia kalibracji.
- Duży wpływ mają: oświetlenie, ruchy głowy, okulary, kąt kamery.
- Dokładność nie pozwala na precyzyjne wskazywanie — patrz
  [wyżej](#dlaczego-klasyfikacja-pól-a-nie-współrzędne-kursora).

## Podpięcie własnej akcji

`on_zone_activated(zone_idx)` w `gaze_grid.py` jest wywoływane raz na każdą
aktywację pola (numeracja 0–11, wypisywana 1–12). Tam podpina się docelowe
zachowanie — odtworzenie słowa lub dźwięku, kliknięcie, zdarzenie do aplikacji
AAC.

## Pliki

| | |
|---|---|
| `gaze_grid.py` | całość — detekcja, kalibracja, klasyfikacja, pętla robocza |
| `requirements.txt` | zależności |
| `face_landmarker.task` | model MediaPipe, pobierany osobno (poza repo) |
| `calibration_model.pkl` | wynik kalibracji, tworzony lokalnie (poza repo) |

Model i kalibracja są celowo poza repozytorium: model jest pobieralny, a
kalibracja dotyczy konkretnej osoby i konkretnego ustawienia kamery.

## Licencja

MIT — patrz [LICENSE](LICENSE).
