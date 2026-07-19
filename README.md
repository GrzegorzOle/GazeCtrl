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
| Kalibracja 12 pól (4 rundy, losowa kolejność) + raport trafności | zaimplementowane, logika przetestowana na danych syntetycznych |
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

**3. Kalibracja** — 12 pól × 40 próbek, w 4 rundach po 10. Patrz na
podświetlane pole; po każdej zmianie masz 1,5 s na przeniesienie wzroku
(z odliczaniem w polu), zanim zacznie się zbieranie próbek. Jeśli to za szybko,
zwiększ `--settle 2.5` — lepiej kalibrować dłużej niż zbierać próbki, w których
wzrok jest jeszcze w drodze:

```bash
.venv/bin/python gaze_grid.py --calibrate
```

Pola zapalają się w **losowej kolejności, innej w każdej rundzie**. Przy stałej
kolejności 1→12 wszystko, co dryfuje w czasie kalibracji — osuwająca się głowa,
zmiana światła, zmęczenie oczu — byłoby skorelowane z numerem pola, a model
mógłby uczyć się dryfu zamiast spojrzenia.

Na koniec wypisywana jest trafność, lista słabo rozpoznawanych pól i **z czym
każde z nich jest mylone**. Kierunek pomyłek mówi więcej niż sama trafność:
pomyłki w pionie (pole mylone z tym nad nim) wskazują na kąt kamery i słabszy
sygnał pitch, pomyłki w poziomie — na zbyt wąskie kolumny, a brak wyraźnego
kierunku na ogólny szum (światło, odbicia w okularach). Poniżej ~70% warto
zadziałać zgodnie z tą podpowiedzią i powtórzyć.

Surowe próbki lądują w `calibration_data.npz`, więc nieudaną kalibrację można
analizować bez powtarzania jej.

Trafność liczona jest na **całej odłożonej ostatniej rundzie**, nie na losowych
próbkach. Próbki w obrębie jednej rundy to kolejne, niemal identyczne klatki —
losowy podział rozdzielałby ich duplikaty między zbiór treningowy i testowy, a
model rozpoznawałby klatki już widziane. Ta liczba jest więc niższa niż przy
losowym podziale, ale jest uczciwym oszacowaniem zachowania na żywo.

**4. Praca:**

```bash
.venv/bin/python gaze_grid.py --run
```

ESC kończy każdy tryb. Wszystkie tryby przyjmują `--camera N`.

Kalibracja i praca wyświetlają siatkę **na pełnym ekranie**, a rozmiar ekranu
wykrywany jest automatycznie. Jest to istotne dla poprawności: pola siatki
wyznaczają kąty spojrzenia, więc kalibracja w małym oknie nauczyłaby
klasyfikator innego rozkładu niż ten, który wystąpi przy pracy. Z tego samego
powodu **oba tryby muszą używać tego samego trybu wyświetlania** — jeśli
kalibrujesz z `--windowed`, pracuj też z `--windowed` (i tym samym
`--width`/`--height`).

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
| `calibration_data.npz` | surowe próbki kalibracyjne, do analizy (poza repo) |

Model i kalibracja są celowo poza repozytorium: model jest pobieralny, a
kalibracja dotyczy konkretnej osoby i konkretnego ustawienia kamery.

## Licencja

MIT — patrz [LICENSE](LICENSE).
