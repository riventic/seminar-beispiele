# 🎓 BVL-Seminar: KI-Modelle Schritt für Schritt mit LLM-Unterstützung entwickeln

Dieses Repository ist die **Referenz für die Seminarteilnehmer**. Es zeigt an zwei durchgängigen Beispielen, wie man ein Machine-Learning-Projekt **mit Hilfe eines KI-Assistenten (z. B. Gemini in Google Colab oder ChatGPT)** von der Rohdatei bis zur fertigen Web-App entwickelt – ohne dass man jede Code-Zeile selbst schreiben muss.

Die Kernidee des Seminars: **Nicht den Code auswendig lernen, sondern lernen, das Problem in saubere Arbeitsschritte zu zerlegen und für jeden Schritt einen guten Prompt zu formulieren.** Der KI-Assistent erzeugt den Code, Sie validieren und korrigieren ihn.

---

## 🧭 Inhaltsverzeichnis

1. [Überblick](#-überblick)
2. [Repository-Struktur](#-repository-struktur)
3. [Die beiden Beispielprojekte](#-die-beiden-beispielprojekte)
4. [Der 7-Schritte-Workflow](#-der-7-schritte-workflow-das-rezept)
5. [Mit den Prompts arbeiten](#-mit-den-prompts-arbeiten)
6. [Schnellstart in Google Colab](#-schnellstart-in-google-colab)
7. [Lokal installieren & ausführen](#-lokal-installieren--ausführen)
8. [Die Gradio-Web-App nutzen](#-die-gradio-web-app-nutzen)
9. [Daten im Detail](#-daten-im-detail)
10. [Technologie-Stack](#-technologie-stack)
11. [Tipps & häufige Stolpersteine](#-tipps--häufige-stolpersteine)
12. [Lizenz](#-lizenz)

---

## 🔭 Überblick

Im Seminar bearbeiten wir zwei klassische ML-Aufgaben aus dem Bereich
**Supervised Learning / Regression**:

| Projekt | Ordner | Was wird vorhergesagt? | Rolle im Seminar |
|---|---|---|---|
| **Transportzeit-Prognose** | [`demand_forecasting/`](demand_forecasting/) | Lieferzeit in Tagen (`delivery_time_days`) | **Komplette Beispiellösung** (Live-Demo mit fertigem Notebook) |
| **Produktqualitäts-Vorhersage** | [`quality_prediction/`](quality_prediction/) | Produktqualität einer Röstmaschine (`quality`) | **Übungsprojekt** (Sie bauen es selbst anhand der Prompts) |

Beide Projekte folgen demselben Vorgehen und werden **vollständig in Google Colab** über eine Abfolge von Prompts entwickelt.

---

## 📂 Repository-Struktur

```
seminar-beispiele/
│
├── README.md                         ← Dieses Dokument
├── LICENSE                           ← MIT-Lizenz
│
├── demand_forecasting/               ← BEISPIEL 1: Transportzeit-Prognose (mit Lösung)
│   ├── BVL_Seminar_Beispiellösung.ipynb   ← Fertige Schritt-für-Schritt-Lösung (Referenz)
│   ├── BVL_Seminar_data.ipynb             ← Notebook, das die synthetischen Daten erzeugt
│   ├── delivery_data.csv                  ← Trainingsdaten (15.000 Zeilen, Trennzeichen ";")
│   ├── delivery_data_eval.csv             ← Mini-Datei zum Testen der Gradio-App (Upload)
│   ├── prompts/                           ← Die Prompts des Seminars (prompt_0 … prompt_7)
│   │   └── prompt_error.md                ← Vorlage, um Fehlermeldungen korrigieren zu lassen
│   └── prompts_optimized/                 ← Verbesserte/„optimierte" Varianten derselben Prompts
│
└── quality_prediction/               ← BEISPIEL 2: Produktqualität Röstmaschine (Übung)
    ├── train_data_merged.csv              ← Trainingsdaten (Trennzeichen ",")
    ├── test_data_merged.csv               ← Testdaten / Upload-Datei für die Gradio-App
    └── prompts/                           ← Prompts (prompt_0 … prompt_7 + prompts_all.md)
        └── prompts_all.md                 ← Alle Prompts gesammelt in einer Datei
```

> **Hinweis:** Im Ordner `demand_forecasting/` finden Sie die **fertige Lösung** als
> Notebook – ideal, um nachzuschlagen, wenn Sie nicht weiterkommen. Im Ordner
> `quality_prediction/` gibt es bewusst **kein** Lösungs-Notebook: Dieses Projekt
> erarbeiten Sie selbst mit denselben Prompts.

---

## 🧪 Die beiden Beispielprojekte

### 1️⃣ Transportzeit-Prognose (`demand_forecasting/`)

**Ziel:** Vorhersage der Transportzeit (in Tagen) für die Lieferung von Materialien
zwischen verschiedenen Standorten – national und international.

**Eingangsmerkmale (Features):**

| Merkmal | Spalte | Beispiel |
|---|---|---|
| Materialart | `material_type` | Elektronik, Rohstoffe, Bauteile … |
| Menge | `quantity` | 839 |
| Lieferantenstandort | `supplier_location` | Mannheim |
| Kundenstandort | `destination_location` | Leipzig |
| Distanz (km) | `distance_km` | 344.98 |
| Transportart | `transport_mode` | Straße, Schiene, Luft, See |
| Routentyp | `route_type` | domestic_germany, intra_european, intercontinental |
| Datum | `order_date` | 2023-06-20 |
| Wetter | `weather_conditions` | Klar, Regnerisch, Schnee … |
| Ferienzeit | `holiday_season` | TRUE / FALSE |

**Ziel-/Vorhersagevariable (Label):** `delivery_time_days`

**Modell & Leistung (laut Beispiellösung):**
Ein `RandomForestRegressor` in einer Scikit-learn-Pipeline (mit `ColumnTransformer`,
`StandardScaler`, `OneHotEncoder`) erreicht auf den Testdaten:

* **R² ≈ 0.98** – das Modell erklärt rund 98 % der Streuung der Transportzeiten.
* **RMSE ≈ 3.82 Tage** – durchschnittlicher Vorhersagefehler.

> Die Trainingsdaten wurden synthetisch erzeugt (siehe `BVL_Seminar_data.ipynb`):
> echte Städte mit Geokoordinaten, realistische Distanzen über die Haversine-Formel,
> distanz- und routenabhängige Transportarten sowie Zollabfertigungszeiten für
> internationale Routen.

---

### 2️⃣ Produktqualitäts-Vorhersage Röstmaschine (`quality_prediction/`)

**Ziel:** Vorhersage der Produktqualität zukünftiger Chargen einer Röstmaschine.

**Prozessbeschreibung:**

* 5 Kammern, pro Kammer 3 Temperatursensoren → **15 Temperatursensoren**
* Messung minütlich, ein Röstzyklus dauert 60 Minuten
* Die Minutenwerte einer Stunde werden zu einem **Mittelwert** aggregiert
* Zwei zusätzliche Eingangsgrößen: **Schichthöhe** und **Feuchtigkeitsgehalt**

**Eingangsmerkmale (Features):**

| Merkmal | Spalten |
|---|---|
| 15 Temperatursensoren (5 Kammern × 3) | `T_data_1_1` … `T_data_5_3` |
| Schichthöhe & Feuchtigkeitsgehalt | `H_data`, `AH_data` |
| Zeitstempel | `date_time` |

**Ziel-/Vorhersagevariable (Label):** `quality`

> Für dieses Projekt gibt es absichtlich keine fertige Lösung. Nutzen Sie die Prompts
> in `quality_prediction/prompts/` und – falls nötig – die Beispiellösung des ersten
> Projekts als Orientierung.

---

## 📋 Der 7-Schritte-Workflow (das „Rezept")

Beide Projekte folgen demselben roten Faden. **Jeder Arbeitsschritt = eine eigene
Code-Zelle** in Colab. Nach jedem Schritt wird kurz **validiert**, ob das Ziel erreicht
wurde, bevor der nächste Schritt beginnt.

| # | Schritt | Worum es geht | Wichtigste Werkzeuge |
|---|---|---|---|
| 1 | **Daten laden** | CSV einlesen, erste Sichtprüfung (`head`, `info`) | pandas |
| 2 | **Daten analysieren** | Mittelwert, Std, Min, Max; kategorische Spalten zählen | pandas `describe()` |
| 3 | **Daten visualisieren** | Zusammenhänge sichtbar machen (z. B. Zeit nach Routentyp/Wetter) | Altair |
| 4 | **Modell trainieren** | Features/Label trennen, Train/Test-Split, Pipeline + Regressor | scikit-learn |
| 5 | **Modell evaluieren** | MAE, MSE, RMSE, R² berechnen und **interpretieren** | scikit-learn |
| 6 | **Modell speichern** | Trainierte Pipeline als Datei sichern (Wiederverwendung) | joblib |
| 7 | **Web-App bereitstellen** | Gradio-Oberfläche: CSV hochladen → Vorhersage → Ergebnis anzeigen | Gradio |

> Im Röstmaschinen-Projekt ist der Ablauf etwas feiner unterteilt (Daten vorbereiten,
> Evaluieren und Speichern als getrennte Schritte), folgt aber derselben Logik.

---

## 💬 Mit den Prompts arbeiten

Die Prompts sind das Herzstück des Seminars. Sie sind durchnummeriert und werden
**nacheinander** in den KI-Assistenten kopiert.

* **`prompt_0.md` – Rolle & Kontext setzen.**
  Definiert die Rolle des Assistenten („Experte für Python und ML"), beschreibt das
  Problem und den geplanten Workflow. Ganz wichtig: Am Ende steht die Anweisung,
  **noch keinen Code zu erzeugen**, sondern auf den ersten Schritt zu warten. So
  bekommt das Modell den vollen Kontext, bevor es loslegt.

* **`prompt_1.md` … `prompt_7.md` – je ein Arbeitsschritt.**
  Jeder Prompt fordert genau **einen** der oben genannten Schritte an. Sie kopieren
  einen Prompt, erhalten eine Code-Zelle, führen sie aus und prüfen das Ergebnis,
  bevor Sie den nächsten Prompt verwenden.

* **`prompt_5` (Ergebnis erklären lassen).**
  Hier fügen Sie die ausgegebenen Metriken ein und lassen sich vom Modell
  verständlich erklären, **ob das Ergebnis gut ist** – sehr nützlich zum Lernen.

* **`prompt_error.md` – Fehler korrigieren lassen.**
  Eine Vorlage: Wenn beim Ausführen ein Fehler auftritt, kopieren Sie diese Vorlage,
  fügen die **vollständige Fehlermeldung** ein und lassen den Code korrigieren.

* **`prompts_optimized/` (nur Projekt 1).**
  Verbesserte Fassungen derselben Prompts. Vergleichen Sie sie mit den Originalen –
  Sie sehen, wie präzisere Anforderungen, klare Ausgabeformate und eingebaute
  Validierungsschritte zu besserem Code führen. **Das ist eine zentrale Lernerfahrung
  des Seminars: Bessere Prompts → besseres Ergebnis.**

* **`prompts_all.md` (nur Projekt 2).**
  Alle Prompts des Röstmaschinen-Projekts gesammelt in einer Datei – praktisch zum
  Überblick.

---

## 🚀 Schnellstart in Google Colab

Empfohlener Weg für das Seminar – hier ist fast alles vorinstalliert.

1. **Daten hochladen:** Öffnen Sie ein neues Notebook auf
   [colab.research.google.com](https://colab.research.google.com) und laden Sie die
   passende CSV-Datei über das Datei-Panel (links, 📁) hoch – z. B. `delivery_data.csv`.
2. **Kontext setzen:** Kopieren Sie den Inhalt von `prompt_0.md` in den KI-Assistenten.
3. **Schritt für Schritt arbeiten:** Kopieren Sie nacheinander `prompt_1`, `prompt_2`, …
   Führen Sie den jeweils erzeugten Code in einer eigenen Zelle aus und prüfen Sie die
   Ausgabe.
4. **Gradio nachinstallieren:** Vor dem letzten Schritt (Web-App) in einer Zelle:
   ```python
   !pip install gradio
   ```
5. **Bei Fehlern:** `prompt_error.md` nutzen und die Fehlermeldung einfügen.

> 💡 Das fertige Notebook `BVL_Seminar_Beispiellösung.ipynb` können Sie auch direkt in
> Colab öffnen (`Datei → Notebook hochladen`) und Zelle für Zelle ausführen.

---

## 💻 Lokal installieren & ausführen

Falls Sie lieber lokal mit Jupyter arbeiten:

```bash
# Repository klonen
git clone <REPO-URL>
cd seminar-beispiele

# (Empfohlen) virtuelle Umgebung anlegen
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Abhängigkeiten installieren
pip install pandas numpy scikit-learn altair gradio joblib jupyter
# Optional, nur zum Neugenerieren der Transport-Daten:
pip install polars
```

Notebook starten:

```bash
jupyter notebook
# dann z. B. demand_forecasting/BVL_Seminar_Beispiellösung.ipynb öffnen
```

> ⚠️ **Pfade:** Die Notebooks laden die CSV per Dateiname (z. B.
> `pd.read_csv('delivery_data.csv', delimiter=';')`). Starten Sie Jupyter im
> jeweiligen Projektordner oder passen Sie den Pfad an, damit die Datei gefunden wird.

---

## 🖥️ Die Gradio-Web-App nutzen

Der letzte Schritt (`prompt_7`) erzeugt eine kleine Web-Oberfläche für **Batch-Vorhersagen**:

1. Beim Start lädt die App das gespeicherte Modell (z. B. `transport_model.joblib`).
2. Sie laden eine CSV-Datei **im erwarteten Format** hoch.
3. Ein Klick auf **„Vorhersage starten"** zeigt Eingabe + Vorhersage als Tabelle und
   bietet die Ergebnisse als CSV zum Download an.

Zum Ausprobieren liegen passende Mini-Dateien bereit:

* Projekt 1: [`demand_forecasting/delivery_data_eval.csv`](demand_forecasting/delivery_data_eval.csv)
* Projekt 2: [`quality_prediction/test_data_merged.csv`](quality_prediction/test_data_merged.csv)

> Die hochgeladene CSV muss exakt die erwarteten Spalten enthalten (Reihenfolge egal,
> Spaltennamen müssen stimmen). Die App akzeptiert sowohl Komma (`,`) als auch
> Semikolon (`;`) als Trennzeichen.

---

## 📊 Daten im Detail

### Transportdaten (`demand_forecasting/`)

* **`delivery_data.csv`** – 15.000 Zeilen, **Trennzeichen `;`**, keine fehlenden Werte.
  12 Spalten (siehe Feature-Tabelle oben) inkl. `order_id` und `delivery_time_days`.
* **`delivery_data_eval.csv`** – 3 Beispielzeilen zum Testen des CSV-Uploads.

### Röstmaschinen-Daten (`quality_prediction/`)

* **`train_data_merged.csv`** – ca. 2.900 Zeilen, **Trennzeichen `,`**.
  Spalten: `date_time`, `T_data_1_1` … `T_data_5_3` (15 Sensoren), `H_data`, `AH_data`,
  `quality`.
* **`test_data_merged.csv`** – 10 Zeilen zum Testen / für den App-Upload.

> Achten Sie auf das **unterschiedliche Trennzeichen** der beiden Datensätze
> (`;` vs. `,`) – ein häufiger Stolperstein beim Laden!

---

## 🛠️ Technologie-Stack

* **Daten & Verarbeitung:** pandas, NumPy (Datengenerierung: Polars)
* **Visualisierung:** Altair
* **Machine Learning:** scikit-learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`,
  `OneHotEncoder`, `RandomForestRegressor`)
* **Modell speichern/laden:** joblib
* **Web-App:** Gradio
* **Umgebung:** Google Colab / Jupyter

---

## 💡 Tipps & häufige Stolpersteine

* **Immer nur einen Schritt anfordern.** Lassen Sie das Modell nicht alles auf einmal
  generieren – so bleibt der Code verständlich und überprüfbar.
* **Nach jedem Schritt validieren.** Stimmen die Spalten? Sieht die Statistik plausibel
  aus? Sind die Metriken sinnvoll? Erst dann weiter.
* **Richtiges Trennzeichen wählen** (`;` bei den Transportdaten, `,` bei den
  Röstmaschinen-Daten).
* **Datei nicht gefunden?** CSV in Colab links ins Datei-Panel hochladen bzw. lokal im
  richtigen Ordner starten.
* **`gradio` ist in Colab nicht vorinstalliert** → vorher `!pip install gradio`.
* **Fehler systematisch beheben:** komplette Fehlermeldung kopieren und mit
  `prompt_error.md` korrigieren lassen.
* **Prompts vergleichen:** Schauen Sie sich `prompts/` vs. `prompts_optimized/` an, um
  ein Gefühl für gutes Prompt-Design zu bekommen.
* **Ein hoher R²-Wert ist nicht automatisch „gut".** Hinterfragen Sie ihn – bei
  synthetischen Daten sind sehr hohe Werte normal, in der Praxis selten.

---

## 📄 Lizenz

Dieses Projekt steht unter der **MIT-Lizenz** – siehe [`LICENSE`](LICENSE).
© 2026 Riventic GmbH.
