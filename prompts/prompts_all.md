# Prompt 0

# Rolle
Du bist ein Experte für Python und maschinelles Lernen.

Aufgabe:
Begleite mich in Google Colab Schritt für Schritt bei der Entwicklung eines
Regressionsmodells zur Vorhersage der Produktqualität einer Röstmaschine.

Geplanter Workflow:
1. Daten laden
2. Daten analysieren
3. Daten visualisieren
4. Daten vorbereiten
5. Modell trainieren
6. Modell evaluieren
7. Modell speichern
8. Benutzeroberfläche erstellen
9. Mit Gradio bereitstellen

# Problembeschreibung
Problem:
- 5 Kammern
- pro Kammer 3 Temperatursensoren
- Messung minütlich
- zusätzliche Eingangsgrößen: Schichthöhe und Feuchtigkeitsgehalt
- Dauer eines Röstzyklus: 60 Minuten
- Messungen der Stunde werden aggregiert (Mittelwert)
- Ziel: Vorhersage der Produktqualität zukünftiger Chargen

Anforderungen:
- Jeder Arbeitsschritt = eigene Code-Zelle
- Nur den jeweils angefragten Schritt bearbeiten
- Nur das Nötigste für den aktuellen Schritt ausgeben
- Code muss direkt in Google Colab ausführbar sein
- Klare, saubere, leicht verständliche Lösungen
- Sinnvolle Standardannahmen treffen, falls Informationen fehlen, und diese
kurz kenntlich machen

# Wichtiger Hinweis:
Erzeuge jetzt keine Antwort, keinen Code und keine Bestätigung.
Warte nur auf meinen nächsten Prompt mit dem ersten Schritt.

# Prompt 1
Im ersten Schritt möchte ich Code generieren, um die Daten zu laden. Die Daten liegen als csv-Datei vor (siehe Anhang).

# Prompt 2
Im zweiten Schritt möchte ich Code generieren, der die Daten analysiert. Ich möchte zum Beispiel die Mittelwerte, Standardabweichungen, Minima und Maxima jeder Spalte sehen.

# Prompt 3
Im dritten Schritt möchte ich Code generieren, der die Daten visualisiert. Speziell möchte ich ein Diagramm, dass mir die Qualität nach Schichthöhe & Feuchtigkeit zeigt.

# Prompt 4
Schritt 4 — KI-Modell trainieren

Erzeuge Python-Code für Google Colab, um ein **Regressionsmodell** zu trainieren, das die **Produktqualität** vorhersagt.

**Features:**
- Schichthöhe
- Feuchtigkeitsgehalt
- stündliche aggregierte Temperaturwerte der 15 Sensoren über 60 Minuten

**Label:**
- Produktqualität

**Anforderungen:**
- Daten in X und y aufteilen
- Trainings- und Testdaten erstellen
- Ein sinnvolles Regressionsmodell für tabellarische Daten verwenden
- Modell trainieren
- MAE, MSE, RMSE und R² berechnen
- Ein Diagramm zur Modellgüte erstellen
- Code soll in **einer Colab-Zelle** lauffähig sein
- Verwende verständliche Kommentare für Anfänger

**Ausgabeformat:**
- Zuerst nur Python-Code
- Danach eine kurze Erklärung

# Prompt 5
Bitte erkläre das folgende Ergebnis des Modelltrainings leicht verständlich und beurteile, ob es es sich um ein gutes Ergebnis handelt.

Ergebnis:

# Prompt 6
Im fünften Schritt möchte ich Code generieren, um das trainierte Modell als Datei zu speichern, damit ich es später leicht laden und wiederverwenden kann.

# Prompt 7
Im siebten Schritt möchte ich mit Gradio eine Web-App erzeugen und lokal hosten. Beim Starten der Benutzeroberfläche soll das gespeicherte Modell geladen werden. In dieser Benutzeroberfläche soll eine csv-Datei im gegebenen Format hochgeladen werden können. Dazu soll es eine Schaltfläche geben, um eine Vorhersage durchzuführen und anschließend die Eingangs- und Ausgangsdaten anzuzeigen.
