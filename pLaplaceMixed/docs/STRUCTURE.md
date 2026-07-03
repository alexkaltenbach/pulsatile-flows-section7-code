# Ordnerstruktur

Die Python-Module bleiben im Hauptordner, weil die vorhandenen Skripte direkte
Imports wie `from DataSym import *` verwenden. Die Uebersicht entsteht deshalb
ueber klare Einstiegspunkte und Dokumentation statt ueber riskante Modulverschiebung.

## Zentrale Dateien

- `run_section_71.py`: Startet die Section-7.1-Faelle fuer `r = 1, 5, 10`.
- `run_section_72.py`: Startet einzelne oder alle Section-7.2-Faelle.
- `README.md`: Kurzanleitung mit Befehlen.
- `DataWomersley.py`: Exakte Loesung fuer Section 7.1.
- `LinearMixedFEM.py`, `PicardIterationLinear.py`: Lineare Section-7.1-
  Diskretisierung und Periodizitaetsiteration.
- `ErrorDecayAnalysisWomersley.py`, `ErrorPlotWomersley.py`: Fehlerauswertung
  und Plot fuer Section 7.1.
- `ErrorDecayAnalysisSym.py`: Generator fuer `Const15`, `Const25`, `Sym1`.
- `ErrorDecayAnalysisNonSym.py`: Generator fuer `NonSym2`.
- `ConvergenceTriangles.py`: Gemeinsame automatische Konvergenzdreiecke fuer
  die Plot-Skripte.
- `pxLaplaceFEM.py`: Zeitschritt- und nichtlineare FEM-Loesung.
- `PicardIteration.py`: Aeussere Periodizitaets-Fixpunktiteration.
- `DataSym.py`, `DataNonSym.py`: Exakte Loesungen und Exponenten.

## Ergebnisordner

- `Womersley1/`, `Womersley5/`, `Womersley10/`: Section-7.1-Faelle fuer
  `r = 1, 5, 10`.
- `Const15/`: Konstanter Fall `p = 3/2`.
- `Const25/`: Konstanter Fall `p = 5/2`.
- `Sym1/`: Gerader stueckweise konstanter Fall.
- `NonSym2/`: Nicht-gerader stueckweise konstanter Fall.

Die aufgefuehrten Ordner sind die reproduzierbaren Ergebnisordner der in
Section 7.1 und Section 7.2 dokumentierten Hauptfaelle.
