# pLaplaceMixed - Sections 7.1 and 7.2

Dieser Ordner enthaelt die FEniCS-Implementierung fuer die numerischen
Experimente aus Sections 7.1 und 7.2 des Artikels.

## Voraussetzungen

Du brauchst eine Python-Umgebung mit FEniCS/DOLFIN sowie `numpy`, `scipy` und
`matplotlib`. Ein normaler `python3` ohne FEniCS reicht nicht aus.

Typischer Start:

```bash
cd ~/Documents/Fenics/pLaplaceMixed
```

## Section 7.1

| Paper-Fall | Befehl | Ergebnisordner |
| --- | --- | --- |
| Womersley/Hagen-Poiseuille, `r = 1` | `python3 run_section_71.py --radius 1 --nmax 12` | `Womersley1/` |
| Womersley/Hagen-Poiseuille, `r = 5` | `python3 run_section_71.py --radius 5 --nmax 12` | `Womersley5/` |
| Womersley/Hagen-Poiseuille, `r = 10` | `python3 run_section_71.py --radius 10 --nmax 12` | `Womersley10/` |

Alle drei Section-7.1-Faelle nacheinander:

```bash
python3 run_section_71.py --radius all --nmax 12
```

Hinweis: `nmax=12` bedeutet, dass die Schleife `n=1,...,11` laeuft. Das ist
die Einstellung aus Section 7.1.

## Section 7.2

| Paper-Fall | Befehl | Ergebnisordner |
| --- | --- | --- |
| Constant case, `p = 3/2`, Formel (7.3) | `python3 run_section_72.py --case const15 --nmax 10` | `Const15/` |
| Constant case, `p = 5/2`, Formel (7.3) | `python3 run_section_72.py --case const25 --nmax 10` | `Const25/` |
| Even case, Formel (7.4) | `python3 run_section_72.py --case even --nmax 10` | `Sym1/` |
| Non-even case, Formel (7.5) | `python3 run_section_72.py --case noneven --nmax 10` | `NonSym2/` |

Alle vier Faelle nacheinander:

```bash
python3 run_section_72.py --case all --nmax 10
```

Schneller Testlauf mit grober Verfeinerung:

```bash
python3 run_section_72.py --case even --nmax 5
```

Hinweis: `nmax=10` bedeutet, dass die Schleife `n=1,...,9` laeuft. Das ist die
Einstellung aus Section 7.2.

## Plots

Wenn du nach dem Rechnen direkt den passenden Plot starten willst:

```bash
python3 run_section_71.py --radius all --nmax 12 --plot
python3 run_section_72.py --case const15 --nmax 10 --plot
python3 run_section_72.py --case const25 --nmax 10 --plot
python3 run_section_72.py --case even --nmax 10 --plot
python3 run_section_72.py --case noneven --nmax 10 --plot
```

Die Plot-Skripte koennen auch direkt ausgefuehrt werden:

```bash
python3 ErrorPlotWomersley.py --radius 1
python3 ErrorPlotWomersley.py --radius 5
python3 ErrorPlotWomersley.py --radius 10
python3 ErrorPlotConst15.py
python3 ErrorPlotConst25.py
python3 ErrorPlotSym.py
python3 ErrorPlotNonSym.py
```

Die Paper-Figuren 5--9 koennen gesammelt gespeichert werden:

```bash
python3 paper_figures.py --figure all --save-dir plots --format pdf png --no-show
```

## Ergebnisdateien

Jeder Lauf schreibt in den passenden Ergebnisordner:

- `errors.pkl`: Fehlerwerte, EOC-Werte, Picard-Iterationszahlen
- `v1.xml`, ..., `v9.xml`: finale diskrete Geschwindigkeiten
- `Gamma1.xml`, ..., `Gamma9.xml`: finale diskrete Lagrange-Multiplikatoren

Diese numerischen Ergebnisordner sind reproduzierbare Ausgaben der Skripte und
werden nicht als Rohdaten im Archiv mitgefuehrt. Die Figures in `plots/` wurden
aus den zugehoerigen Laeufen erstellt.

## Implementierungsnotizen

- `Gamma` wird ueber echte Zeitintervalle ausgewertet.
- Der `Gamma`-Fehler entspricht der L1-Groesse aus (7.6), ohne zusaetzliche Quadratwurzel.
- Die ausgegebene `Gamma_EOC` bezieht sich auf die natuerliche geplottete Groesse
  `sqrt(Gamma)`; der EOC des rohen Modulars wird separat als `Gamma_Modular_EOC`
  gespeichert.
- Die `H1/f`- und `Gamma`-Fehler werden ueber echte Zeitintervalle ausgewertet.
- Plot-Skripte veraendern keine gespeicherten Fehlerdaten.
- Konvergenzdreiecke werden in den Plot-Skripten automatisch aus den letzten
  Verfeinerungsstufen erzeugt; Kurven mit gleicher gerundeter Rate teilen sich
  ein zweifarbig schraffiertes Dreieck.
- Section 7.1 verwendet einen eigenen linearen Solver.
- In Section 7.1 ist `alpha` als Querschnittsmittel gespeichert; die schwache
  rechte Seite integriert `alpha` ueber `Sigma` und erfuellt damit die totale
  Flussbedingung des exakten Profils. Der gespeicherte `Gamma`-Fehler verwendet
  das Vorzeichen des Lagrange-Multiplikators in der implementierten schwachen
  Form.
- `run_section_72.py` ist der zentrale Einstiegspunkt fuer alle Section-7.2-Faelle.
- Der Picard-Maximalwert ist `Kmax = 100` wie im Artikel; der innere
  Residual-Log ist standardmaessig aus, damit Terminalausgaben sauber bleiben.
- Die p-Laplace-Nichtlinearitaet verwendet weiterhin die kleine FEniCS-
  Regularisierung `DOLFIN_EPS`, damit die Terme fuer `p < 2` an Stellen mit
  verschwindendem Gradienten numerisch wohldefiniert bleiben.
