# IT-ticket Classificatie met DistilBERT en LIME

**Auteur:** Ertugrul Demirci  
**Opleiding:** Business IT & Management, Avans Hogeschool  
**Keuzemodule:** Slim beslissen met AI (P3, High-Tech AI route)

---

## Wat doet dit project?

IT-servicedesks verwerken dagelijks tientallen tot honderden meldingen in vrije tekst. Elk ticket handmatig lezen en in de juiste categorie plaatsen kost tijd en leidt regelmatig tot fouten. Dit project bouwt een AI-model dat die categorie automatisch voorstelt op basis van de tekst van het ticket.

Het model geeft een suggestie. De medewerker bevestigt of corrigeert. Via LIME wordt elke voorspelling uitgelegd: welke woorden gaven de doorslag?

**Technieken:** NLP, DistilBERT (fine-tuned), LIME (Explainable AI), Gradio

---

## De bijbehorende e-learning

Er is een interactieve e-learning gemaakt die uitlegt hoe dit systeem werkt, gericht op IT-medewerkers en teamleads zonder AI-achtergrond. Onderdeel 7 van die e-learning loopt stap voor stap door dit notebook.

**E-learning:** [kegdemirci98.github.io/Kennisproduct-SBMAI](https://kegdemirci98.github.io/Kennisproduct-SBMAI/)

---

## Vereisten

### Python-versie

Gebruik **Python 3.12.x** (aanbevolen: 3.12.9).

Python 3.13 of hoger werkt niet goed samen met de CUDA-versie van PyTorch die nodig is voor GPU-training. Controleer je versie met:

```
python --version
```

### Hardware

Een NVIDIA GPU met CUDA-ondersteuning is sterk aanbevolen. Training met een RTX 3070 duurt circa 8 minuten. Zonder GPU draait het model op de CPU, maar dan duurt training meerdere uren.

---

## Installatie

### Stap 1: Repository clonen

```bash
git clone https://github.com/kegdemirci98/Kennisproduct-SBMAI.git
cd Kennisproduct-SBMAI
```

### Stap 2: Afhankelijkheden installeren

```bash
pip install kagglehub pandas matplotlib seaborn scikit-learn lime transformers gradio
```

PyTorch met GPU-ondersteuning (aanbevolen):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

PyTorch zonder GPU (CPU-only):

```bash
pip install torch torchvision torchaudio
```

### Stap 3: Kaggle API instellen

Het notebook downloadt de dataset automatisch via de Kaggle API. Hiervoor heb je een Kaggle-account nodig en een API-sleutel.

1. Ga naar [kaggle.com](https://www.kaggle.com) en log in
2. Ga naar je profiel → Settings → API → Create New Token
3. Download het bestand `kaggle.json`
4. Plaats het in de juiste map:
   - Windows: `C:\Users\[jouwgebruikersnaam]\.kaggle\kaggle.json`
   - Mac of Linux: `~/.kaggle/kaggle.json`

---

## Notebook draaien

Open het notebook in Jupyter:

```bash
jupyter notebook IT-servicedesktickets.ipynb
```

Draai alle cellen van boven naar beneden via Kernel → Restart & Run All.

Het notebook doorloopt de volgende stappen:

1. Dataset downloaden en verkennen
2. Data voorbereiden en splitsen (70/15/15)
3. Baseline model bouwen: TF-IDF + Logistic Regression
4. DistilBERT instellen en fine-tunen
5. Modellen vergelijken en evalueren
6. LIME uitlegbaarheid toevoegen
7. Gradio-demo starten

De Gradio-interface opent automatisch in de browser op `http://127.0.0.1:7860`.

### Opmerking over resultaatvariatie

Elke keer dat je het notebook opnieuw draait, kunnen de exacte percentages en grafieken licht afwijken. Dit komt door willekeurige elementen in het trainingsproces, zoals de volgorde waarin het model de trainingsdata verwerkt. De resultaten die in de e-learning staan komen uit een specifieke run. Jouw run kan iets anders uitpakken, maar de verhoudingen blijven hetzelfde: DistilBERT presteert consequent beter dan de baseline.

---

## Dataset

**Naam:** IT Service Ticket Classification Dataset  
**Bron:** [kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)  
**Omvang:** 47.837 tickets, 8 categorieën, Engelstalig

| Categorie | Aantal | Aandeel |
|---|---|---|
| Hardware | 13.617 | 28.5% |
| HR Support | 10.915 | 22.8% |
| Access | 7.125 | 14.9% |
| Miscellaneous | 7.060 | 14.8% |
| Storage | 2.777 | 5.8% |
| Purchase | 2.464 | 5.2% |
| Internal Project | 2.119 | 4.4% |
| Administrative rights | 1.760 | 3.7% |

---

## Resultaten

| Model | Accuracy (testset) |
|---|---|
| TF-IDF + Logistic Regression (Baseline) | ~85.9% |
| DistilBERT (fine-tuned) | ~87.8% |

Exacte waarden variëren licht per run. Het verschil van circa 1.9 procentpunt vertaalt zich bij 200 tickets per dag naar gemiddeld 4 minder verkeerd gesorteerde tickets per dag, ofwel bijna 1.000 per jaar.

### F1-score per categorie (DistilBERT)

| Categorie | F1-score |
|---|---|
| Purchase | 0.92 |
| Access | 0.92 |
| Storage | 0.90 |
| HR Support | 0.89 |
| Internal Project | 0.88 |
| Hardware | 0.87 |
| Miscellaneous | 0.84 |
| Administrative rights | 0.79 |

Administrative rights scoort het laagst door overlap met Access-tickets: beide gaan over rechten en toegang.

---

## Modelparameters

| Parameter | Waarde |
|---|---|
| Basismodel | distilbert-base-uncased |
| Max tokenlengte | 128 |
| Batchgrootte | 32 |
| Epochs | 3 |
| Leersnelheid | 2e-5 |
| Optimizer | AdamW |
| Datasplit | 70/15/15 |
| Klasse-gewichten | Ja (class_weight balanced) |

---

## Bestanden

| Bestand | Beschrijving |
|---|---|
| `index.html` | Interactieve e-learning (7 onderdelen) |
| `IT-servicedesktickets.ipynb` | Het volledige notebook |
| `README.md` | Dit bestand |
| `output.png` | Categorie-verdeling (gegenereerd door notebook) |
| `2.png` | Tekstlengte-histogram |
| `3.png` | Confusion matrix baseline |
| `4.png` | Trainingsgrafieken DistilBERT |
| `5.png` | Confusion matrix DistilBERT |
| `6.png` | LIME-visualisatie |