# Szemantikus keresés + Megerősítéses tanulás projekt

Ez a projekt szemantikus keresési képességeket kombinál megerősítéses tanulással a jogi dokumentumok keresési eredményeinek javítása érdekében.

## Telepítés

1. Virtuális környezet létrehozása:
```bash
python -m venv venv
```

2. Virtuális környezet aktiválása:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Függőségek telepítése:
```bash
pip install -r requirements.txt
```

## Projekt felépítése

- `src/`: Forráskódok
  - `models/`: Gépi tanulási modellek
  - `search/`: Szemantikus keresés implementációja
  - `rl/`: Megerősítéses tanulás implementációja
- `data/`: Adatfájlok könyvtára
  - `raw/`: Nyers adatok
  - `processed/`: Feldolgozott adatok
- `notebooks/`: Jupyter notebookok kísérletekhez
- `tests/`: Egységtesztek
- `configs/`: Konfigurációs fájlok
- `requirements.txt`: Python függőségek
- `setup.sh`: Telepítő script

## Használat

Futtassa a fő szkriptet a szemantikus keresés és az RL környezet teszteléséhez:

```bash
python src/main.py
```

Paraméterek:
```bash
python src/main.py --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" --top_k 5 --query "Jogellenes felmondás munkahelyen"
```

## Valós adatok hozzáadása

Cserélje le a példa adatokat a `src/main.py` fájlban valós jogi dokumentumokkal a `documents` lista módosításával, vagy készítsen betöltő függvényt a `data/raw` könyvtárból.