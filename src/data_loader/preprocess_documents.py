# Ez a szkript felelős a nyers dokumentumok (RTF, DOCX) és a hozzájuk tartozó
# JSON metaadatok feldolgozásáért, majd egyetlen CSV fájlba történő mentéséért.
import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
import sys
import os
import logging
from striprtf.striprtf import rtf_to_text
import csv
from docx import Document
from bs4 import BeautifulSoup
import html
from typing import Set

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Konfiguráció importálása
try:
    from configs import config
except ImportError as e:
    print(f"HIBA: configs modul import sikertelen: {e}")
    sys.exit(1)

# Loggolás beállítása
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

# Magyar stopword lista
HUNGARIAN_STOPWORDS = {
    # Alapvető stopwordök
    'a', 'az', 'és', 'vagy', 'de', 'hogy', 'mint', 'mely', 'aki', 'ami', 'ez', 'az',
    'ezek', 'azok', 'ezt', 'azt', 'ennek', 'annak', 'erre', 'arra', 'ide', 'oda',
    'itt', 'ott', 'itt', 'ott', 'itt', 'ott', 'itt', 'ott', 'itt', 'ott',
    
    # Névelők és determinánsok
    'egy', 'minden', 'valamennyi', 'néhány', 'sok', 'kevés', 'több', 'kevesebb',
    'annyi', 'ennyi', 'annyi', 'ennyi', 'annyi', 'ennyi', 'annyi', 'ennyi',
    
    # Névmások
    'én', 'te', 'ő', 'mi', 'ti', 'ők', 'magam', 'magad', 'maga', 'magunk', 'magatok', 'maguk',
    'ön', 'önök', 'önmaga', 'önmaguk', 'mindenki', 'valaki', 'senki', 'bárki',
    
    # Elöljárók és kötőszók
    'alatt', 'által', 'belül', 'ellen', 'előtt', 'felett', 'helyett', 'keresztül',
    'kívül', 'mellett', 'mögött', 'nélkül', 'szerint', 'után', 'valamint', 'végül',
    
    # Határozószók
    'itt', 'ott', 'hol', 'mikor', 'hogyan', 'miért', 'milyen', 'mennyi', 'ahol',
    'amikor', 'ahogy', 'amíg', 'míg', 'mialatt', 'mióta', 'mióta', 'mivel',
    
    # Segédigék és módosítószók
    'van', 'vannak', 'volt', 'voltak', 'lesz', 'lesznek', 'lenni', 'lenni',
    'fog', 'fognak', 'tud', 'tudnak', 'akar', 'akarnak', 'kell', 'kellenek',
    'szabad', 'szabadnak', 'muszáj', 'muszájnak', 'lehet', 'lehetnek',
    
    # Gyakori igék
    'csinál', 'csinálnak', 'tesz', 'tennek', 'ad', 'adnak', 'vesz', 'vesznek',
    'jön', 'jönnek', 'megy', 'mennek', 'áll', 'állnak', 'ül', 'ülnek',
    'fekszik', 'fekszenek', 'alszik', 'alszanak', 'eszik', 'esznek',
    
    # Módosítószók és kötőszók
    'is', 'sem', 'csak', 'még', 'már', 'most', 'akkor', 'soha', 'mindig',
    'néha', 'gyakran', 'ritkán', 'hamar', 'későn', 'korán', 'később',
    'előbb', 'utóbb', 'azután', 'azelőtt', 'ekkor', 'akkor', 'mikor',
    
    # Számok és mennyiségek
    'egy', 'kettő', 'három', 'négy', 'öt', 'hat', 'hét', 'nyolc', 'kilenc', 'tíz',
    'száz', 'ezer', 'millió', 'milliárd', 'első', 'második', 'harmadik',
    
    # Időtartamok
    'ma', 'tegnap', 'holnap', 'idén', 'tavaly', 'jövőre', 'hét', 'hónap', 'év',
    'perc', 'óra', 'nap', 'hét', 'hónap', 'év', 'évszázad', 'évezred',
    
    # Helyek és irányok
    'itt', 'ott', 'hol', 'ahol', 'ide', 'oda', 'erre', 'arra', 'fel', 'le',
    'felül', 'alul', 'kint', 'bent', 'külső', 'belső', 'jobb', 'bal',
    
    # Módosítószók és jelzők
    'nagy', 'kicsi', 'hosszú', 'rövid', 'széles', 'keskeny', 'magas', 'alacsony',
    'vastag', 'vékony', 'erős', 'gyenge', 'kemény', 'puha', 'hideg', 'meleg',
    'új', 'régi', 'fiatal', 'öreg', 'szép', 'csúnya', 'jó', 'rossz',
    
    # Gyakori szavak a jogi szövegekben
    'szerint', 'alapján', 'értelmében', 'megfelelően', 'szem előtt', 'figyelembe',
    'véve', 'tekintettel', 'kifolyólag', 'okán', 'miatt', 'folytán', 'eredményeként',
    'következtében', 'alapján', 'szerint', 'megfelelően', 'megfelelő', 'megfelel',
    'megfelelnek', 'megfelelő', 'megfelelően', 'megfelelő', 'megfelelően',
    
    # Üres szavak és gyakori kifejezések
    'szóval', 'vagyis', 'tehát', 'ugyanis', 'mivel', 'mert', 'mert', 'mert',
    'ugyanakkor', 'azonban', 'viszont', 'ellenben', 'ellenkezőleg', 'fordítva',
    'egyébként', 'egyébként', 'egyébként', 'egyébként', 'egyébként',
    
    # Rövidítések és gyakori kifejezések
    'stb', 'st', 'vö', 'lásd', 'ld', 'pl', 'például', 'például', 'például',
    'stb', 'st', 'vö', 'lásd', 'ld', 'pl', 'például', 'például', 'például'
}

def remove_hungarian_stopwords(text: str, stopwords: Set[str] | None = None) -> tuple[str, int]:
    """
    Magyar stopwordök eltávolítása a szövegből.
    
    Args:
        text: A tisztítandó szöveg
        stopwords: Stopword halmaz (alapértelmezett: HUNGARIAN_STOPWORDS)
    
    Returns:
        Tuple: (A stopwordök nélküli szöveg, eltávolított stopwordök száma)
    """
    if not isinstance(text, str) or not text.strip():
        return "", 0
    
    if stopwords is None:
        stopwords = HUNGARIAN_STOPWORDS
    
    # Szöveg szavakra bontása, megtartva a szóközöket és írásjeleket
    words = text.split()
    filtered_words = []
    removed_count = 0
    
    for word in words:
        # Szó tisztítása (írásjelek eltávolítása a hasonlítás előtt)
        clean_word = re.sub(r'[^\wáéíóöőúüűÁÉÍÓÖŐÚÜŰ]', '', word.lower())
        
        # Ha a tisztított szó nincs a stopword listában, megtartjuk
        if clean_word and clean_word not in stopwords:
            filtered_words.append(word)
        elif clean_word:
            removed_count += 1
    
    return ' '.join(filtered_words), removed_count

def clean_text_for_embedding(text: str, remove_stopwords: bool = True) -> tuple[str, int]:
    """
    Szöveg alapos tisztítása embedding generálás előtt.
    Eltávolítja a HTML tageket, speciális karaktereket, URL-eket, normalizálja a whitespace-t,
    és opcionálisan kiszűri a magyar stopwordöket.
    
    Args:
        text: A tisztítandó szöveg
        remove_stopwords: Ha True, eltávolítja a magyar stopwordöket
    
    Returns:
        Tuple: (A tisztított szöveg, eltávolított stopwordök száma)
    """
    if not isinstance(text, str) or not text.strip():
        return "", 0
    
    # HTML entitások dekódolása (pl. &amp; -> &)
    try:
        text = html.unescape(text)
    except Exception:
        pass # Ha hiba történik, a nyers szöveggel megyünk tovább

    # HTML tagek eltávolítása BeautifulSoup-pal
    try:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
    except Exception:
        # Ha a BeautifulSoup hibát dob, egyszerű regex-szel próbáljuk
        text = re.sub(r'<[^>]+>', '', text)
    
    # URL-ek, email címek eltávolítása
    text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text, flags=re.MULTILINE)

    # Null byte és egyéb nem szöveges vezérlő karakterek eltávolítása
    text = text.replace('\x00', '')
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # RTF specifikus maradványok eltávolítása, amik a konverzió után maradhatnak
    text = re.sub(r'\\[a-zA-Z]+\d*\s?', '', text)
    text = re.sub(r'[{}]', '', text)
    
    # Egységes kisbetűre alakítás
    text = text.lower()
    
    # Többszörös szóközök, tabulátorok, új sorok cseréje egyetlen szóközre
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Magyar stopwordök eltávolítása (opcionális)
    removed_count = 0
    if remove_stopwords:
        text, removed_count = remove_hungarian_stopwords(text)
        # Újabb whitespace normalizálás a stopword eltávolítás után
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text, removed_count

# Adat könyvtár elérési útja a konfigurációból
root_dir_to_scan = project_root / 'data'
paths = list(root_dir_to_scan.rglob('*'))

# A feldolgozott rekordokat egyetlen listában gyűjtjük
all_records = []
total_records = 0
stopwords_removed_count = 0  # Statisztika a stopword eltávolításról

# Támogatott szövegfájl kiterjesztések
SUPPORTED_EXTENSIONS = tuple(ext.lower() for ext in config.SUPPORTED_TEXT_EXTENSIONS)

logging.info(f"Feldolgozás kezdése, cél: egyetlen CSV fájl.")
logging.info(f"Találva {len(paths):,} potenciális fájl")
if config.REMOVE_HUNGARIAN_STOPWORDS:
    logging.info("🇭🇺 Magyar stopword szűrés AKTÍV - a stopwordök eltávolításra kerülnek a szövegekből")
else:
    logging.info("🇭🇺 Magyar stopword szűrés INAKTÍV - a stopwordök megmaradnak a szövegekben")

for path in tqdm(paths, desc="Dokumentumfájlok feldolgozása"):
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        base_filename = path.stem
        text_path = path
        json_filename = base_filename + '.RTF_OBH.JSON'
        json_path = path.with_name(json_filename)

        text_content = ""
        if text_path.suffix.lower() == '.rtf':
            try:
                with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                    rtf_content = f.read()
                text_content = rtf_to_text(rtf_content, errors="ignore")
            except Exception as e:
                logging.warning(f"Nem sikerült kinyerni a szöveget az RTF fájlból ({text_path}): {e}")
        elif text_path.suffix.lower() == '.docx':
            try:
                doc = Document(str(text_path))
                text_content = ' \n'.join(para.text for para in doc.paragraphs if para.text.strip())
            except Exception as e:
                logging.warning(f"Nem sikerült kinyerni a szöveget a DOCX fájlból ({text_path}): {e}")
        
        # A kinyert nyers szöveg azonnali tisztítása
        cleaned_text_content, removed_stopwords = clean_text_for_embedding(
            text_content, 
            remove_stopwords=config.REMOVE_HUNGARIAN_STOPWORDS
        )
        stopwords_removed_count += removed_stopwords

        # Csak akkor dolgozzuk fel a rekordot, ha a tisztítás után is maradt értékelhető szöveg
        if len(cleaned_text_content) < config.CLEANING_MIN_TEXT_LENGTH:
            logging.debug(f"Dokumentum átugorva, mert a tisztított szöveg túl rövid: {path.name}")
            continue

        extracted_metadata = {}
        all_related_ugyszam = []
        all_related_birosag = []

        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    metadata_dict = json.load(jf)
                    if 'List' in metadata_dict and isinstance(metadata_dict['List'], list) and len(metadata_dict['List']) > 0:
                        extracted_metadata = metadata_dict['List'][0]
                        if 'KapcsolodoHatarozatok' in extracted_metadata and isinstance(extracted_metadata['KapcsolodoHatarozatok'], list):
                            for related_case in extracted_metadata['KapcsolodoHatarozatok']:
                                if isinstance(related_case, dict):
                                    all_related_ugyszam.append(related_case.get('KapcsolodoUgyszam'))
                                    all_related_birosag.append(related_case.get('KapcsolodoBirosag'))
                                else:
                                    logging.warning(f"A KapcsolodoHatarozatok lista egyik eleme nem szótár a {json_path} fájlban.")
                                    all_related_ugyszam.append(None)
                                    all_related_birosag.append(None)
                        if 'Jogszabalyhelyek' in extracted_metadata and not isinstance(extracted_metadata['Jogszabalyhelyek'], (str, int, float, bool)):
                            extracted_metadata['Jogszabalyhelyek'] = json.dumps(extracted_metadata['Jogszabalyhelyek'], ensure_ascii=False)
                        if 'KapcsolodoHatarozatok' in extracted_metadata and not isinstance(extracted_metadata['KapcsolodoHatarozatok'], (str, int, float, bool)):
                            extracted_metadata['KapcsolodoHatarozatok'] = json.dumps(extracted_metadata['KapcsolodoHatarozatok'], ensure_ascii=False)
            except json.JSONDecodeError:
                logging.warning(f"Nem sikerült dekódolni a JSON fájlt: {json_path}")
            except Exception as e:
                logging.warning(f"Hiba a JSON fájl feldolgozása közben ({json_path}): {e}")

        birosag_from_path = None
        try:
            abs_root_dir = root_dir_to_scan.resolve()
            abs_path = path.resolve()
            if abs_path.is_relative_to(abs_root_dir):
                 rel_parts = abs_path.relative_to(abs_root_dir).parts
                 if len(rel_parts) > 1:
                    birosag_from_path = rel_parts[0]
        except Exception as e_path:
             logging.warning(f"Váratlan hiba a bíróság nevének útvonalból történő kinyerése közben ({path}): {e_path}")

        record = {
            'text': cleaned_text_content,
            **extracted_metadata,
            'AllKapcsolodoUgyszam': json.dumps(all_related_ugyszam, ensure_ascii=False) if all_related_ugyszam else None,
            'AllKapcsolodoBirosag': json.dumps(all_related_birosag, ensure_ascii=False) if all_related_birosag else None,
        }
        record['doc_id'] = extracted_metadata.get('Azonosito', base_filename)
        record['birosag'] = extracted_metadata.get('MeghozoBirosag', birosag_from_path)
        
        record.pop('Szoveg', None)
        record.pop('RezumeSzovegKornyezet', None)
        record.pop('DownloadLink', None)
        record.pop('metadata', None)

        all_records.append(record)
        total_records += 1

# ===== EGYESÍTETT, TISZTÍTOTT PARQUET LÉTREHOZÁSA ÉS MENTÉSE =====
logging.info("Feldolgozás befejezve, egységes DataFrame létrehozása...")

if all_records:
    try:
        df = pd.DataFrame(all_records)
        
        # A 'birosag' oszlop feltöltése, ha hiányzik (fontos a konzisztenciához)
        df['birosag'] = df['birosag'].fillna('ISMERETLEN')

        # Oszlopok sorrendjének biztosítása a jobb átláthatóságért
        expected_cols = [
            'doc_id', 'text', 'birosag', 'JogTerulet', 'Azonosito', 'MeghozoBirosag',
            'EgyediAzonosito', 'HatarozatEve', 'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag',
            'KapcsolodoHatarozatok', 'Jogszabalyhelyek'
        ]
        
        final_cols = [col for col in expected_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in final_cols]
        df = df[final_cols + other_cols]

        # A kimeneti útvonal most a Parquet fájlra mutat
        out_path = config.CLEANED_PARQUET_DATA_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Mentés egyetlen, tömörített Parquet fájlba
        df.to_parquet(
            path=out_path,
            engine='pyarrow',
            compression='snappy',
            index=False,
        )
        
        logging.info(f"Tisztított Parquet fájl sikeresen mentve: {out_path} ({len(df):,} sor)")

    except Exception as e:
        logging.error(f"Hiba a Parquet fájl létrehozásában vagy mentésében: {e}", exc_info=True)

# ===== VÉGSŐ ÜZENETEK =====
print(f"\n✅ PREPROCESSING BEFEJEZVE!")
print(f"📊 Feldolgozott rekordok: {total_records:,}")
if config.REMOVE_HUNGARIAN_STOPWORDS:
    print(f"🇭🇺 Eltávolított magyar stopwordök: {stopwords_removed_count:,}")
print(f"📄 Kimeneti fájl: {config.CLEANED_PARQUET_DATA_PATH}")