# Supplier Normalization & Duplicate Detection Workbench

A procurement data-quality tool that standardizes messy supplier names, detects duplicate vendor records, recommends supplier families, flags questionable matches for human review, and exports cleaner supplier data for spend analytics.

## Features

- Supplier name cleaning
- Known alias matching
- RapidFuzz fuzzy matching
- Match confidence scoring
- Human review queue
- Golden record recommendations
- Before/after supplier count impact
- Exportable normalized supplier file

## Tech Stack

- Python
- Streamlit
- Pandas
- RapidFuzz
- OpenPyXL

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py