# Multi-Modal Document QA Engine (ColPali)

This repository contains the full implementation for **DSAI 413 - Assignment 1**.

## Overview
It implements an advanced **Vision-guided Retrieval-Augmented Generation (RAG)** pipeline. Instead of relying on brittle text extraction libraries (like PyMuPDF) capable of losing table layouts, it leverages ColPali. ColPali represents visually-rich document pages as image patches embedded via a contrastive late-interaction model, allowing exact page retrieval without data loss. The generation phase leverages `Qwen2-VL`, a native Vision-Language Model that directly answers from the visual context.

## Files
- `app.py`: Streamlit User Interface.
- `backend.py`: Core implementation of the ColPali retrieval index and Qwen-VL generation.
- `download_data.py`: Helper script to fetch complex PDF datasets (e.g., ColPali arXiv paper with diagrams).
- `report.md`: 2-page technical architecture summary.
- `requirements.txt`: Python package constraints.

## How to Run

1. Initialize Environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download sample dataset (ColPali PDF):
   ```bash
   python3 download_data.py
   ```
3. Run the UI:
   ```bash
   streamlit run app.py
   ```

*Note: Initializing the UI takes a moment as it downloads/loads the ColPali Multi-vector retrieval model and Qwen-VL Generation model into memory (~10-12GB VRAM needed or runs mapped on CPU/MPS).*
