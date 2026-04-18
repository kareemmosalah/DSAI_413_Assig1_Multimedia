# Execution & Architecture Guide

## 1. What Models Does This Code Use?
Your code **DOES NOT** use Gemini. It is fully compliant with the assignment requirements by relying purely on open-source, local AI models.
- **Retrieval Engine:** `vidore/colpali-v1.2` (**ColPali**) - Used to view PDF pages natively as images and generate a multi-vector index.
- **QA Generation Engine:** `Qwen/Qwen2-VL-2B-Instruct` - Used to read the retrieved images alongside your natural language query to generate the final citation-backed answer.

*(Note: We removed the Gemini lightweight approach to guarantee you do not lose points, as your assignment explicitly asked to use ColPali).*

---

## 2. Expected Execution Times

### Environment A: Google Colab (Recommended)
Google Colab provides a free **T4 GPU** which is highly optimized for running ColPali and Qwen2-VL.
- **Initial Setup (Downloading Models):** ~1 to 2 minutes
- **Ingesting the PDF (20 pages):** ~30 to 60 seconds
- **Answering a Query:** ~10 to 15 seconds

### Environment B: Local MacBook
Running AI Vision models locally pushes the Mac Unified Memory (RAM). If you have an M1/M2/M3 chip with **16GB+ RAM**, it will work. If you have 8GB RAM, your machine will freeze and you must use Google Colab.
- **Initial Setup (Downloading ~10GB Models):** ~3 to 5 minutes (Depends on your Wi-Fi)
- **Ingesting the PDF (20 pages):** ~3 to 4 minutes (Slower on CPU/MPS)
- **Answering a Query:** ~30 to 60 seconds

---

## 3. How to Run

### Method 1: Running on Google Colab (For the 5-min Video Demo)
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File -> New Notebook**.
3. In the menu, click **Runtime -> Change runtime type** and select **T4 GPU**.
4. Paste the following block of code into the first cell and press the Play button:

```python
# Clone your Github repository
!git clone https://github.com/kareemmosalah/DSAI_413_Assig1_Multimedia.git
%cd DSAI_413_Assig1_Multimedia

# Install Required Packages
!pip install -r requirements.txt
!npm install localtunnel

# Launch Streamlit securely via LocalTunnel
import urllib
print("Password for LocalTunnel:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))
!streamlit run app.py & npx localtunnel --port 8501
```

5. Click the `localtunnel` link that is generated. It will ask for an endpoint password—paste the password printed by the cell above. You are now inside your App!

### Method 2: Running Locally on MacBook
If you have a powerful Mac and plenty of internet bandwidth:
1. Open your terminal inside the `Assig1_multimedia` folder.
2. Run these commands:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 download_data.py
streamlit run app.py
```
3. A browser tab will open automatically. Wait for the models to finish downloading in the background.
