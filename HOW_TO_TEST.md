# How To Test & Record Your Video Demo

This guide provides exactly what you need to do to test your application and produce the 2-5 minute video demonstration required by your assignment rubric.

## Step 1: Open the App in Google Colab (Recommended)
By testing in Colab, you completely avoid using your local Mac's memory.
1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Go to `Runtime` -> `Change runtime type` -> select **T4 GPU**.
3. Paste and run this exact block of code:
```python
!git clone https://github.com/kareemmosalah/DSAI_413_Assig1_Multimedia.git
%cd DSAI_413_Assig1_Multimedia
!pip install -r requirements.txt
!npm install localtunnel
import urllib
print("Password for LocalTunnel:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))
!streamlit run app.py & npx localtunnel --port 8501
```
4. Click the `localtunnel` link that is produced, enter the password printed by the code, and your Streamlit web interface will appear!

---

## Step 2: Ingest the Dataset
1. Start recording your screen at this point (using QuickTime on Mac).
2. On the left sidebar of the Streamlit app, notice that `data/colpali_paper.pdf` is already typed in. 
3. Click the **"Ingest Document"** button.
   - *Explain in your video that this step is converting the complex PDF pages strictly into Image form to map visual metadata natively into a multi-vector space, perfectly satisfying the "Document Ingestion Pipeline" grading requirement.*

---

## Step 3: Test with Prompts
To get the full grade, you must prove the system can handle **Charts, Tables, and Text**. Once ingestion says it's done, paste these exact prompts one by one into the "Ask a question" box and click "Generate Answer".

### Prompt 1: Testing Architecture / Diagrams
*"Based on the diagrams in the paper, describe the complete ColPali multi-vector retrieval mechanism from input to generating the query-document matching similarity scores."*
- **What to say in video:** Highlight that the system surfaces the correct diagram page as a source and correctly answers the question purely visually.

### Prompt 2: Testing Tables & Dense Data
*"According to Table 2, what is the exact average score (Avg) of the ColPali model compared to BGE-M3 when evaluating the different ViDoRe benchmarks?"*
- **What to say in video:** Emphasize that traditional text-parsing RAGs completely destroy table columns, but because ColPali handles it visually, it gets the exact statistical number correctly.

### Prompt 3: Testing Text Evaluation
*"What is the difference drawn between 'text-only' Late Interaction models and 'vision-based' Late Interaction models according to the abstract?"*
- **What to say in video:** Show the source page rendered alongside the final output and point out the citation numbers logic. 

**Stop your recording and upload it! You are done!**
