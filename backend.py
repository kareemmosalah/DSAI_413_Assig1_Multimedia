import sys
import os

# Set dummy environment variable before Streamlit imports to prevent xcode warnings
os.environ["NUMEXPR_MAX_THREADS"] = "4"

import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
import pdf2image

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

class RAGSystem:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16 # Safest precision for T4 GPU
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32
        else:
            self.device = "cpu"
            self.dtype = torch.bfloat16
            
        print(f"Using device: {self.device}")
        
        # 1. Initialize ColPali for Retrieval
        self.retriever_name = "vidore/colpali-v1.2"
        print(f"Loading retriever: {self.retriever_name}")
        self.retriever_model = ColPali.from_pretrained(
            self.retriever_name,
            torch_dtype=self.dtype,
            device_map=self.device
        ).eval()
        self.retriever_processor = ColPaliProcessor.from_pretrained(self.retriever_name)
        
        # 2. Initialize VLM for Generation
        self.generator_name = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"Loading generator: {self.generator_name}")
        
        self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.generator_name,
            torch_dtype=self.dtype,
            device_map=self.device
        ).eval()
        
        self.vlm_processor = AutoProcessor.from_pretrained(self.generator_name)
        
        self.document_embeddings = []
        self.document_images = []
        
    def ingest_pdf(self, pdf_path):
        print("Parsing PDF to images...")
        self.document_images = pdf2image.convert_from_path(pdf_path)
        
        self.document_embeddings = []
        batch_size = 2
        print(f"Embedding {len(self.document_images)} pages...")
        for i in range(0, len(self.document_images), batch_size):
            batch = self.document_images[i:i+batch_size]
            inputs = self.retriever_processor.process_images(batch).to(self.retriever_model.device)
            with torch.no_grad():
                embeddings = self.retriever_model(**inputs)
            
            for emb in list(torch.unbind(embeddings.to("cpu"))):
                 self.document_embeddings.append(emb)
        print("Ingestion Done.")

    def query_pipeline(self, text, top_k=2):
        if not self.document_embeddings:
            return "No document ingested.", None
            
        print("Running ColPali retrieval...")
        inputs = self.retriever_processor.process_queries([text]).to(self.retriever_model.device)
        with torch.no_grad():
            query_embedding = self.retriever_model(**inputs).to("cpu")
            
        scores = self.retriever_processor.score_multi_vector(query_embedding, torch.stack(self.document_embeddings))[0]
        
        top_k = min(top_k, len(scores))
        top_indices = scores.topk(top_k).indices.tolist()
        
        retrieved_images = []
        page_numbers = []
        for idx in top_indices:
            img = self.document_images[idx].copy()
            # Downscale resolution to heavily save GPU VRAM for the Generator
            img.thumbnail((768, 768), Image.Resampling.LANCZOS)
            retrieved_images.append(img)
            page_numbers.append(idx + 1)
            
        print(f"Retrieved pages: {page_numbers}")
        print("Generating answer with Qwen-VL...")
        
        # Build prompt for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        # Add all retrieved images as context
        for _ in range(len(retrieved_images)):
             messages[0]["content"].append({"type": "image"})
        
        messages[0]["content"].append({
            "type": "text", 
            "text": f"Based on the provided document pages, answer the user's question. Add citations (e.g. Page X) where appropriate.\nQuestion: {text}"
        })
        
        # Prepare inputs
        text_prompt = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.vlm_processor(
            text=[text_prompt],
            images=retrieved_images,
            padding=True,
            return_tensors="pt"
        ).to(self.vlm_model.device)
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
        
        with torch.no_grad():
            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=256)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        final_answer = f"{output_text}\n\n*Sources: Pages {', '.join(map(str, page_numbers))}*"
        return final_answer, retrieved_images[0]
