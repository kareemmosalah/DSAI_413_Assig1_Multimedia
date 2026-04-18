import streamlit as st
import os
from backend import RAGSystem

st.set_page_config(page_title="Multi-Modal Document QA", layout="wide")

@st.cache_resource
def load_system():
    return RAGSystem()

def main():
    st.title("Multi-Modal RAG with ColPali & Qwen2-VL")
    st.markdown("Query dense PDFs (with images, charts, and tables) directly using Vision-Language Models.")
    
    with st.spinner("Loading AI Models (~12GB memory required, may take a few minutes)..."):
        system = load_system()
        
    st.sidebar.header("Document Setup")
    # By default, use the ColPali paper or another downloaded PDF
    doc_path = st.sidebar.text_input("Dataset PDF path", "data/colpali_paper.pdf")
    
    if st.sidebar.button("Ingest Document"):
        if os.path.exists(doc_path):
            with st.spinner("Converting & Indexing Pages..."):
                system.ingest_pdf(doc_path)
            st.sidebar.success(f"Indexed {len(system.document_images)} pages!")
        else:
            st.sidebar.error("File not found.")
            
    st.header("Query Panel")
    query = st.text_input("Ask a question about the document:")
    
    if st.button("Generate Answer") and query:
        with st.spinner("Retrieving and Generating..."):
            try:
                answer, img = system.query_pipeline(query, top_k=1) # using top_k=1 to limit context size
                
                # Setup columns for Answer vs Source
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Answer:")
                    st.write(answer)
                
                with col2:
                    if img:
                        st.subheader("Source Page:")
                        st.image(img, use_container_width=True)
            except Exception as e:
                import traceback
                st.error(f"Error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
