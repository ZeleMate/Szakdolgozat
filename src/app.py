"""
Streamlit frontend for the Semantic Search + Reinforcement Learning project.
"""

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Existing imports
import streamlit as st
from src.models.embedding import load_embedding_model
from src.search.semantic_search import SemanticSearch, GraphSemanticSearch
from src.rl.environment import LegalSearchEnv, LegalSearchRLHF
import numpy as np
import time

def load_demo_documents():
    """Load demo documents. Can be replaced with actual data loading."""
    return [
        "A munkáltató rendkívüli felmondással megszüntette a munkaviszonyt.",
        "A munkavállaló bírósághoz fordult az elbocsátása miatt.",
        "A bíróság elutasította a keresetet, mert az eljárás jogszerű volt.",
        "A munkáltató nem tartotta be a felmondási időt.",
        "A per tárgya a végkielégítés mértéke volt."
    ]

def main():
    st.set_page_config(
        page_title="Jogi dokumentum kereső",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("Jogi dokumentum kereső")
    st.subheader("Szemantikus keresés + Megerősítéses tanulás")
    
    # Sidebar for configuration
    st.sidebar.header("Beállítások")
    model_name = st.sidebar.selectbox(
        "Embedding modell",
        ["MODERNBert",
         "distiluse-base-multilingual-cased-v1"]
    )
    top_k = st.sidebar.slider("Találatok száma", 1, 10, 5)
    
    # Load model and documents (with caching for performance)
    @st.cache_resource
    def load_model_and_search(model_name):
        with st.spinner("Modell betöltése..."):
            model = load_embedding_model(model_name)
            documents = load_demo_documents()
            semantic_search = SemanticSearch(model, documents)
            env = LegalSearchEnv(model, documents, target_doc_idx=1)
            return model, documents, semantic_search, env

    model, documents, semantic_search, env = load_model_and_search(model_name)
    
    # Main content - Search
    st.header("Dokumentum keresés")
    query = st.text_input("Keresési kifejezés", "Jogellenes felmondás munkahelyen")
    
    if st.button("Keresés") or query:
        with st.spinner("Keresés folyamatban..."):
            # Run semantic search
            results = semantic_search.search(query, top_k)
            
            # Display results
            st.subheader("Keresési eredmények")
            for i, (idx, doc, score) in enumerate(results, 1):
                st.markdown(f"**{i}. Dokumentum** (relevancia: {score:.4f})")
                st.info(doc)
    
    # RL Demo section
    st.header("Megerősítéses tanulás demonstráció")
    if st.button("Futtatás"):
        with st.spinner("RL ágens futtatása..."):
            # Reset environment with the current query
            env.query_text = query
            env.query = env.model.encode([query])[0]
            obs = env.reset()
            
            # Sample a document randomly (simulating agent behavior)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Kiválasztott dokumentum")
                st.info(documents[action])
            
            with col2:
                st.subheader("RL visszajelzés")
                if reward > 0:
                    st.success(f"Jutalom: {reward} (Megfelelő dokumentum!)")
                else:
                    st.error(f"Jutalom: {reward} (Nem optimális választás)")
                
                st.write("Cél dokumentum:")
                st.info(documents[env.target_doc_idx])
    
    # Additional info
    st.markdown("---")
    st.write("A projektről: Szemantikus keresés és megerősítéses tanulás jogi dokumentumokhoz")

if __name__ == "__main__":
    main()