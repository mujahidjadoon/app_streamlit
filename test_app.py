import streamlit as st

def test_streamlit_import():
    """Test that Streamlit can be imported."""
    assert 'streamlit' in globals()

def test_page_title():
    """Test that the page title is set correctly."""
    assert st.get_page_config().page_title == "Assignment 1: RAG Q&A System"
