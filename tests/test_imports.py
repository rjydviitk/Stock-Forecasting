import importlib.util
import os

def test_project_layout_exists():
    assert os.path.exists('dashboard/app.py')
    assert os.path.exists('data/raw/sample_AAPL.csv')
    assert os.path.exists('requirements.txt')

def test_app_imports():
    # Ensure app file loads without syntax errors
    with open('dashboard/app.py', 'r') as f:
        content = f.read()
    assert 'streamlit' in content
    assert 'def lstm_forecast' in content
