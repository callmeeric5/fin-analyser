# run_app.sh
#!/bin/bash
uv sync
uv run --env-file=.env streamlit run app/app.py
