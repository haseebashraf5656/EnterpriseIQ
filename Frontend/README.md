# EnterpriseIQ frontend

Streamlit UI for the document Q&A API. `BACKEND_URL` defaults to `http://localhost:8000`.

## Docker Compose

From the repository root:

```bash
docker compose up --build
```

Open http://localhost:8501. Compose sets `BACKEND_URL=http://backend:8000`.

## Local

```bash
cd Frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```
