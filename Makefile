.PHONY: train api ui test docker clean

VENV = source .venv/bin/activate &&

## ─── Training ────────────────────────────────────────────────────────────────
train:
	@echo "▶ Running full training pipeline …"
	$(VENV) python -m src.train
	@echo "✅ Training complete. Artifacts in model/ and reports/figures/"

## ─── API ─────────────────────────────────────────────────────────────────────
api:
	@echo "▶ Starting FastAPI on http://localhost:8000 …"
	$(VENV) uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

## ─── Streamlit UI ────────────────────────────────────────────────────────────
ui:
	@echo "▶ Starting Streamlit UI on http://localhost:8501 …"
	$(VENV) streamlit run app/app.py

## ─── Tests ───────────────────────────────────────────────────────────────────
test:
	@echo "▶ Running pytest …"
	$(VENV) pytest tests/ -v --tb=short
	@echo "✅ All tests passed"

## ─── MLflow UI ───────────────────────────────────────────────────────────────
mlflow:
	@echo "▶ Starting MLflow UI on http://localhost:5000 …"
	$(VENV) mlflow ui --port 5000

## ─── Docker ──────────────────────────────────────────────────────────────────
docker:
	@echo "▶ Building and starting Docker services …"
	docker compose up --build

docker-stop:
	docker compose down

## ─── Setup (first-time) ──────────────────────────────────────────────────────
setup:
	@echo "▶ Creating virtual environment …"
	python3 -m venv .venv
	$(VENV) pip install --upgrade pip
	$(VENV) pip install -r requirements.txt
	@echo "✅ Setup complete. Run: make train"

## ─── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true

## ─── Full rebuild from scratch ───────────────────────────────────────────────
rebuild: clean train test
	@echo "✅ Full rebuild complete"
