.PHONY: help install data train serve app test lint docker clean

PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

help:
	@echo "make install  安装依赖（含开发依赖）"
	@echo "make data     下载并缓存 Ames Housing 数据集"
	@echo "make train    训练全部候选模型并注册新版本"
	@echo "make serve    启动 FastAPI 服务（http://127.0.0.1:8000/docs）"
	@echo "make app      启动 Streamlit 演示页（http://127.0.0.1:8501）"
	@echo "make test     运行 pytest"
	@echo "make lint     ruff 静态检查"
	@echo "make docker   构建并启动容器"

install:
	python -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -c "from houseprice.data.loader import fetch_ames; df = fetch_ames(); print(df.shape)"

train:
	$(PY) -m houseprice.models.train

serve:
	$(PY) -m uvicorn houseprice.api.main:app --host 127.0.0.1 --port 8000 --reload

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

docker:
	docker compose up --build

clean:
	rm -rf models/v* reports/metrics.md .pytest_cache .ruff_cache
