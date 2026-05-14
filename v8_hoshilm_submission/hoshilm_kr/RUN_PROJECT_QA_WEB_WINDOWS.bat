@echo off
cd /d %~dp0
python build_project_qa_corpus.py
python project_qa_api.py --host 127.0.0.1 --port 8000
pause
