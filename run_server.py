"""CodeWeaver backend launcher: python run_server.py"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./backend'))
os.environ.setdefault('CODEWEAVER_DATA_DIR', './data')
os.environ.setdefault('CODEWEAVER_WORKSPACE_ROOT', './data/workspaces')
os.environ.setdefault('CODEWEAVER_DB_URL', 'sqlite+aiosqlite:///./data/forgeai.db')

from app.api.main import app
import uvicorn

if __name__ == '__main__':
    print("CodeWeaver — Autonomous AI Engineering backend")
    print("Starting server at http://127.0.0.1:8600 (press Ctrl+C to stop) ...")
    uvicorn.run(app, host='127.0.0.1', port=8600, log_level='info')
