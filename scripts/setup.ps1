if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}

.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r ./scripts/requirements.txt