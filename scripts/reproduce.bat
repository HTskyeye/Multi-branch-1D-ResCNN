@echo off
setlocal
REM Reproduce the full pipeline: preprocess -> train
REM Usage: scripts\reproduce.bat

set ROOT_DIR=%~dp0..\
set PYTHONPATH=%ROOT_DIR%src

python -m spectra_dl.preprocess.spectrum --input "%ROOT_DIR%data\raw\spectrum_data.xlsx" --output "%ROOT_DIR%data\processed\spectrum_labels.xlsx" --n_points 120
python -m spectra_dl.preprocess.response --input "%ROOT_DIR%data\raw\100-IT.xls" --output "%ROOT_DIR%data\processed\ai_features.xlsx" --window 320
python -m spectra_dl.train --config "%ROOT_DIR%configs\default.yaml"

endlocal
