@echo off
echo Checking Docker and Qdrant Database...

:: Try to start existing container. If it fails (||), create a new one.
docker start qdrant 2>nul || docker run -d --name qdrant -p 6333:6333 -v "C:\Users\syami\Desktop\Meem\qdrant_storage:/qdrant/storage" qdrant/qdrant

echo.
echo Starting Local Voice Agent...
cd C:\Users\syami\Desktop\Meem\local_voice_agent
call venv\Scripts\activate.bat
python -m streamlit run app.py


