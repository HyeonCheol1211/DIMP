import os
import subprocess

# 패키지 자동 설치
#subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# FastAPI 실행
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)