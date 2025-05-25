# 프로젝트 실행 및 Nginx 설정 가이드

## 1. FE 빌드

- 프론트엔드(FE) 프로젝트 루트에서 아래 명령어 실행

```bash
npm run build
```

##  2.  nginx 실행 혹은 재실행 

### 실행
```bash
./nginx.exe
```

### 재실행
```bash
./nginx.exe -s -reload
```

## 3. BE에서 서버 실행

### 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. localhost:80에 접속

### Nginx 버전
[다운사이트](https://nginx.org/en/download.html)
- Stable version
- nginx/Windows-1.28.0  pgp