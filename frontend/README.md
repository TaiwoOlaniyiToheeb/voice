# Frontend (Next.js, for Vercel)

A Next.js app that gives the classifier a browser UI, styled to match the
Streamlit version. It doesn't run the model itself — it makes HTTP
requests to the FastAPI backend in `../backend`. Deploy that backend
first (see `../backend/README.md`), then point this frontend at it.

## Run locally

```bash
cd frontend
npm install
cp .env.local.example .env.local
# edit .env.local if your backend isn't on localhost:8000
npm run dev
```
Opens at `http://localhost:3000`. Make sure the backend (`../backend`) is
running first, or the Classify/Analyze buttons will show a connection error.

**This was actually tested, not just written and assumed to work:**
- `npm install` — confirmed **0 vulnerabilities** (initially scaffolded
  with Next.js 14.2.5, `npm audit` flagged 2 issues; upgraded to 16.3.1,
  clean)
- `npm run build` — compiled successfully with no errors
- Full integration test: booted the backend and this frontend together,
  confirmed the CORS preflight succeeds, and a real cross-origin POST
  from the frontend's origin to the backend returned a correct live
  prediction (typed "abeg buy data for me" → `buy_data`, 98.1% confidence)

## Deploy to Vercel

1. Push this repo to GitHub (the whole repo, or just push with this
   folder — Vercel lets you set a **Root Directory**).
2. Go to **vercel.com/new**, import the GitHub repo.
3. Set **Root Directory** to `frontend`.
4. Vercel auto-detects Next.js — no build command changes needed.
5. Under **Environment Variables**, add:
   ```
   NEXT_PUBLIC_API_URL = https://your-backend-url.onrender.com
   ```
   (Use whatever URL you got from deploying `../backend` — Render,
   Railway, or Fly.io.)
6. Click **Deploy**. You'll get a URL like `https://your-app.vercel.app`.

**Important:** the backend must be deployed and its CORS `allow_origins`
updated to include your Vercel domain (see `../backend/README.md` →
CORS section) before the deployed frontend can successfully call it.
`NEXT_PUBLIC_*` variables are baked in at build time, so if you change
`NEXT_PUBLIC_API_URL` in Vercel's settings after the first deploy,
trigger a redeploy for it to take effect.

## Project structure

```
frontend/
├── app/
│   ├── layout.js       # root layout, page metadata
│   ├── page.js         # the whole UI: tabs, forms, results, footer
│   └── globals.css     # styling (same color palette as the Streamlit app)
├── .env.local.example  # copy to .env.local for local dev
├── next.config.js
└── package.json
```

## Why a file upload instead of live microphone recording?

Same reasoning as the Streamlit version: live browser microphone capture
needs extra components (e.g. `MediaRecorder` API + a way to stream audio
to the backend) that add real fragility to a first deployment. File
upload reuses the exact same `/predict/audio` backend logic that was
already tested end-to-end. For live mic input, `python src/voice_predict.py --mic`
in the main project still works great locally.
