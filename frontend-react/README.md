# Medical Interpreter Frontend (React)

Standalone React frontend for the Medical Report Interpreter. Built with Vite + React and integrates with the Flask API.

## Prerequisites
- Node.js 18+ and npm (or pnpm/yarn)
- Backend API running (Python/Flask)

## Configure API base URL
The app reads the API base from `VITE_API_BASE`. Default is `http://localhost:5000`.

Create a `.env.local` (optional):

```
VITE_API_BASE=http://localhost:5000
```

## Install and run
From this folder:

```bash
npm install
npm run dev
```

- Dev server runs at http://localhost:5173
- Backend should run at http://localhost:5000 (adjust `VITE_API_BASE` if different)

To preview a production build on port 8080:

```bash
npm run build
npm run preview
```

## Features parity
- Tabs: Upload PDF, Manual Entry, Batch Upload, Model Comparison
- Upload/Batch: uses FormData to call `/api/interpret` and `/api/batch`
- Manual: posts JSON to `/api/interpret/json`
- Metrics: fetches `/api/model-metrics` and renders charts + confusion matrix with Chart.js
- Results card: measurements, interpretations (with ML/Rule source labels), JSON view and copy button

## Notes
- Ensure CORS is enabled in the backend (already configured via `flask-cors`).
- Start the Flask API from the `medical_interpreter` project root so relative paths resolve properly.
