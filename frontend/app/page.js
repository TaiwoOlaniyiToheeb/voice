"use client";

import { useState } from "react";

// The backend URL is injected at build/runtime via an environment variable
// (set in Vercel project settings, or a local .env.local file -- see
// .env.local.example). Falls back to localhost for local development.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLES = ["abeg check my balance", "send 2k give my mama", "call my broda"];

const CONFIDENCE_COLOR = (score) => {
  if (score >= 0.7) return "#10b981";
  if (score >= 0.4) return "#f59e0b";
  return "#ef4444";
};

function ResultCard({ result }) {
  const color = CONFIDENCE_COLOR(result.confidence);
  return (
    <div className="result-card" style={{ borderLeftColor: color }}>
      <div className="result-intent">
        {result.predicted_intent.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())}
      </div>
      <div className="result-desc">{result.description}</div>

      <div className="confidence-label">
        <span>Confidence</span>
        <span>{(result.confidence * 100).toFixed(1)}%</span>
      </div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${Math.min(result.confidence * 100, 100)}%`, background: color }}
        />
      </div>

      <div style={{ fontWeight: 600, fontSize: "0.9rem", marginBottom: "0.6rem" }}>
        Top 5 Predicted Intents
      </div>
      {result.top_5.map((item) => (
        <div className="top5-row" key={item.intent}>
          <div className="top5-label">{item.intent.replaceAll("_", " ")}</div>
          <div className="top5-track">
            <div
              className="top5-fill"
              style={{
                width: `${Math.min(item.confidence * 100, 100)}%`,
                background: CONFIDENCE_COLOR(item.confidence),
              }}
            />
          </div>
          <div className="top5-value">{(item.confidence * 100).toFixed(1)}%</div>
        </div>
      ))}

      {result.transcription_engine && (
        <div className="message info" style={{ marginTop: "1rem" }}>
          Transcribed with <strong>{result.transcription_engine}</strong>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("text");

  const [text, setText] = useState("");
  const [textResult, setTextResult] = useState(null);
  const [textError, setTextError] = useState(null);
  const [textLoading, setTextLoading] = useState(false);

  const [audioFile, setAudioFile] = useState(null);
  const [audioResult, setAudioResult] = useState(null);
  const [audioError, setAudioError] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);

  async function classifyText(commandText) {
    setTextLoading(true);
    setTextError(null);
    setTextResult(null);
    try {
      const res = await fetch(`${API_URL}/predict/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: commandText }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setTextResult(data);
    } catch (err) {
      setTextError(
        err.message === "Failed to fetch"
          ? `Could not reach the API at ${API_URL}. Is the backend running and NEXT_PUBLIC_API_URL set correctly?`
          : err.message
      );
    } finally {
      setTextLoading(false);
    }
  }

  async function handleClassifyClick() {
    if (!text.trim()) {
      setTextError("Please type a command first.");
      return;
    }
    await classifyText(text);
  }

  async function handleAnalyzeClick() {
    if (!audioFile) {
      setAudioError("Please choose an audio file first.");
      return;
    }
    setAudioLoading(true);
    setAudioError(null);
    setAudioResult(null);
    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      const res = await fetch(`${API_URL}/predict/audio`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setAudioResult(data);
    } catch (err) {
      setAudioError(
        err.message === "Failed to fetch"
          ? `Could not reach the API at ${API_URL}. Is the backend running and NEXT_PUBLIC_API_URL set correctly?`
          : err.message
      );
    } finally {
      setAudioLoading(false);
    }
  }

  return (
    <div className="container">
      <div className="hero">
        <div className="ai-badge">✨ AI-POWERED INTENT RECOGNITION</div>
        <h1 className="hero-title">🎙️ Voice & Text Command Classifier</h1>
        <p className="hero-subtitle">
          AI-powered intent recognition for English and Nigerian Pidgin voice/text commands.
        </p>
        <div className="badge-row">
          <span className="feature-badge">Support Vector Classifier</span>
          <span className="feature-badge">TF-IDF</span>
          <span className="feature-badge">12 Intents</span>
          <span className="feature-badge">Voice + Text</span>
        </div>
      </div>

      <div className="tabs">
        <button
          className={`tab-button ${activeTab === "text" ? "active" : ""}`}
          onClick={() => setActiveTab("text")}
        >
          Text Command
        </button>
        <button
          className={`tab-button ${activeTab === "audio" ? "active" : ""}`}
          onClick={() => setActiveTab("audio")}
        >
          Voice Command
        </button>
      </div>

      {activeTab === "text" && (
        <div className="card" style={{ borderTopLeftRadius: 0 }}>
          <h3>Type a Command</h3>
          <p style={{ color: "var(--color-muted)", marginTop: "-0.5rem" }}>
            Try Pidgin or English e.g. &quot;abeg check my balance&quot; or &quot;send money to
            my brother&quot;.
          </p>

          <div className="example-row">
            {EXAMPLES.map((ex) => (
              <button key={ex} className="example-chip" onClick={() => setText(ex)}>
                {ex}
              </button>
            ))}
          </div>

          <input
            className="text-input"
            type="text"
            placeholder="Type a command such as 'check my balance'..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleClassifyClick()}
          />

          <div className="button-row">
            <button className="btn" onClick={handleClassifyClick} disabled={textLoading}>
              {textLoading ? "Classifying..." : "Classify Command"}
            </button>
            <button
              className="btn secondary"
              onClick={() => {
                setText("");
                setTextResult(null);
                setTextError(null);
              }}
            >
              Clear
            </button>
          </div>

          {textError && <div className="message error">⚠️ {textError}</div>}
          {textResult && <ResultCard result={textResult} />}
        </div>
      )}

      {activeTab === "audio" && (
        <div className="card" style={{ borderTopLeftRadius: 0 }}>
          <h3>🎤 Upload a Voice Command</h3>
          <p style={{ color: "var(--color-muted)", marginTop: "-0.5rem" }}>
            Supported formats: .wav, .flac, .aiff
          </p>

          <div className="upload-area">
            <div>Drop your audio file here, or browse</div>
            <input
              type="file"
              accept=".wav,.flac,.aiff,.aif"
              onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
            />
          </div>

          <div className="button-row">
            <button className="btn" onClick={handleAnalyzeClick} disabled={audioLoading}>
              {audioLoading ? "Analyzing..." : "Analyze Audio"}
            </button>
          </div>

          <div className="message info" style={{ marginTop: "1rem" }}>
            Want live microphone input? Run{" "}
            <code>python src/voice_predict.py --mic</code> from the main project locally —
            that path captures directly from your microphone.
          </div>

          {audioError && <div className="message error">⚠️ {audioError}</div>}
          {audioResult && <ResultCard result={audioResult} />}
        </div>
      )}

      <div className="footer">
        Developed by <strong>Taiwo Olaniyi Toheeb</strong>
        <br />
        Built with
        <span className="tech-pill">Next.js</span>
        <span className="tech-pill">FastAPI</span>
        <span className="tech-pill">Scikit-learn</span>
        <span className="tech-pill">SpeechRecognition</span>
        <span className="tech-pill">TF-IDF</span>
        <span className="tech-pill">Support Vector Classifier</span>
      </div>
    </div>
  );
}
