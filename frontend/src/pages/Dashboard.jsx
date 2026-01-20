import { useEffect, useState } from "react";
import { askModel, getHistory } from "../api";
import "../styles.css";
import bgVideo from "../assets/videos/background1.mp4";

export default function Dashboard({ onLogout }) {
  const [question, setQuestion] = useState("");
  const [output, setOutput] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await getHistory();
      setHistory(Array.isArray(data) ? data : []);
    } catch {
      setHistory([]);
    }
  };

  const generate = async () => {
    if (!question.trim()) {
      setOutput("⚠️ Please enter a question.");
      return;
    }

    setLoading(true);
    setOutput("⏳ Generating response...");

    try {
      const res = await askModel(question);
      setOutput(res?.output || "❌ No response received");
      setQuestion("");
      loadHistory();
    } catch {
      setOutput("❌ Server error");
    }

    setLoading(false);
  };

  const formatOutput = (text) => {
    if (!text) return null;

    const lines = text
      .replace(/\r/g, "")
      .split(". ")
      .map((l) => l.trim())
      .filter((l) => l.length > 5);

    if (lines.length > 1) {
      return (
        <ul className="answer-list">
          {lines.map((l, i) => (
            <li key={i}>{l.endsWith(".") ? l : l + "."}</li>
          ))}
        </ul>
      );
    }

    return <p>{text}</p>;
  };

  return (
    <div className="dashboard-wrapper">
      {/* 🎥 BACKGROUND VIDEO */}
      <video autoPlay loop muted playsInline className="dashboard-video">
        <source src={bgVideo} type="video/mp4" />
      </video>

      {/* 🌑 SOFT OVERLAY */}
      <div className="dashboard-overlay"></div>

      <div className="dashboard-layout">
        {/* LEFT HISTORY */}
        <aside className="dashboard-sidebar">
          <h2>🕘 History</h2>

          <div className="history-scroll">
            {history.length === 0 && (
              <div className="empty">No history yet</div>
            )}

            {history.map((item, i) => (
              <div
                key={i}
                className="history-item"
                onClick={() => {
                  setQuestion(item.input);
                  setOutput(item.output);
                }}
              >
                <strong>Q:</strong>{" "}
                {item.input.length > 40
                  ? item.input.slice(0, 40) + "..."
                  : item.input}
              </div>
            ))}
          </div>
        </aside>

        {/* MAIN */}
        <main className="dashboard-main">
          <div className="dashboard-header">
            <h1>🤖 Chemistry AI Assistant</h1>
            <span className="model-name">
              FLAN-T5 + LoRA Fine-Tuned Model
            </span>

            <button className="logout-btn" onClick={onLogout}>
              Logout
            </button>
          </div>

          <div className="card">
            <label>Enter your question</label>
            <textarea
              rows={5}
              placeholder="Ask chemistry questions..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />

            <button onClick={generate} disabled={loading}>
              {loading ? "Generating..." : "Generate Response"}
            </button>
          </div>

          <div className="card">
            <h3>Model Output</h3>
            <div className="output-box">{formatOutput(output)}</div>
          </div>
        </main>
      </div>
    </div>
  );
}
