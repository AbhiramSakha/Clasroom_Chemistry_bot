// ================= BASE URL =================
// Netlify env var:
// VITE_API_BASE_URL=https://clasroomchemistrybot-production.up.railway.app
const BASE = import.meta.env.VITE_API_BASE_URL;

// ================= AUTH API (EXPRESS) =================
export async function login(email, password) {
  const res = await fetch(`${BASE}/api/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ email, password })
  });

  if (!res.ok) throw new Error("Login failed");
  return res.json();
}

export async function signup(email, password) {
  const res = await fetch(`${BASE}/api/signup`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ email, password })
  });

  if (!res.ok) throw new Error("Signup failed");
  return res.json();
}

// ================= AI API (FASTAPI) =================
export async function askModel(text) {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });

  if (!res.ok) throw new Error("Model request failed");
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${BASE}/history`);

  if (!res.ok) throw new Error("History fetch failed");
  return res.json();
}
