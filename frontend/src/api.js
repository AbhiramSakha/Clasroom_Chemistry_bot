// Use environment variable (Netlify / Vite)
const BASE = import.meta.env.VITE_API_BASE_URL;

/* ================= AUTH HEADERS ================= */
const authHeaders = () => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${localStorage.getItem("token") || ""}`
});

/* ================= LOGIN ================= */
export async function login(email, password) {
  const res = await fetch(`${BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  });

  if (!res.ok) throw new Error("Login failed");
  return res.json();
}

/* ================= SIGNUP ================= */
export async function signup(email, password) {
  const res = await fetch(`${BASE}/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  });

  if (!res.ok) throw new Error("Signup failed");
  return res.json();
}

/* ================= ASK MODEL ================= */
export async function askModel(text) {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ text })
  });

  if (!res.ok) throw new Error("Model request failed");
  return res.json();
}

/* ================= HISTORY ================= */
export async function getHistory() {
  const res = await fetch(`${BASE}/history`, {
    headers: authHeaders()
  });

  if (!res.ok) throw new Error("History fetch failed");
  return res.json();
}
