const BASE = "http://localhost:8000";

const authHeaders = () => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${localStorage.getItem("token")}`
});

export async function login(email, password) {
  const res = await fetch(`${BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  });
  if (!res.ok) throw new Error("Login failed");
  return res.json();
}

export async function signup(email, password) {
  const res = await fetch(`${BASE}/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  });
  if (!res.ok) throw new Error("Signup failed");
  return res.json();
}

export async function askModel(text) {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ text })
  });
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${BASE}/history`, {
    headers: authHeaders()
  });
  return res.json();
}
