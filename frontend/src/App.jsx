import { useState, useEffect } from "react";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import "./styles.css";

const API = import.meta.env.VITE_API_BASE_URL;

export default function App() {
  const [page, setPage] = useState("login");

  // 🔥 MODEL WARMUP (RUNS ONCE)
  useEffect(() => {
    fetch(`${API}/warmup`, {
      method: "POST"
    }).catch(() => {
      // ignore warmup errors
    });
  }, []);

  return (
    <>
      {page === "login" && (
        <Login
          onLoginSuccess={() => setPage("dashboard")}
          goSignup={() => setPage("signup")}
        />
      )}

      {page === "signup" && (
        <Signup goLogin={() => setPage("login")} />
      )}

      {page === "dashboard" && (
        <Dashboard onLogout={() => setPage("login")} />
      )}
    </>
  );
}
