import { useState } from "react";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import "./styles.css";

export default function App() {
  // Initialize page based on session
  const [page, setPage] = useState(() => {
    return localStorage.getItem("chem-session") ? "dashboard" : "login";
  });

  const handleLoginSuccess = () => {
    localStorage.setItem("chem-session", "1");
    window.history.pushState({}, "", "/dashboard");
    setPage("dashboard");
  };

  const handleLogout = () => {
    localStorage.removeItem("chem-session");
    localStorage.removeItem("chem-terms");
    window.history.pushState({}, "", "/");
    setPage("login");
  };

  return (
    <>
      {page === "login" && (
        <Login
          onLoginSuccess={handleLoginSuccess}
          goSignup={() => setPage("signup")}
        />
      )}

      {page === "signup" && (
        <Signup goLogin={() => setPage("login")} />
      )}

      {page === "dashboard" && (
        <Dashboard onLogout={handleLogout} />
      )}
    </>
  );
}