import { useState } from "react";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import "./styles.css";

export default function App() {
  const [page, setPage] = useState("login");
  // login | signup | dashboard

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
