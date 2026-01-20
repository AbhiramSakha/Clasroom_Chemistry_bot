import { useState } from "react";
import "../styles.css";

export default function ForgotPassword({ goLogin }) {
  const [email, setEmail] = useState("");
  const [oldPass, setOldPass] = useState("");
  const [newPass, setNewPass] = useState("");

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h2>Reset Password</h2>

        <input
          placeholder="Registered Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          placeholder="Old Password"
          value={oldPass}
          onChange={(e) => setOldPass(e.target.value)}
        />

        <input
          type="password"
          placeholder="New Password"
          value={newPass}
          onChange={(e) => setNewPass(e.target.value)}
        />

        <button onClick={goLogin}>Update Password</button>

        <p className="link" onClick={goLogin}>Back to Login</p>
      </div>
    </div>
  );
}
