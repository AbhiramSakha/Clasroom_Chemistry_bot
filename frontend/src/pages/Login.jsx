import { useState } from "react";
import "../styles.css";
import img1 from "../assets/auth/chem1.jpg";
import img2 from "../assets/auth/chem2.jpg";
import img3 from "../assets/auth/chem3.jpg";
import img4 from "../assets/auth/chem4.jpg";
import img5 from "../assets/auth/chem5.jpg";
import bgVideo from "../assets/videos/background.mp4";

export default function Login({ onLoginSuccess }) {
  const [show, setShow] = useState(false);
  const [mode, setMode] = useState("login");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");

  /* ================= SIGNUP ================= */
  const handleSignup = () => {
    if (!email || !password || !confirm) {
      alert("Please fill all fields");
      return;
    }
    if (password !== confirm) {
      alert("Passwords do not match");
      return;
    }

    localStorage.setItem(
      "chem-user",
      JSON.stringify({ email, password })
    );

    alert("Signup successful! Redirecting to login...");
    setEmail("");
    setPassword("");
    setConfirm("");
    setMode("login");
  };

  /* ================= LOGIN ================= */
  const handleLogin = () => {
    setError("");
    const savedUser = JSON.parse(localStorage.getItem("chem-user"));

    if (!savedUser) {
      setError("No account found. Please signup first.");
      return;
    }

    if (email === savedUser.email && password === savedUser.password) {
      onLoginSuccess();
    } else {
      setError("Incorrect email or password");
    }
  };

  /* ================= FORGOT PASSWORD (FIXED ✅) ================= */
  const handleForgotPassword = () => {
    const savedUser = JSON.parse(localStorage.getItem("chem-user"));

    if (!savedUser) {
      alert("No account found. Please signup first.");
      return;
    }

    if (email !== savedUser.email) {
      alert("Email not registered");
      return;
    }

    if (!password || !confirm) {
      alert("Please fill all fields");
      return;
    }

    if (password !== confirm) {
      alert("Passwords do not match");
      return;
    }

    // Update password
    localStorage.setItem(
      "chem-user",
      JSON.stringify({ email, password })
    );

    alert("Password updated successfully! Please login.");
    setPassword("");
    setConfirm("");
    setMode("login");
  };

  return (
    <div className="auth-page">
      {/* VIDEO BACKGROUND */}
      <video autoPlay loop muted className="background-video">
        <source src={bgVideo} type="video/mp4" />
      </video>

      {/* TOP NAV */}
      <div className="top-nav">
        <button onClick={() => setShow(true)}>Login / Signup</button>
      </div>

      {/* LEFT CONTENT */}
      <div className="auth-left">
        <h1 className="auth-title split-text">
          <span className="fly-left">Learn Chemistry</span><br />
          <span className="fly-right">with <span className="ai-text">AI</span></span>
        </h1>

        <p className="auth-sub fade-up">
          AI-powered chemistry assistant providing instant explanations,
          reactions, and concepts.
        </p>

        <div className="auth-about fly-in">
          <h3>About</h3>
          <p>
            This platform helps students understand chemistry concepts using
            modern AI models with real-time responses.
          </p>
        </div>

        <div className="partners">
          <h4>Partners</h4>
          <div className="partner-wheel">
            <div className="partner">KIET Group of Institutions</div>
            <div className="partner">JNTU-KAKINADA</div>
          </div>

          {/* DEVELOPERS (RESTORED ✅) */}
          <div className="developers">
            <h4>Developed By</h4>
            <div className="developer-wheel">
              <div className="developer">Sakha Abhiram</div>
              <div className="developer">Tamada Srivalli</div>
              <div className="developer">Vantakula Manikanta</div>
              <div className="developer">Dulam Shanmuka M S Vishnu Vardhan</div>
              <div className="developer">Samanthula Avinash</div>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT IMAGES – 3 COLUMNS */}
      <div className="auth-right">
        <div className="image-column down">
          <img src={img1} alt="Chemistry" />
          <img src={img2} alt="Chemistry" />
          <img src={img3} alt="Chemistry" />
          <img src={img4} alt="Chemistry" />
          <img src={img5} alt="Chemistry" />
        </div>

        <div className="image-column up">
          <img src={img2} alt="Chemistry" />
          <img src={img1} alt="Chemistry" />
          <img src={img3} alt="Chemistry" />
          <img src={img4} alt="Chemistry" />
          <img src={img5} alt="Chemistry" />
        </div>

        <div className="image-column down">
          <img src={img1} alt="Chemistry" />
          <img src={img2} alt="Chemistry" />
          <img src={img3} alt="Chemistry" />
          <img src={img4} alt="Chemistry" />
          <img src={img5} alt="Chemistry" />
        </div>
      </div>

      {/* AUTH MODAL */}
      {show && (
        <div className="auth-modal">
          <div className="auth-card">
            <span className="modal-close" onClick={() => setShow(false)}>
              &times;
            </span>

            {/* LOGIN */}
            {mode === "login" && (
              <>
                {error && <p style={{ color: "red", textAlign: "center" }}>{error}</p>}
                <h2>Login</h2>
                <input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
                <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} />
                <button className="auth-btn" onClick={handleLogin}>Login</button>
                <button className="auth-btn secondary" onClick={() => setMode("signup")}>Signup</button>
                <p><span onClick={() => setMode("forgot")}>Forgot password?</span></p>
              </>
            )}

            {/* SIGNUP */}
            {mode === "signup" && (
              <>
                <h2>Create Account</h2>
                <input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
                <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} />
                <input type="password" placeholder="Confirm Password" value={confirm} onChange={e => setConfirm(e.target.value)} />
                <button className="auth-btn" onClick={handleSignup}>Signup</button>
                <p>Back to <span onClick={() => setMode("login")}>Login</span></p>
              </>
            )}

            {/* FORGOT PASSWORD – FIXED ✅ */}
            {mode === "forgot" && (
              <>
                <h2>Reset Password</h2>
                <input placeholder="Registered Email" value={email} onChange={e => setEmail(e.target.value)} />
                <input type="password" placeholder="New Password" value={password} onChange={e => setPassword(e.target.value)} />
                <input type="password" placeholder="Re-enter New Password" value={confirm} onChange={e => setConfirm(e.target.value)} />
                <button className="auth-btn" onClick={handleForgotPassword}>Update Password</button>
                <p>Back to <span onClick={() => setMode("login")}>Login</span></p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
