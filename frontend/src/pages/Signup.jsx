import { useState } from "react";
import img1 from "../assets/auth/chem1.jpg";
import img2 from "../assets/auth/chem2.jpg";
import img3 from "../assets/auth/chem3.jpg";
import img4 from "../assets/auth/chem4.jpg";
import img5 from "../assets/auth/chem5.jpg";
import "../styles.css";

export default function Signup({ goLogin }) {
  const [show, setShow] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");

  const handleSignup = () => {
    if (!email || !password || !confirm) {
      alert("Please fill all fields");
      return;
    }
    if (password !== confirm) {
      alert("Passwords do not match");
      return;
    }

    // Store user data in localStorage
    localStorage.setItem(
      "chem-user",
      JSON.stringify({ email, password })
    );

    alert("Signup successful! Redirecting to login...");

    // Redirect to login
    goLogin();
  };

  return (
    <div className="auth-page">
      {/* TOP NAV */}
      <div className="top-nav">
        <button onClick={() => setShow(true)}>Signup</button>
      </div>

      {/* LEFT CONTENT */}
      <div className="auth-left">
        <h1 className="auth-title">
          Learn Chemistry <br /> with <span>AI</span>
        </h1>

        <p className="auth-sub">
          Start learning chemistry smarter with AI-powered explanations.
        </p>

        <div className="auth-about">
          <h3>About</h3>
          <p>
            This platform helps students understand chemistry concepts using
            modern AI models with real-time responses.
          </p>
        </div>

        <div className="partners">
          <h4>Partners</h4>
          <div className="partner">KIET Group of Institutions</div>
          <div className="partner">JNTUK-KAKINADA</div>
        </div>
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

      {/* RIGHT IMAGES */}
      <div className="auth-right">
        {/* COLUMN 1 – DOWN */}
        <div className="image-column down">
          <img src={img1} alt="Chemistry" />
          <img src={img2} alt="Chemistry" />
          <img src={img3} alt="Chemistry" />
          <img src={img4} alt="Chemistry" />
          <img src={img5} alt="Chemistry" />
        </div>

        {/* COLUMN 2 – UP */}
        <div className="image-column up">
          <img src={img1} alt="Chemistry" />
          <img src={img2} alt="Chemistry" />
          <img src={img3} alt="Chemistry" />
          <img src={img4} alt="Chemistry" />
          <img src={img5} alt="Chemistry" />
        </div>

        {/* COLUMN 3 – DOWN */}
        <div className="image-column down">
          <img src={img1} alt="Chemistry" />
          <img src={img2} alt="Chemistry" />
          <img src={img3} alt="Chemistry" />
          <img src={img2} alt="Chemistry" />
          <img src={img1} alt="Chemistry" />
          <img src={img3} alt="Chemistry" />
          <img src={img1} alt="Chemistry" />
        </div>
      </div>

      {/* SIGNUP MODAL */}
      {show && (
        <div className="auth-modal">
          <div className="auth-card">
            <span className="modal-close" onClick={() => setShow(false)}>
              &times;
            </span>

            <h2>Create Account</h2>

            <input
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />

            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <input
              type="password"
              placeholder="Confirm Password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />

            <button className="auth-btn" onClick={handleSignup}>
              Signup
            </button>

            <p>
              Already have an account?{" "}
              <span onClick={goLogin}>Login</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
