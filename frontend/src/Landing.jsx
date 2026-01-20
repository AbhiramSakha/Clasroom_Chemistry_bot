import Login from "./Login";
import Signup from "./Signup";

export default function Landing({ page, setPage }) {
  return (
    <>
      {/* ========= SECTION 1: LOGIN / SIGNUP ========= */}
      {page === "login" && (
        <Login
          onLoginSuccess={() => setPage("dashboard")}
          goSignup={() => setPage("signup")}
        />
      )}

      {page === "signup" && (
        <Signup
          goLogin={() => setPage("login")}
        />
      )}

      {/* ========= SECTION 2: ABOUT ========= */}
      <section className="section about-section">
        <div className="section-content">
          <h2>About Chemistry AI</h2>
          <p>
            Chemistry AI is an intelligent chatbot designed to help students,
            educators, and researchers with instant chemistry explanations,
            reactions, concepts, and problem-solving using modern AI models.
          </p>
        </div>
      </section>

      {/* ========= SECTION 3: PARTNERS ========= */}
      <section className="section partners">
        <h2>Initiative Partners</h2>

        <div className="partner-grid">
          <div className="partner-card">
            <h3>KIET Group of Institutions</h3>
            <p>Academic & Innovation Partner</p>
          </div>

          <div className="partner-card">
            <h3>JNTUK Kakinada</h3>
            <p>University Knowledge Partner</p>
          </div>
        </div>
      </section>
    </>
  );
}
