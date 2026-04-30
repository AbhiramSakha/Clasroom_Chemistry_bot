import { useState, useEffect, useRef, useCallback } from "react";

/* ══════════════════════════════════════════════════════════════
   CHEMISTRY AI — LOGIN v3.0  "Real-World 3D"
   ──────────────────────────────────────────────────────────────
   Visual stack:
   • Three.js rotating 3D atom (nucleus + 3 orbital rings + 3 electrons)
   • 2D glowing particle network (canvas)
   • Perspective 3D card tilt on mouse move
   • Animated scanline overlay
   • Radial ambient orbs (CSS animated)
   • Isometric grid background
   • Periodic table element strip (rotating highlight)
   • Feature info grid, tech badges, stats
   • Typed headline (cycles through phrases)
   • Shimmer gradient submit button
   • Password strength meter
   • Progress steps indicator
══════════════════════════════════════════════════════════════ */

export default function Login({ onLoginSuccess }) {

  /* ── Form state ────────────────────────────────────── */
  const [mode,        setMode]        = useState("login");
  const [email,       setEmail]       = useState("");
  const [password,    setPassword]    = useState("");
  const [confirm,     setConfirm]     = useState("");
  const [error,       setError]       = useState("");
  const [success,     setSuccess]     = useState("");
  const [showPass,    setShowPass]    = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading,     setLoading]     = useState(false);

  /* ── Animation state ────────────────────────────────── */
  const [mousePos,      setMousePos]      = useState({ x:0, y:0 });
  const [cardTilt,      setCardTilt]      = useState({ x:0, y:0 });
  const [typedText,     setTypedText]     = useState("");
  const [activeElem,    setActiveElem]    = useState(0);

  /* ── Refs ────────────────────────────────────────────── */
  const canvasRef  = useRef(null);
  const threeRef   = useRef(null);
  const raf2d      = useRef(null);
  const cardRef    = useRef(null);
  const headRef    = useRef(0);
  const charRef    = useRef(0);
  const delRef     = useRef(false);
  const timerRef   = useRef(null);

  const HEADLINES = ["Explore Chemistry","Master Reactions","Decode Elements","Ace Your Exams","Think Molecular"];

  /* ── Typing animation ─────────────────────────────── */
  useEffect(() => {
    const tick = () => {
      const cur = HEADLINES[headRef.current];
      if (!delRef.current) {
        setTypedText(cur.slice(0, charRef.current + 1));
        charRef.current++;
        if (charRef.current === cur.length) {
          delRef.current = true;
          timerRef.current = setTimeout(tick, 2200);
          return;
        }
      } else {
        setTypedText(cur.slice(0, charRef.current - 1));
        charRef.current--;
        if (charRef.current === 0) {
          delRef.current = false;
          headRef.current = (headRef.current + 1) % HEADLINES.length;
        }
      }
      timerRef.current = setTimeout(tick, delRef.current ? 45 : 88);
    };
    timerRef.current = setTimeout(tick, 600);
    return () => clearTimeout(timerRef.current);
  }, []);

  /* ── Global mouse parallax ─────────────────────────── */
  useEffect(() => {
    const h = (e) => setMousePos({
      x: (e.clientX / window.innerWidth  - 0.5) * 38,
      y: (e.clientY / window.innerHeight - 0.5) * 26,
    });
    window.addEventListener("mousemove", h);
    return () => window.removeEventListener("mousemove", h);
  }, []);

  /* ── Card 3-D tilt ───────────────────────────────────── */
  const onCardMove = useCallback((e) => {
    const el = cardRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const dx = (e.clientX - (r.left + r.width  / 2)) / (r.width  / 2);
    const dy = (e.clientY - (r.top  + r.height / 2)) / (r.height / 2);
    setCardTilt({ x: -dy * 9, y: dx * 9 });
  }, []);
  const onCardLeave = useCallback(() => setCardTilt({ x:0, y:0 }), []);

  /* ── 2D glowing particle network ─────────────────────── */
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    let W = cv.width  = window.innerWidth;
    let H = cv.height = window.innerHeight;

    const PTS = Array.from({ length: 72 }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.34,
      vy: (Math.random() - 0.5) * 0.34,
      r: Math.random() * 1.6 + 0.5,
      ph: Math.random() * Math.PI * 2,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      PTS.forEach(p => {
        p.x += p.vx; p.y += p.vy; p.ph += 0.028;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
        const alpha = 0.35 + 0.35 * Math.sin(p.ph);
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
        g.addColorStop(0, `rgba(56,189,248,${alpha})`);
        g.addColorStop(1, "rgba(56,189,248,0)");
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
      });
      PTS.forEach((a, i) => {
        PTS.slice(i + 1).forEach(b => {
          const d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < 145) {
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `rgba(56,189,248,${0.12 * (1 - d / 145)})`;
            ctx.lineWidth = 0.6; ctx.stroke();
          }
        });
      });
      raf2d.current = requestAnimationFrame(draw);
    };
    draw();
    const resize = () => { W = cv.width = window.innerWidth; H = cv.height = window.innerHeight; };
    window.addEventListener("resize", resize);
    return () => { cancelAnimationFrame(raf2d.current); window.removeEventListener("resize", resize); };
  }, []);

  /* ── Three.js 3D rotating atom ──────────────────────── */
  useEffect(() => {
    const cv = threeRef.current;
    if (!cv) return;
    let threeRaf;
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    script.onload = () => {
      const THREE = window.THREE;
      const W = cv.clientWidth, H = cv.clientHeight;
      const renderer = new THREE.WebGLRenderer({ canvas: cv, alpha: true, antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(W, H);

      const scene  = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
      camera.position.z = 7;

      /* Nucleus core */
      const nucGeo = new THREE.SphereGeometry(0.55, 48, 48);
      const nucMat = new THREE.MeshPhongMaterial({
        color: 0x0ea5e9, emissive: 0x0c4a6e, shininess: 150,
        transparent: true, opacity: 0.95,
      });
      const nucleus = new THREE.Mesh(nucGeo, nucMat);
      scene.add(nucleus);

      /* Nucleus glow halo */
      const halGeo = new THREE.SphereGeometry(0.78, 32, 32);
      const halMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.1 });
      scene.add(new THREE.Mesh(halGeo, halMat));

      /* Orbital rings */
      const makeRing = (rx, ry, rz, color) => {
        const geo = new THREE.TorusGeometry(2.3, 0.022, 16, 100);
        const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55 });
        const m   = new THREE.Mesh(geo, mat);
        m.rotation.set(rx, ry, rz);
        scene.add(m);
        return m;
      };
      const rings = [
        makeRing(0, 0, 0, 0x38bdf8),
        makeRing(Math.PI / 3,  0, Math.PI / 5,  0x818cf8),
        makeRing(-Math.PI / 3, 0, -Math.PI / 5, 0x06b6d4),
      ];

      /* Electrons */
      const makeElec = (color) => {
        const geo = new THREE.SphereGeometry(0.14, 20, 20);
        const mat = new THREE.MeshPhongMaterial({ color, emissive: color, shininess: 220 });
        const m   = new THREE.Mesh(geo, mat);
        /* Glow */
        const hgeo = new THREE.SphereGeometry(0.26, 20, 20);
        const hmat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.2 });
        m.add(new THREE.Mesh(hgeo, hmat));
        scene.add(m);
        return m;
      };
      const elecs = [
        makeElec(0x38bdf8),
        makeElec(0xa78bfa),
        makeElec(0x34d399),
      ];

      /* Lights */
      scene.add(new THREE.AmbientLight(0x1e3a5f, 3.5));
      const dl = new THREE.DirectionalLight(0x38bdf8, 3);
      dl.position.set(5, 5, 5);
      scene.add(dl);
      const pl = new THREE.PointLight(0x818cf8, 2, 14);
      pl.position.set(-4, -3, 2);
      scene.add(pl);
      const pl2 = new THREE.PointLight(0x34d399, 1.5, 10);
      pl2.position.set(3, -4, -2);
      scene.add(pl2);

      /* Orbital tilt configs */
      const tilts = [
        [0, 0, 0],
        [Math.PI / 3,  0, Math.PI / 5],
        [-Math.PI / 3, 0, -Math.PI / 5],
      ];

      let t = 0;
      const animate = () => {
        threeRaf = requestAnimationFrame(animate);
        t += 0.011;
        nucleus.rotation.y += 0.007;
        nucleus.rotation.x += 0.003;

        elecs.forEach((e, i) => {
          const angle = t * [1, 0.72, 1.28][i] + [0, 2.1, 4.2][i];
          const R = 2.3;
          const bx = Math.cos(angle) * R;
          const bz = Math.sin(angle) * R;
          const [rx, , rz] = tilts[i];
          const cx = Math.cos(rz), sx = Math.sin(rz);
          const cy = Math.cos(rx), sy = Math.sin(rx);
          const y1 = -bz * sy;
          const z1 =  bz * cy;
          e.position.set(bx * cx - y1 * sx, bx * sx + y1 * cx, z1);
        });

        rings.forEach((r, i) => { r.rotation.y += [0.004, 0.003, 0.005][i]; });
        renderer.render(scene, camera);
      };
      animate();
    };
    document.head.appendChild(script);
    return () => {
      if (threeRaf) cancelAnimationFrame(threeRaf);
      try { document.head.removeChild(script); } catch {}
    };
  }, []);

  /* ── Cycling element highlight ─────────────────────── */
  useEffect(() => {
    const id = setInterval(() => setActiveElem(n => (n + 1) % ELEMENTS.length), 2000);
    return () => clearInterval(id);
  }, []);

  /* ── Auth logic ──────────────────────────────────────── */
  const clearFields = () => { setEmail(""); setPassword(""); setConfirm(""); setError(""); setSuccess(""); };
  const switchMode  = (m) => { clearFields(); setMode(m); };

  const simulate = (fn) => {
    setLoading(true); setError(""); setSuccess("");
    setTimeout(() => { setLoading(false); fn(); }, 950);
  };

  const handleSignup = () => {
    if (!email || !password || !confirm) { setError("Please fill in all fields."); return; }
    if (!/\S+@\S+\.\S+/.test(email))    { setError("Enter a valid email address."); return; }
    if (password.length < 6)             { setError("Password must be at least 6 characters."); return; }
    if (password !== confirm)            { setError("Passwords do not match."); return; }
    simulate(() => {
      const users = JSON.parse(localStorage.getItem("chem-users")) || [];
      if (users.find(u => u.email === email)) { setError("Email already registered."); return; }
      users.push({ email, password });
      localStorage.setItem("chem-users", JSON.stringify(users));
      setSuccess("Account created! Please sign in.");
      setTimeout(() => switchMode("login"), 1300);
    });
  };

  const handleLogin = () => {
    if (!email || !password) { setError("Please enter your email and password."); return; }
    simulate(() => {
      const users = JSON.parse(localStorage.getItem("chem-users")) || [];
      const ok = users.find(u => u.email === email && u.password === password);
      if (ok) { onLoginSuccess(); }
      else      { setError("Invalid email or password."); }
    });
  };

  const handleForgot = () => {
    if (!email)                { setError("Enter the email linked to your account."); return; }
    if (!password || !confirm) { setError("Enter and confirm your new password."); return; }
    if (password !== confirm)  { setError("Passwords do not match."); return; }
    simulate(() => {
      const users = JSON.parse(localStorage.getItem("chem-users")) || [];
      const idx = users.findIndex(u => u.email === email);
      if (idx === -1) { setError("Email not found."); return; }
      users[idx].password = password;
      localStorage.setItem("chem-users", JSON.stringify(users));
      setSuccess("Password updated!");
      setTimeout(() => switchMode("login"), 1300);
    });
  };

  const submit = mode === "login" ? handleLogin : mode === "signup" ? handleSignup : handleForgot;

  /* ── Password strength ─────────────────────────────── */
  const pwStrength = !password ? 0
    : password.length >= 10 && /[A-Z]/.test(password) && /[0-9]/.test(password) && /[^A-Za-z0-9]/.test(password) ? 4
    : password.length >= 8  && /[A-Z]/.test(password) && /[0-9]/.test(password) ? 3
    : password.length >= 6  ? 2 : 1;
  const pwLabel  = ["","Weak","Fair","Strong","Excellent"][pwStrength];
  const pwColor  = ["","#ef4444","#f59e0b","#22c55e","#38bdf8"][pwStrength];

  /* ── Steps ─────────────────────────────────────────── */
  const steps = mode === "login"  ? ["Credentials","Access"]
    : mode === "signup" ? ["Identity","Security","Confirm"]
    : ["Email","New Password","Update"];
  const curStep = mode === "login" ? (password ? 1 : 0)
    : mode === "signup" ? (confirm ? 2 : password ? 1 : 0)
    : (password ? 1 : 0);

  const cardTransform = `perspective(1100px) rotateX(${cardTilt.x}deg) rotateY(${cardTilt.y}deg)`;

  /* ══════════════════════════════════════ RENDER ══════════════════════════════════════ */
  return (
    <div style={R.root}>
      <style>{CSS}</style>

      {/* 2D particles */}
      <canvas ref={canvasRef} style={R.canvas} />

      {/* Dark deep-space bg */}
      <div style={R.bg} />

      {/* Isometric grid */}
      <div style={R.grid} />

      {/* Moving scanline */}
      <div style={R.scanline} />

      {/* Ambient glowing orbs */}
      <div style={R.orb1} />
      <div style={R.orb2} />
      <div style={R.orb3} />

      {/* ═══════════ LEFT PANEL ═══════════ */}
      <div style={{ ...R.left, transform: `translate(${mousePos.x * 0.055}px, ${mousePos.y * 0.055}px)` }}>

        {/* Logo */}
        <div style={R.badge} className="fadeIn">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="1.8">
            <circle cx="12" cy="12" r="2.5" fill="#38bdf8"/>
            <ellipse cx="12" cy="12" rx="10" ry="4"/>
            <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)"/>
            <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)"/>
          </svg>
          <span style={R.badgeText}>ChemBot AI</span>
          <span style={R.badgePill}>KIET · JNTU Kakinada</span>
        </div>

        {/* Typed headline */}
        <h1 style={R.h1} className="slideR">
  {/* Animated headline */}
  <span style={R.typed}>
    {typedText || "Explore Chemistry"}
    <span style={R.cursor}>|</span>
  </span>

  {/* Static line */}
  <div style={{ marginTop: 6 }}>
    <span style={R.hlWith}>with </span>
    <span style={R.hlAI}>AI</span>
  </div>
</h1>

        <p style={R.sub} className="fadeIn">
          A university-grade platform powered by{" "}
          <strong style={{ color:"#38bdf8" }}>FLAN-T5 + LoRA</strong> — point-wise
          chemistry answers, molecular diagrams, PDF analysis, interactive quizzes, and
          multilingual support across 11 languages.
        </p>

        {/* ── 3D Atom canvas ── */}
        <div style={R.atomWrap} className="fadeIn">
          <canvas ref={threeRef} style={R.threeCv} />
          {/* Floating molecule badges */}
          {FLOAT_MOLS.map((m, i) => (
            <div key={m.f} style={{ ...R.floatMol, top:`${m.top}%`, left:`${m.left}%`, animationDelay:`${i*0.8}s`, borderColor:m.c+"44", background:m.c+"10", color:m.c }}>
              <span style={{ fontSize:11, fontWeight:900 }}>{m.f}</span>
              <span style={{ fontSize:9, opacity:0.65 }}>{m.mw}</span>
            </div>
          ))}
          <div style={R.atomLbl}>3D Molecular Model · Real-time WebGL</div>
        </div>

        {/* ── Features grid ── */}
        <div style={R.featGrid} className="fadeIn">
          {FEATS.map((f,i) => (
            <div key={f.t} style={{ ...R.featCard, animationDelay:`${i*0.09}s` }} className="fadeIn"
              onMouseEnter={e => { e.currentTarget.style.background="rgba(56,189,248,0.07)"; e.currentTarget.style.borderColor="rgba(56,189,248,0.25)"; e.currentTarget.style.transform="translateY(-2px)"; }}
              onMouseLeave={e => { e.currentTarget.style.background="rgba(255,255,255,0.02)"; e.currentTarget.style.borderColor="rgba(255,255,255,0.07)"; e.currentTarget.style.transform="translateY(0)"; }}>
              <span style={{ fontSize:19 }}>{f.i}</span>
              <span style={{ color:"#f1f5f9", fontSize:11, fontWeight:700, lineHeight:1.3 }}>{f.t}</span>
              <span style={{ color:"#64748b", fontSize:10, lineHeight:1.4, marginTop:1 }}>{f.d}</span>
            </div>
          ))}
        </div>

        {/* ── Periodic table strip ── */}
        <div style={R.elemStrip} className="fadeIn">
          {ELEMENTS.map((el, i) => (
            <div key={el.s} style={{
              ...R.elemCell,
              background: i === activeElem ? el.c+"1a" : "rgba(255,255,255,0.025)",
              borderColor: i === activeElem ? el.c : "rgba(255,255,255,0.07)",
              color: i === activeElem ? el.c : "#475569",
              transform: i === activeElem ? "scale(1.14) translateY(-4px)" : "scale(1)",
              boxShadow: i === activeElem ? `0 4px 18px ${el.c}40` : "none",
              transition: "all 0.35s ease",
            }}>
              <div style={{ fontSize:7, fontWeight:700, opacity:0.7 }}>{el.n}</div>
              <div style={{ fontSize:14, fontWeight:900 }}>{el.s}</div>
              <div style={{ fontSize:7, opacity:0.6, textAlign:"center" }}>{el.nm}</div>
            </div>
          ))}
        </div>

        {/* ── Stats ── */}
        <div style={R.stats} className="fadeIn">
          {STATS.map(({ n, l, ic }) => (
            <div key={l} style={R.statBox}>
              <span style={{ fontSize:16 }}>{ic}</span>
              <span style={R.statN}>{n}</span>
              <span style={R.statL}>{l}</span>
            </div>
          ))}
        </div>

        {/* ── Tech stack ── */}
        <div style={R.techRow} className="fadeIn">
          {["FLAN-T5","LoRA Adapter","FastAPI","MongoDB","RDKit","PyMuPDF","googletrans"].map(t => (
            <span key={t} style={R.tech}>{t}</span>
          ))}
        </div>
      </div>

      {/* ═══════════ RIGHT PANEL ═══════════ */}
      <div style={R.right}>
        {/* Depth shadow layers for fake 3D card lift */}
        <div style={{ ...R.shadow, transform:"translate(10px,10px) scale(0.98)" }} />
        <div style={{ ...R.shadow, transform:"translate(5px,5px) scale(0.99)", opacity:0.5 }} />

        <div ref={cardRef} style={{ ...R.card, transform: cardTransform }}
          onMouseMove={onCardMove} onMouseLeave={onCardLeave}>

          {/* Top glow line */}
          <div style={R.cardGlow} />

          {/* Animated scanline inside card */}
          <div style={R.cardScan} />

          {/* Corner accent brackets */}
          {[
            { top:0, left:0,  borderTop:"1.5px solid #38bdf8", borderLeft:"1.5px solid #38bdf8" },
            { top:0, right:0, borderTop:"1.5px solid #38bdf8", borderRight:"1.5px solid #38bdf8" },
            { bottom:0, left:0,  borderBottom:"1.5px solid #38bdf8", borderLeft:"1.5px solid #38bdf8" },
            { bottom:0, right:0, borderBottom:"1.5px solid #38bdf8", borderRight:"1.5px solid #38bdf8" },
          ].map((cs, i) => <div key={i} style={{ position:"absolute", width:14, height:14, ...cs }} />)}

          {/* ── Mode tabs ── */}
          <div style={R.tabs}>
            {["login","signup"].map(m => (
              <button key={m} style={{ ...R.tab, ...(mode===m ? R.tabOn : {}) }} onClick={() => switchMode(m)}>
                {m === "login" ? "🔑 Sign In" : "🚀 Sign Up"}
              </button>
            ))}
          </div>

          {/* Heading */}
          <div style={{ marginBottom:18 }}>
            <h2 style={R.title}>
              {mode==="login" ? "Welcome Back 👋" : mode==="signup" ? "Create Account 🚀" : "Reset Password 🔑"}
            </h2>
            <p style={R.titleSub}>
              {mode==="login" ? "Sign in to your Chemistry AI dashboard"
                : mode==="signup" ? "Join the KIET Chemistry AI platform"
                : "Set a new password for your account"}
            </p>
          </div>

          {/* Progress steps */}
          <div style={R.stepsRow}>
            {steps.map((s, i) => (
              <div key={s} style={{ display:"flex", alignItems:"center", gap:5, flex:1 }}>
                <div style={{ ...R.dot, background: i<=curStep ? "linear-gradient(135deg,#38bdf8,#0ea5e9)" : "rgba(255,255,255,0.08)", boxShadow: i<=curStep ? "0 0 12px rgba(56,189,248,0.5)" : "none" }}>
                  {i<curStep ? "✓" : i+1}
                </div>
                <span style={{ ...R.dotLbl, color: i<=curStep ? "#38bdf8" : "#475569" }}>{s}</span>
                {i < steps.length-1 && <div style={{ ...R.stepLine, background: i<curStep ? "linear-gradient(90deg,#38bdf8,#0ea5e9)" : "rgba(255,255,255,0.08)" }} />}
              </div>
            ))}
          </div>

          {/* Feedback */}
          {error   && <div style={R.errBox}><span>⚠️</span>{error}</div>}
          {success && <div style={R.sucBox}><span>✅</span>{success}</div>}

          {/* Email */}
          <Field label="Email Address" icon="✉">
            <input type="email" value={email} placeholder="you@university.edu"
              onChange={e => setEmail(e.target.value)}
              onKeyDown={e => e.key==="Enter" && submit()} style={R.inp} />
          </Field>

          {/* Password */}
          <Field label={mode==="forgot" ? "New Password" : "Password"} icon="🔒">
            <input type={showPass?"text":"password"} value={password}
              placeholder={mode==="forgot" ? "New secure password" : "Your password"}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key==="Enter" && submit()}
              style={{ ...R.inp, paddingRight:44 }} />
            <button style={R.eyeBtn} onClick={() => setShowPass(v => !v)}>{showPass?"🙈":"👁"}</button>
          </Field>

          {/* Strength */}
          {(mode==="signup"||mode==="forgot") && password && (
            <div style={{ display:"flex", alignItems:"center", gap:5, marginTop:-8, marginBottom:12 }}>
              {[1,2,3,4].map(lv => (
                <div key={lv} style={{ flex:1, height:3, borderRadius:3, background: lv<=pwStrength ? pwColor : "rgba(255,255,255,0.08)", transition:"background 0.4s" }} />
              ))}
              <span style={{ fontSize:10, fontWeight:700, color:pwColor, minWidth:52 }}>{pwLabel}</span>
            </div>
          )}

          {/* Confirm */}
          {(mode==="signup"||mode==="forgot") && (
            <Field label="Confirm Password" icon="🔒">
              <input type={showConfirm?"text":"password"} value={confirm}
                placeholder="Repeat password"
                onChange={e => setConfirm(e.target.value)}
                onKeyDown={e => e.key==="Enter" && submit()}
                style={{ ...R.inp, paddingRight:44, borderColor: confirm ? (confirm===password?"#22c55e":"#ef4444") : undefined }} />
              <button style={R.eyeBtn} onClick={() => setShowConfirm(v => !v)}>{showConfirm?"🙈":"👁"}</button>
              {confirm && <div style={{ position:"absolute", bottom:-18, left:0, fontSize:11, color: confirm===password?"#22c55e":"#ef4444" }}>
                {confirm===password ? "✓ Passwords match" : "✗ Don't match"}
              </div>}
            </Field>
          )}

          {/* Forgot link */}
          {mode==="login" && (
            <div style={{ textAlign:"right", marginTop:-4, marginBottom:10 }}>
              <span style={R.lnk} onClick={() => switchMode("forgot")}>Forgot password?</span>
            </div>
          )}

          {/* Submit */}
          <button className="shimmerBtn" style={{ ...R.submitBtn, opacity: loading ? 0.72 : 1 }}
            onClick={submit} disabled={loading}>
            {loading
              ? <span style={R.spin} />
              : <>{mode==="login" ? "Sign In to Dashboard" : mode==="signup" ? "Create Account" : "Update Password"}<span style={{marginLeft:8}}>→</span></>}
          </button>

          {/* Switch */}
          <p style={R.switchTxt}>
            {mode==="login"
              ? <>New here? <span style={R.lnk} onClick={() => switchMode("signup")}>Create account</span></>
              : <>Have an account? <span style={R.lnk} onClick={() => switchMode("login")}>Sign in</span></>}
          </p>
          {mode==="forgot" && (
            <p style={{ ...R.switchTxt, marginTop:4 }}>
              <span style={R.lnk} onClick={() => switchMode("login")}>← Back to Sign In</span>
            </p>
          )}

          {/* Security note */}
          <div style={R.secNote}>
            <span style={{ fontSize:11 }}>🔐</span>
            <span>Data stored locally · KIET University · JNTU Kakinada</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Field wrapper component
───────────────────────────────────────────── */
function Field({ label, icon, children }) {
  return (
    <div style={{ marginBottom:16 }}>
      <label style={{ display:"block", fontSize:11, fontWeight:700, color:"#94a3b8", marginBottom:6, letterSpacing:0.5, textTransform:"uppercase" }}>{label}</label>
      <div style={{ position:"relative", display:"flex", alignItems:"center" }}>
        <span style={{ position:"absolute", left:13, fontSize:14, zIndex:1, pointerEvents:"none", opacity:0.5 }}>{icon}</span>
        {children}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   DATA
───────────────────────────────────────────── */
const FEATS = [
  { i:"⚗️", t:"Point-wise Answers",   d:"Expandable structured chemistry explanations" },
  { i:"🧮", t:"Formula Database",     d:"400+ formulas with molecular weights" },
  { i:"🎨", t:"RDKit Structures",     d:"2D molecular diagrams from SMILES data" },
  { i:"🔬", t:"PDF Analysis",         d:"Summary · Quiz · Video script from PDFs" },
  { i:"🧩", t:"AI Quiz Engine",       d:"MCQ quizzes with instant scored feedback" },
  { i:"🌐", t:"11 Languages",         d:"Telugu, Hindi, Tamil, Kannada + 7 more" },
];

const ELEMENTS = [
  { n:"1",  s:"H",  nm:"Hydrogen",  c:"#38bdf8" },
  { n:"6",  s:"C",  nm:"Carbon",    c:"#94a3b8" },
  { n:"7",  s:"N",  nm:"Nitrogen",  c:"#60a5fa" },
  { n:"8",  s:"O",  nm:"Oxygen",    c:"#f87171" },
  { n:"11", s:"Na", nm:"Sodium",    c:"#fb923c" },
  { n:"17", s:"Cl", nm:"Chlorine",  c:"#4ade80" },
  { n:"26", s:"Fe", nm:"Iron",      c:"#f59e0b" },
  { n:"79", s:"Au", nm:"Gold",      c:"#fbbf24" },
  { n:"47", s:"Ag", nm:"Silver",    c:"#cbd5e1" },
];

const FLOAT_MOLS = [
  { f:"H₂O",  mw:"18 g/mol",  top:8,  left:4,  c:"#38bdf8" },
  { f:"CO₂",  mw:"44 g/mol",  top:18, left:88, c:"#94a3b8" },
  { f:"NaCl", mw:"58 g/mol",  top:74, left:5,  c:"#fb923c" },
  { f:"C₆H₆", mw:"78 g/mol", top:78, left:86, c:"#a78bfa" },
];

const STATS = [
  { n:"50K+", l:"Students",  ic:"👥" },
  { n:"500+", l:"Reactions", ic:"⚗️" },
  { n:"11",   l:"Languages", ic:"🌐" },
  { n:"99%",  l:"Accuracy",  ic:"🎯" },
];

/* ─────────────────────────────────────────────
   CSS STRING (injected via <style>)
───────────────────────────────────────────── */
const CSS = `
  @keyframes blink     { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes spin      { to{transform:rotate(360deg)} }
  @keyframes floatUp   { 0%,100%{transform:translateY(0px)} 50%{transform:translateY(-9px)} }
  @keyframes orb1anim  { 0%,100%{transform:scale(1) translate(0,0)} 50%{transform:scale(1.1) translate(18px,-14px)} }
  @keyframes orb2anim  { 0%,100%{transform:scale(1) translate(0,0)} 50%{transform:scale(1.07) translate(-14px,12px)} }
  @keyframes shimmer   { 0%{background-position:200% center} 100%{background-position:-200% center} }
  @keyframes scanMove  { 0%{top:-50%} 100%{top:130%} }
  @keyframes glowPulse { 0%,100%{box-shadow:0 30px 90px rgba(0,0,0,0.85),0 0 40px rgba(56,189,248,0.1),inset 0 1px 0 rgba(255,255,255,0.07)} 50%{box-shadow:0 30px 90px rgba(0,0,0,0.85),0 0 60px rgba(56,189,248,0.22),inset 0 1px 0 rgba(255,255,255,0.07)} }
  @keyframes fadeIn    { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
  @keyframes slideR    { from{opacity:0;transform:translateX(-24px)} to{opacity:1;transform:translateX(0)} }
  @keyframes cardScan  { 0%{top:-60px} 100%{top:110%} }

  .fadeIn { animation:fadeIn 0.7s ease both; }
  .slideR { animation:slideR 0.7s ease both; }

  .shimmerBtn {
    background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 40%, #818cf8 80%, #0ea5e9 100%) !important;
    background-size: 300% auto !important;
    animation: shimmer 3.5s linear infinite !important;
  }
  .shimmerBtn:hover:not(:disabled) {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(56,189,248,0.55), 0 0 0 1px rgba(56,189,248,0.3) !important;
  }
  .shimmerBtn:active:not(:disabled) { transform:translateY(0) !important; }

  input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 1000px rgba(8,12,28,0.95) inset !important;
    -webkit-text-fill-color: #f1f5f9 !important; caret-color:#f1f5f9;
  }
  input:focus {
    border-color: rgba(56,189,248,0.7) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.14), 0 0 14px rgba(56,189,248,0.18) !important;
    outline:none;
  }
  button:hover:not(:disabled) { filter:brightness(1.1); }
  ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-thumb{background:rgba(56,189,248,0.3);border-radius:3px}
`;

/* ─────────────────────────────────────────────
   STYLES
───────────────────────────────────────────── */
const R = {
  root: {
    minHeight:"100vh", display:"flex", alignItems:"stretch",
    background:"#020617", fontFamily:"'Segoe UI',system-ui,sans-serif",
    position:"relative", overflow:"hidden",
  },
  canvas: { position:"fixed", inset:0, zIndex:0, pointerEvents:"none" },
  bg: {
    position:"fixed", inset:0, zIndex:0, pointerEvents:"none",
    background:`
      radial-gradient(ellipse 75% 55% at 12% 28%, rgba(56,189,248,0.08) 0%, transparent 60%),
      radial-gradient(ellipse 55% 45% at 88% 72%, rgba(99,102,241,0.07) 0%, transparent 60%),
      radial-gradient(ellipse 35% 25% at 50% 8%,  rgba(6,182,212,0.04) 0%, transparent 50%),
      linear-gradient(180deg, #020617 0%, #060f26 50%, #020617 100%)
    `,
  },
  grid: {
    position:"fixed", inset:0, zIndex:0, pointerEvents:"none",
    backgroundImage:`linear-gradient(rgba(56,189,248,0.022) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,0.022) 1px,transparent 1px)`,
    backgroundSize:"55px 55px",
    maskImage:"radial-gradient(ellipse 85% 85% at 50% 50%,black 0%,transparent 100%)",
  },
  scanline: {
    position:"fixed", left:0, right:0, height:"42%", zIndex:0, pointerEvents:"none",
    background:"linear-gradient(180deg,transparent 0%,rgba(56,189,248,0.013) 50%,transparent 100%)",
    animation:"scanMove 9s linear infinite",
  },
  orb1: { position:"fixed", width:620, height:620, borderRadius:"50%", zIndex:0, pointerEvents:"none", background:"radial-gradient(circle,rgba(56,189,248,0.11) 0%,transparent 70%)", top:-220, left:-170, filter:"blur(90px)", animation:"orb1anim 13s ease-in-out infinite" },
  orb2: { position:"fixed", width:520, height:520, borderRadius:"50%", zIndex:0, pointerEvents:"none", background:"radial-gradient(circle,rgba(99,102,241,0.09) 0%,transparent 70%)", bottom:-160, right:280, filter:"blur(80px)", animation:"orb2anim 16s ease-in-out infinite" },
  orb3: { position:"fixed", width:320, height:320, borderRadius:"50%", zIndex:0, pointerEvents:"none", background:"radial-gradient(circle,rgba(34,197,94,0.06) 0%,transparent 70%)", top:"42%", right:"9%", filter:"blur(70px)", animation:"orb1anim 20s ease-in-out infinite reverse" },

  /* LEFT */
  left: {
    flex:1,padding:"70px 50px 48px 58px", position:"relative", zIndex:2,
    display:"flex", flexDirection:"column", justifyContent:"center",
    gap:0, transition:"transform 0.12s ease-out",
    overflowY:"auto", maxHeight:"100vh",
  },

  badge: {
    display:"inline-flex", alignItems:"center", gap:8,
    background:"rgba(56,189,248,0.07)", border:"1px solid rgba(56,189,248,0.22)",
    borderRadius:40, padding:"6px 15px 6px 9px", marginBottom:26, alignSelf:"flex-start",
  },
  badgeText: { fontSize:13, fontWeight:800, color:"#38bdf8", letterSpacing:0.4 },
  badgePill: { fontSize:9,  fontWeight:700, color:"#475569", marginLeft:4, letterSpacing:0.5 },

  h1: {
  fontSize: 44,
  fontWeight: 900,
  lineHeight: 1.2,
  color: "#fff",
  margin: "0 0 18px",
  letterSpacing: -1,
},

typed: {
  display: "block",
  minHeight: "1.2em",
  fontWeight: 900,

  background: "linear-gradient(120deg, #38bdf8, #818cf8, #34d399)",
  backgroundSize: "200% auto",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",

  textShadow: "0 0 18px rgba(56,189,248,0.35)",
  animation: "shimmer 4s linear infinite",
},

withRow: {
  display: "flex",
  alignItems: "center",
  gap: 10,
  marginTop: 6,
},

hlWith: {
  color: "#64748b",
  fontWeight: 400,
  fontSize: 22,
},

hlAI: {
  fontSize: 34,
  fontWeight: 900,
  background: "linear-gradient(120deg,#38bdf8,#818cf8,#34d399)",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
},

  sub: { fontSize:13, color:"#94a3b8", lineHeight:1.72, maxWidth:490, margin:"0 0 20px" },

  /* 3D atom */
  atomWrap: { position:"relative", width:"100%", maxWidth:400, height:210, marginBottom:18 },
  threeCv: {
    width:"100%", height:"100%", borderRadius:14,
    background:"radial-gradient(ellipse at center,rgba(56,189,248,0.04) 0%,transparent 70%)",
  },
  atomLbl: { position:"absolute", bottom:5, left:"50%", transform:"translateX(-50%)", fontSize:9, color:"#38bdf8", fontWeight:700, letterSpacing:1.3, textTransform:"uppercase", opacity:0.55 },
  floatMol: {
    position:"absolute", display:"flex", flexDirection:"column", alignItems:"center",
    padding:"5px 8px", borderRadius:8, border:"1px solid",
    animation:"floatUp 3.2s ease-in-out infinite", lineHeight:1.2,
  },

  /* Feature grid */
  featGrid: { display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8, marginBottom:16 },
  featCard: {
    padding:"10px 11px", borderRadius:10,
    background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.07)",
    display:"flex", flexDirection:"column", gap:2, cursor:"default", transition:"all 0.22s ease",
  },

  /* Periodic strip */
  elemStrip: { display:"flex", gap:5, marginBottom:16, flexWrap:"wrap" },
  elemCell: {
    display:"flex", flexDirection:"column", alignItems:"center",
    padding:"5px 6px", borderRadius:7, border:"1px solid", minWidth:36, cursor:"default",
  },

  /* Stats */
  stats: { display:"flex", gap:10, marginBottom:14 },
  statBox: { display:"flex", flexDirection:"column", alignItems:"center", padding:"9px 14px", borderRadius:11, background:"rgba(56,189,248,0.05)", border:"1px solid rgba(56,189,248,0.12)", gap:2 },
  statN: { fontSize:20, fontWeight:900, color:"#38bdf8", lineHeight:1 },
  statL: { fontSize:10, color:"#64748b", fontWeight:600 },

  /* Tech */
  techRow: { display:"flex", flexWrap:"wrap", gap:6 },
  tech: { padding:"3px 10px", borderRadius:18, fontSize:10, fontWeight:700, background:"rgba(56,189,248,0.06)", border:"1px solid rgba(56,189,248,0.18)", color:"#7dd3fc" },

  /* RIGHT */
  right: { width:455, padding:"36px 32px", position:"relative", zIndex:2, display:"flex", alignItems:"center", justifyContent:"center" },
  shadow: { position:"absolute", inset:0, borderRadius:24, zIndex:-1, background:"rgba(56,189,248,0.05)", border:"1px solid rgba(56,189,248,0.09)", backdropFilter:"blur(8px)" },
  card: {
    width:"100%", background:"rgba(10,13,30,0.9)", backdropFilter:"blur(36px)",
    border:"1px solid rgba(255,255,255,0.08)", borderRadius:24,
    padding:"32px 28px 24px", position:"relative", overflow:"hidden",
    transition:"transform 0.18s ease, box-shadow 0.18s ease",
    animation:"glowPulse 4.5s ease-in-out infinite",
  },
  cardGlow: { position:"absolute", top:0, left:"9%", right:"9%", height:1.5, background:"linear-gradient(90deg,transparent,#38bdf8 30%,#818cf8 70%,transparent)", borderRadius:"0 0 3px 3px" },
  cardScan: { position:"absolute", left:0, right:0, height:70, pointerEvents:"none", background:"linear-gradient(180deg,transparent,rgba(56,189,248,0.018),transparent)", animation:"cardScan 7s linear infinite" },

  /* Tabs */
  tabs: { display:"flex", background:"rgba(255,255,255,0.04)", borderRadius:10, padding:3, marginBottom:20, gap:3 },
  tab: { flex:1, padding:"8px 0", border:"none", borderRadius:8, background:"transparent", color:"#64748b", fontSize:12, fontWeight:600, cursor:"pointer", transition:"all 0.25s" },
  tabOn: { background:"rgba(56,189,248,0.12)", color:"#38bdf8", boxShadow:"0 0 14px rgba(56,189,248,0.18)", border:"1px solid rgba(56,189,248,0.25)" },

  title: { fontSize:20, fontWeight:800, color:"#f1f5f9", margin:"0 0 5px" },
  titleSub: { fontSize:12, color:"#64748b", margin:0, lineHeight:1.5 },

  /* Steps */
  stepsRow: { display:"flex", alignItems:"center", marginBottom:16 },
  dot: { width:23, height:23, borderRadius:"50%", display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, fontWeight:700, color:"#fff", flexShrink:0, transition:"all 0.3s" },
  dotLbl: { fontSize:9, fontWeight:600, whiteSpace:"nowrap", transition:"color 0.3s" },
  stepLine: { flex:1, height:2, borderRadius:2, margin:"0 5px", transition:"background 0.3s" },

  /* Feedback */
  errBox: { background:"rgba(239,68,68,0.07)", border:"1px solid rgba(239,68,68,0.22)", borderRadius:9, padding:"9px 12px", color:"#fca5a5", fontSize:12, marginBottom:12, display:"flex", alignItems:"center", gap:7 },
  sucBox: { background:"rgba(34,197,94,0.07)", border:"1px solid rgba(34,197,94,0.22)", borderRadius:9, padding:"9px 12px", color:"#86efac", fontSize:12, marginBottom:12, display:"flex", alignItems:"center", gap:7 },

  /* Inputs */
  inp: { width:"100%", padding:"11px 16px 11px 40px", background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:10, color:"#f1f5f9", fontSize:13, outline:"none", boxSizing:"border-box", transition:"all 0.3s", fontFamily:"inherit" },
  eyeBtn: { position:"absolute", right:11, background:"transparent", border:"none", cursor:"pointer", color:"#64748b", padding:3, fontSize:14, display:"flex", alignItems:"center" },

  /* Submit */
  submitBtn: { width:"100%", padding:"13px", border:"none", borderRadius:11, color:"#020617", fontSize:13, fontWeight:800, cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:6, marginTop:6, transition:"all 0.3s", boxSizing:"border-box" },
  spin: { width:15, height:15, border:"2px solid rgba(2,6,23,0.25)", borderTopColor:"#020617", borderRadius:"50%", display:"inline-block", animation:"spin 0.7s linear infinite" },

  lnk: { color:"#38bdf8", cursor:"pointer", fontWeight:600, fontSize:12 },
  switchTxt: { textAlign:"center", fontSize:12, color:"#64748b", marginTop:12, marginBottom:0 },
  secNote: { textAlign:"center", fontSize:10, color:"#334155", marginTop:12, display:"flex", alignItems:"center", justifyContent:"center", gap:5, padding:"7px 10px", borderRadius:7, background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.04)" },
};
