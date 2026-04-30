const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const User = require("./models/User");

const app = express();

/* ================= MIDDLEWARE ================= */
app.use(cors());
app.use(express.json());

/* ================= MONGODB LOCAL CONNECTION ================= */
mongoose
  .connect("mongodb://127.0.0.1:27017/chem_ai_auth", {
    useNewUrlParser: true,
    useUnifiedTopology: true
  })
  .then(() => console.log("✅ MongoDB Connected (Localhost)"))
  .catch((err) => console.error("❌ MongoDB Error:", err));

/* ================= SIGNUP API ================= */
app.post("/api/signup", async (req, res) => {
  const { email, password } = req.body;

  try {
    const exists = await User.findOne({ email });
    if (exists) {
      return res.status(400).json({ message: "User already exists" });
    }

    const user = new User({ email, password });
    await user.save();

    res.status(201).json({ message: "Signup successful" });
  } catch (err) {
    res.status(500).json({ message: "Server error" });
  }
});

/* ================= LOGIN API ================= */
app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: "No account found" });
    }

    if (user.password !== password) {
      return res.status(401).json({ message: "Incorrect email or password" });
    }

    res.json({ message: "Login successful" });
  } catch (err) {
    res.status(500).json({ message: "Server error" });
  }
});

/* ================= START SERVER ================= */
app.listen(5000, () => {
  console.log("🚀 Server running at http://localhost:5000");
});
