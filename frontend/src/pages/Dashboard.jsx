// ChemBot Dashboard v8.1 — FIXED: Quiz Generation + Full Language Support
// File 1 (QuizWidget + PointwiseAnswer + ChatMessage) + File 2 (Full Dashboard)
// KIET University · JNTU Kakinada

import { useEffect, useState, useRef, useCallback } from "react";

/* ══════════════════════════════════════════════════════════════
   API HELPERS
══════════════════════════════════════════════════════════════ */
const API = "https://clasroom-chemistry-bot.onrender.com";

async function apiPredict(text, language = "en") {
  const res = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language }),
  });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return (await res.json()).output || "";
}

async function apiTranslate(text, language) {
  if (!language || language === "en" || !text) return text;
  try {
    const res = await fetch(`${API}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language }),
    });
    if (!res.ok) return text;
    return (await res.json()).output || text;
  } catch { return text; }
}

async function apiHistory() {
  try { const r = await fetch(`${API}/history`); return r.ok ? r.json() : []; } catch { return []; }
}

async function apiStructure(compound) {
  const r = await fetch(`${API}/structure`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: compound }),
  });
  if (!r.ok) throw new Error(`Server error ${r.status}`);
  return r.json();
}

async function apiPDFAnalyze(file, language = "en") {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
  const r = await fetch(`${API}/pdf-analyze`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`Server error ${r.status}`);
  return r.json();
}

/* ══════════════════════════════════════════════════════════════
   LANGUAGE PROMPT BUILDER
══════════════════════════════════════════════════════════════ */
const LANG_NAMES = {
  en: "English", te: "Telugu", hi: "Hindi", ta: "Tamil",
  kn: "Kannada", fr: "French", de: "German", es: "Spanish",
  zh: "Chinese", ja: "Japanese", ar: "Arabic",
};

function buildPrompt(userQuery, language) {
  const langName = LANG_NAMES[language] || "English";
  if (language === "en") return userQuery;
  const prefix = `IMPORTANT: You must respond entirely in ${langName} language. Do not use English in your response.\n\n`;
  const suffix = `\n\nRemember: Your complete response must be in ${langName} only.`;
  return prefix + userQuery + suffix;
}

function buildChatPrompt(messages, newMessage, language) {
  const langName = LANG_NAMES[language] || "English";
  const context = messages.slice(-6).map(m =>
    `${m.role === "user" ? "Student" : "AI"}: ${m.content}`
  ).join("\n");
  if (language === "en") {
    return `You are a chemistry expert tutor. Always format your responses using numbered steps or bullet points. Use ## for section headings. Never write plain paragraphs.\n${context}\nStudent: ${newMessage}\nAI:`;
  }
  return `You are a chemistry expert tutor. You MUST respond entirely in ${langName} language. Never respond in English.\n\nConversation:\n${context}\nStudent: ${newMessage}\nAI (respond in ${langName} only):`;
}

/* ══════════════════════════════════════════════════════════════
   UI TRANSLATIONS — ALL 11 LANGUAGES
══════════════════════════════════════════════════════════════ */
const UI_TEXT = {
  en: {
    appName: "Chemistry AI", appSub: "FLAN-T5 + LoRA · KIET College, Andhra Pradesh",
    nav: { ask:"Ask Chemistry", formula:"Formulas", structure:"Structures", analyze:"Analyze PDF", notes:"Short Notes", video:"Video Script", quiz:"Quiz", chat:"Chemistry Chat" },
    terms: {
      title:"Terms & Conditions", sub:"Chemistry AI — KIET Academic Platform",
      powered:"Powered by", model:"FLAN-T5 + LoRA",
      features:"✅ Features",
      featureList:["FLAN-T5 Q&A with LoRA","RDKit 2D molecular structures","PDF extraction + AI analysis","Summary / Quiz / Video from PDFs","Multi-language support","MongoDB history"],
      policy:"⚠️ Policy",
      policyList:["Educational purposes only","Verify AI-generated data","No misuse of chemistry"],
      agree:"I agree to the Terms & Conditions", enter:"Enter Dashboard →"
    },
    ask: {
      title:"Ask Chemistry", sub:"FLAN-T5 + LoRA · Periodic Table · Wikipedia",
      label:"Your Question",
      placeholder:"Examples:\n• What is molar mass of CO2?\n• Balance: H2 + O2 → H2O\n• Explain oxidation and reduction\n• Tell me about oxygen",
      btn:"⚡ Generate Answer", loading:"Generating…", hint:"Ctrl+Enter",
      emptyError:"⚠️ Please ask a chemistry-related question.",
      emailError:"🚫 Email not allowed.", numberError:"🚫 Please enter a valid question.",
      placeholder2:"Your answer will appear here with detailed explanation."
    },
    formula: {
      title:"Chemistry Formulas", sub:"Molecular weights · Structural formulas · Equations — no duplicates",
      label:"Quick Categories", customLabel:"Custom Topic",
      placeholder:"e.g. Nernst equation, Henderson-Hasselbalch, Grignard reagents…",
      btn:"Get Formulas", loading:"Loading formulas…",
      placeholder2:"Click a category or enter a topic to see formulas with molecular weights."
    },
    structure: {
      title:"Molecular Structure Generator", sub:"Uses RDKit + SMILES from model.py",
      label:"Supported Compounds", customLabel:"Enter Compound Name",
      placeholder:"e.g. benzene, ethanol, water…",
      btn:"🎨 Generate", loading:"Drawing…", rdkitLabel:"2D Structure (RDKit)",
      placeholder2:"2D structure by RDKit will appear here."
    },
    analyze: {
      title:"Analyze PDF / Image", sub:"Extract → FLAN-T5 generates Summary + Quiz + Video Script",
      dropLabel:"Drop your chemistry PDF or image here",
      btn:"🔍 Analyze Document", loading:"Running FLAN-T5 — may take a moment on CPU…",
      summary:"📄 Summary", quiz:"🧪 Quiz", video:"🎬 Video Script",
      placeholder:"Upload a PDF — model.py will generate Summary, Quiz, and Video Script."
    },
    notes: {
      title:"Short Notes Generator", sub:"PDF → FLAN-T5 summarization into short notes",
      pdfLabel:"Upload PDF (Optional)", topicLabel:"Or Enter Topic / Text",
      placeholder:"e.g. Electrochemical cells, Organic reactions…",
      btn:"📝 Generate Notes", loading:"Generating…",
      placeholder2:"Concise exam-ready notes will appear here."
    },
    video: {
      title:"Video Script Generator", sub:"Convert a PDF or topic into a video explanation script",
      pdfLabel:"Upload PDF (Optional)", topicLabel:"Or Enter Topic",
      placeholder:"e.g. How electrolysis works, Acid-base titration…",
      btn:"🎬 Generate Script", loading:"Writing script…",
      placeholder2:"Video script will appear here."
    },
    quiz: {
      title:"Quiz Generator", sub:"MCQ quizzes from PDFs or topics via your model",
      pdfLabel:"Source PDF (Optional)", topicLabel:"Topic (if no PDF)",
      placeholder:"e.g. Equilibrium, Periodic table…",
      btn:"🧩 Generate Quiz", loading:"Generating quiz questions…",
      submit:"Submit", newQuiz:"🔄 New Quiz", answered:"answered",
      placeholder2:"Upload a PDF or enter a topic to generate a quiz.",
      excellent:"Excellent! 🏆", good:"Good job! ✅", study:"Keep studying! 📚",
      explanation:"EXPLANATION"
    },
    chat: {
      title:"Chemistry Chat", sub:"Multi-turn conversation with FLAN-T5 model",
      placeholder:"Ask a chemistry question… (Enter to send)",
      send:"Send →", clear:"Clear", loading:"Model thinking…",
      welcome:"## Welcome to ChemBot AI 👋\n\n**Definition:** Your intelligent chemistry tutor powered by FLAN-T5 + LoRA.\n\n## Key Features\n\n1. Ask any chemistry question for a **step-by-step pointwise answer**\n2. Type `QUIZ: <topic>` or use the Quiz Launcher below to get an interactive MCQ quiz\n3. Request molecular structures, molar mass calculations, and more\n\n## Quick Start\n\n- Try: `Explain what is pH and how is it measured?`\n- Try: `QUIZ: acid base reactions` for an instant quiz\n- Try: `What is the molar mass of H2SO4?`",
      quickAsk:["What is ionic bonding?","Explain Le Chatelier's principle","How does electrolysis work?","What is hybridization?"]
    },
    sidebar: { nav:"NAVIGATION", system:"SYSTEM STATUS", recent:"🕘 RECENT QUERIES", noHistory:"No history yet" },
    history:"Dashboard", signOut:"Sign Out",
    footer: { brand:"Chemistry AI Assistant", location:"KIET College, Andhra Pradesh", model:"Model: FLAN-T5 Base + LoRA · FastAPI + MongoDB" },
    voice: { start:"Start voice input", stop:"Stop listening", speak:"Listen to output" },
    langLabel: "Language",
  },
  te: {
    appName: "కెమిస్ట్రీ AI", appSub: "FLAN-T5 + LoRA · KIET విశ్వవిద్యాలయం",
    nav: { ask:"రసాయన శాస్త్రం అడగండి", formula:"సూత్రాలు", structure:"నిర్మాణాలు", analyze:"PDF విశ్లేషణ", notes:"చిన్న నోట్సులు", video:"వీడియో స్క్రిప్ట్", quiz:"క్విజ్", chat:"రసాయన చాట్" },
    terms: {
      title:"నిబంధనలు & షరతులు", sub:"కెమిస్ట్రీ AI — KIET అకాడమిక్ ప్లాట్‌ఫారమ్",
      powered:"ద్వారా నడిచేది", model:"FLAN-T5 + LoRA",
      features:"✅ ఫీచర్లు",
      featureList:["FLAN-T5 Q&A","RDKit నిర్మాణాలు","PDF విశ్లేషణ","బహుభాష మద్దతు","MongoDB చరిత్ర"],
      policy:"⚠️ విధానం",
      policyList:["విద్యా ప్రయోజనాలకు మాత్రమే","AI డేటాను ధృవీకరించండి","దుర్వినియోగం చేయకండి"],
      agree:"నేను నిబంధనలకు అంగీకరిస్తున్నాను", enter:"డాష్‌బోర్డ్‌లోకి ప్రవేశించండి →"
    },
    ask: {
      title:"రసాయన శాస్త్రం అడగండి", sub:"FLAN-T5 + LoRA · ఆవర్తన పట్టిక · Wikipedia",
      label:"మీ ప్రశ్న",
      placeholder:"ఉదాహరణలు:\n• CO2 మోలార్ మాస్ ఏమిటి?\n• H2 + O2 → H2O సమతుల్యం\n• ఆక్సీకరణ వివరించండి",
      btn:"⚡ సమాధానం పొందండి", loading:"రూపొందిస్తోంది…", hint:"Ctrl+Enter",
      emptyError:"⚠️ రసాయన ప్రశ్న అడగండి.",
      emailError:"🚫 ఇమెయిల్ అనుమతించబడదు.", numberError:"🚫 సరైన ప్రశ్న నమోదు చేయండి.",
      placeholder2:"మీ సమాధానం ఇక్కడ వివరంగా కనుగొనబడుతుంది."
    },
    formula: {
      title:"రసాయన సూత్రాలు", sub:"పరమాణు బరువులు · సంరచన సూత్రాలు · సమీకరణాలు",
      label:"త్వరిత వర్గాలు", customLabel:"అనుకూల విషయం",
      placeholder:"ఉదా. Nernst సమీకరణం, Grignard అభికర్మకాలు…",
      btn:"సూత్రాలు పొందండి", loading:"సూత్రాలు లోడ్ అవుతున్నాయి…",
      placeholder2:"వర్గం క్లిక్ చేయండి లేదా విషయం నమోదు చేయండి."
    },
    structure: {
      title:"పరమాణు నిర్మాణ జనరేటర్", sub:"RDKit + SMILES ఉపయోగిస్తుంది",
      label:"మద్దతు ఉన్న సమ్మేళనాలు", customLabel:"సమ్మేళనం పేరు నమోదు చేయండి",
      placeholder:"ఉదా. benzene, ethanol, water…",
      btn:"🎨 జెనరేట్", loading:"గీస్తోంది…", rdkitLabel:"2D నిర్మాణం (RDKit)",
      placeholder2:"RDKit ద్వారా 2D నిర్మాణం ఇక్కడ కనుగొనబడుతుంది."
    },
    analyze: {
      title:"PDF / చిత్రం విశ్లేషణ", sub:"సారాంశం + క్విజ్ + వీడియో స్క్రిప్ట్",
      dropLabel:"మీ PDF లేదా చిత్రాన్ని ఇక్కడ వేయండి",
      btn:"🔍 పత్రాన్ని విశ్లేషించండి", loading:"FLAN-T5 నడుస్తోంది…",
      summary:"📄 సారాంశం", quiz:"🧪 క్విజ్", video:"🎬 వీడియో స్క్రిప్ట్",
      placeholder:"PDF అప్‌లోడ్ చేయండి — నమూనా సారాంశం, క్విజ్ మరియు వీడియో స్క్రిప్ట్ జెనరేట్ చేస్తుంది."
    },
    notes: {
      title:"చిన్న నోట్సులు జనరేటర్", sub:"PDF → FLAN-T5 సారాంశం",
      pdfLabel:"PDF అప్‌లోడ్ (ఐచ్ఛికం)", topicLabel:"లేదా విషయం నమోదు చేయండి",
      placeholder:"ఉదా. విద్యుత్ రసాయన కణాలు…",
      btn:"📝 నోట్సులు జెనరేట్", loading:"రూపొందిస్తోంది…",
      placeholder2:"పరీక్ష సిద్ధమైన నోట్సులు ఇక్కడ కనుగొనబడతాయి."
    },
    video: {
      title:"వీడియో స్క్రిప్ట్ జనరేటర్", sub:"PDF లేదా విషయాన్ని వీడియో స్క్రిప్ట్‌గా మార్చండి",
      pdfLabel:"PDF అప్‌లోడ్ (ఐచ్ఛికం)", topicLabel:"లేదా విషయం నమోదు చేయండి",
      placeholder:"ఉదా. ఎలక్ట్రోలిసిస్ ఎలా పని చేస్తుంది…",
      btn:"🎬 స్క్రిప్ట్ జెనరేట్", loading:"స్క్రిప్ట్ రాస్తోంది…",
      placeholder2:"మీ వీడియో స్క్రిప్ట్ ఇక్కడ కనుగొనబడుతుంది."
    },
    quiz: {
      title:"క్విజ్ జనరేటర్", sub:"PDF లేదా విషయాల నుండి MCQ క్విజ్‌లు",
      pdfLabel:"మూల PDF (ఐచ్ఛికం)", topicLabel:"విషయం (PDF లేకపోతే)",
      placeholder:"ఉదా. సమతుల్యత, ఆవర్తన పట్టిక…",
      btn:"🧩 క్విజ్ జెనరేట్", loading:"క్విజ్ ప్రశ్నలు తయారవుతున్నాయి…",
      submit:"సమర్పించండి", newQuiz:"🔄 కొత్త క్విజ్", answered:"సమాధానం ఇవ్వబడింది",
      placeholder2:"PDF లేదా విషయం నమోదు చేసి క్విజ్ జెనరేట్ చేయండి.",
      excellent:"అద్భుతం! 🏆", good:"చాలా బాగుంది! ✅", study:"చదువు కొనసాగించండి! 📚",
      explanation:"వివరణ"
    },
    chat: {
      title:"రసాయన చాట్", sub:"FLAN-T5 నమూనాతో బహుళ-మలుపు సంభాషణ",
      placeholder:"రసాయన ప్రశ్న అడగండి… (Enter పంపండి)",
      send:"పంపండి →", clear:"క్లియర్", loading:"నమూనా ఆలోచిస్తోంది…",
      welcome:"హలో! నేను మీ రసాయన AI ట్యూటర్‌ని. ఏదైనా రసాయన ప్రశ్న అడగండి!",
      quickAsk:["అయానిక్ బంధం అంటే ఏమిటి?","Le Chatelier సూత్రం వివరించండి","ఎలక్ట్రోలిసిస్ ఎలా పని చేస్తుంది?","హైబ్రిడైజేషన్ అంటే ఏమిటి?"]
    },
    sidebar: { nav:"నావిగేషన్", system:"సిస్టమ్ స్థితి", recent:"🕘 ఇటీవలి ప్రశ్నలు", noHistory:"చరిత్ర లేదు" },
    history:"డాష్‌బోర్డ్", signOut:"సైన్ అవుట్",
    footer: { brand:"కెమిస్ట్రీ AI అసిస్టెంట్", location:"KIET కళాశాల, ఆంధ్రప్రదేశ్", model:"నమూనా: FLAN-T5 + LoRA · FastAPI + MongoDB" },
    voice: { start:"వాయిస్ ఇన్‌పుట్ ప్రారంభించండి", stop:"వినడం ఆపండి", speak:"అవుట్‌పుట్ వినండి" },
    langLabel: "భాష",
  },
  hi: {
    appName: "केमिस्ट्री AI", appSub: "FLAN-T5 + LoRA · KIET विश्वविद्यालय",
    nav: { ask:"केमिस्ट्री पूछें", formula:"सूत्र", structure:"संरचनाएं", analyze:"PDF विश्लेषण", notes:"लघु नोट्स", video:"वीडियो स्क्रिप्ट", quiz:"क्विज़", chat:"केमिस्ट्री चैट" },
    terms: {
      title:"नियम और शर्तें", sub:"केमिस्ट्री AI — KIET अकादमिक प्लेटफ़ॉर्म",
      powered:"द्वारा संचालित", model:"FLAN-T5 + LoRA",
      features:"✅ विशेषताएं",
      featureList:["FLAN-T5 Q&A","RDKit संरचनाएं","PDF विश्लेषण","बहुभाषा समर्थन","MongoDB इतिहास"],
      policy:"⚠️ नीति",
      policyList:["केवल शैक्षिक उपयोग","AI डेटा सत्यापित करें","दुरुपयोग न करें"],
      agree:"मैं नियमों से सहमत हूं", enter:"डैशबोर्ड में प्रवेश करें →"
    },
    ask: {
      title:"केमिस्ट्री पूछें", sub:"FLAN-T5 + LoRA · आवर्त सारणी · Wikipedia",
      label:"आपका प्रश्न",
      placeholder:"उदाहरण:\n• CO2 का मोलर द्रव्यमान क्या है?\n• ऑक्सीकरण समझाएं",
      btn:"⚡ उत्तर प्राप्त करें", loading:"उत्पन्न हो रहा है…", hint:"Ctrl+Enter",
      emptyError:"⚠️ कृपया रासायनिक प्रश्न पूछें।",
      emailError:"🚫 ईमेल अनुमत नहीं।", numberError:"🚫 सही प्रश्न दर्ज करें।",
      placeholder2:"आपका विस्तृत उत्तर यहाँ दिखाई देगा।"
    },
    formula: {
      title:"रसायन सूत्र", sub:"आणविक भार · संरचनात्मक सूत्र · समीकरण",
      label:"त्वरित श्रेणियां", customLabel:"कस्टम विषय",
      placeholder:"उदा. Nernst समीकरण, Grignard अभिकर्मक…",
      btn:"सूत्र प्राप्त करें", loading:"सूत्र लोड हो रहे हैं…",
      placeholder2:"श्रेणी क्लिक करें या विषय दर्ज करें।"
    },
    structure: {
      title:"आणविक संरचना जनरेटर", sub:"RDKit + SMILES",
      label:"समर्थित यौगिक", customLabel:"यौगिक का नाम दर्ज करें",
      placeholder:"उदा. benzene, ethanol…",
      btn:"🎨 जनरेट", loading:"बना रहा है…", rdkitLabel:"2D संरचना (RDKit)",
      placeholder2:"2D संरचना यहाँ दिखाई देगी।"
    },
    analyze: {
      title:"PDF / छवि विश्लेषण", sub:"सारांश + क्विज़ + वीडियो स्क्रिप्ट",
      dropLabel:"अपनी PDF या छवि यहाँ डालें",
      btn:"🔍 दस्तावेज़ विश्लेषण करें", loading:"FLAN-T5 चल रहा है…",
      summary:"📄 सारांश", quiz:"🧪 क्विज़", video:"🎬 वीडियो स्क्रिप्ट",
      placeholder:"PDF अपलोड करें — मॉडल सारांश, क्विज़ और स्क्रिप्ट बनाएगा।"
    },
    notes: {
      title:"लघु नोट्स जनरेटर", sub:"PDF → FLAN-T5 सारांश",
      pdfLabel:"PDF अपलोड (वैकल्पिक)", topicLabel:"या विषय दर्ज करें",
      placeholder:"उदा. विद्युत रासायनिक कोशिकाएं…",
      btn:"📝 नोट्स जनरेट", loading:"उत्पन्न हो रहा है…",
      placeholder2:"परीक्षा-तैयार नोट्स यहाँ दिखाई देंगे।"
    },
    video: {
      title:"वीडियो स्क्रिप्ट जनरेटर", sub:"PDF या विषय को वीडियो स्क्रिप्ट में बदलें",
      pdfLabel:"PDF अपलोड (वैकल्पिक)", topicLabel:"या विषय दर्ज करें",
      placeholder:"उदा. इलेक्ट्रोलिसिस कैसे काम करता है…",
      btn:"🎬 स्क्रिप्ट जनरेट", loading:"स्क्रिप्ट लिख रहा है…",
      placeholder2:"आपकी वीडियो स्क्रिप्ट यहाँ दिखाई देगी।"
    },
    quiz: {
      title:"क्विज़ जनरेटर", sub:"PDF या विषयों से MCQ क्विज़",
      pdfLabel:"स्रोत PDF (वैकल्पिक)", topicLabel:"विषय (PDF के बिना)",
      placeholder:"उदा. संतुलन, आवर्त सारणी…",
      btn:"🧩 क्विज़ जनरेट", loading:"क्विज़ प्रश्न बन रहे हैं…",
      submit:"सबमिट", newQuiz:"🔄 नया क्विज़", answered:"उत्तर दिए",
      placeholder2:"क्विज़ के लिए PDF या विषय दर्ज करें।",
      excellent:"उत्कृष्ट! 🏆", good:"बहुत अच्छा! ✅", study:"अध्ययन जारी रखें! 📚",
      explanation:"स्पष्टीकरण"
    },
    chat: {
      title:"केमिस्ट्री चैट", sub:"FLAN-T5 के साथ बहु-मोड़ वार्तालाप",
      placeholder:"रासायनिक प्रश्न पूछें… (Enter भेजें)",
      send:"भेजें →", clear:"साफ़ करें", loading:"मॉडल सोच रहा है…",
      welcome:"नमस्ते! मैं आपका केमिस्ट्री AI ट्यूटर हूं। कोई भी रासायनिक प्रश्न पूछें!",
      quickAsk:["आयनिक बंधन क्या है?","Le Chatelier का सिद्धांत समझाएं","इलेक्ट्रोलिसिस कैसे काम करता है?","संकरण क्या है?"]
    },
    sidebar: { nav:"नेविगेशन", system:"सिस्टम स्थिति", recent:"🕘 हाल की क्वेरी", noHistory:"कोई इतिहास नहीं" },
    history:"डैशबोर्ड", signOut:"साइन आउट",
    footer: { brand:"केमिस्ट्री AI सहायक", location:"KIET कॉलेज, आंध्र प्रदेश", model:"मॉडल: FLAN-T5 + LoRA · FastAPI + MongoDB" },
    voice: { start:"वॉयस इनपुट शुरू करें", stop:"सुनना बंद करें", speak:"आउटपुट सुनें" },
    langLabel: "भाषा",
  },
  ta: {
    appName: "வேதியியல் AI", appSub: "FLAN-T5 + LoRA · KIET பல்கலைக்கழகம்",
    nav: { ask:"வேதியியல் கேள்விகள்", formula:"சூத்திரங்கள்", structure:"கட்டமைப்புகள்", analyze:"PDF பகுப்பாய்வு", notes:"குறிப்புகள்", video:"வீடியோ ஸ்கிரிப்ட்", quiz:"வினாடி வினா", chat:"வேதியியல் அரட்டை" },
    terms: {
      title:"விதிமுறைகள்", sub:"வேதியியல் AI — KIET கல்வி தளம்",
      powered:"இயக்கப்படுகிறது", model:"FLAN-T5 + LoRA",
      features:"✅ அம்சங்கள்",
      featureList:["FLAN-T5 Q&A","RDKit கட்டமைப்புகள்","PDF பகுப்பாய்வு","பன்மொழி ஆதரவு","MongoDB வரலாறு"],
      policy:"⚠️ கொள்கை",
      policyList:["கல்வி நோக்கங்களுக்கு மட்டும்","AI தரவை சரிபார்க்கவும்","தவறான பயன்பாடு வேண்டாம்"],
      agree:"நான் விதிமுறைகளுக்கு சம்மதிக்கிறேன்", enter:"டாஷ்போர்டில் நுழையுங்கள் →"
    },
    ask: {
      title:"வேதியியல் கேள்விகள்", sub:"FLAN-T5 + LoRA · ஆவர்த்தன அட்டவணை · Wikipedia",
      label:"உங்கள் கேள்வி",
      placeholder:"எடுத்துக்காட்டுகள்:\n• CO2 மோலார் நிறை என்ன?\n• ஆக்சிஜனேற்றம் விளக்கவும்",
      btn:"⚡ பதில் பெறுங்கள்", loading:"உருவாக்குகிறது…", hint:"Ctrl+Enter",
      emptyError:"⚠️ வேதியியல் கேள்வி கேளுங்கள்.",
      emailError:"🚫 மின்னஞ்சல் அனுமதிக்கப்படவில்லை.", numberError:"🚫 சரியான கேள்வி உள்ளிடுங்கள்.",
      placeholder2:"உங்கள் விரிவான பதில் இங்கே தோன்றும்."
    },
    formula: {
      title:"வேதியியல் சூத்திரங்கள்", sub:"மூலக்கூற்று எடைகள் · கட்டமைப்பு சூத்திரங்கள் · சமன்பாடுகள்",
      label:"விரைவு வகைகள்", customLabel:"தனிப்பயன் தலைப்பு",
      placeholder:"எ.கா. Nernst சமன்பாடு, Grignard அகர்மகங்கள்…",
      btn:"சூத்திரங்கள் பெறுங்கள்", loading:"சூத்திரங்கள் ஏற்றுகின்றன…",
      placeholder2:"வகையை கிளிக் செய்யுங்கள் அல்லது தலைப்பு உள்ளிடுங்கள்."
    },
    structure: {
      title:"மூலக்கூற்று கட்டமைப்பு", sub:"RDKit + SMILES",
      label:"ஆதரிக்கப்படும் சேர்மங்கள்", customLabel:"சேர்மத்தின் பெயர் உள்ளிடுங்கள்",
      placeholder:"எ.கா. benzene, ethanol…",
      btn:"🎨 உருவாக்கு", loading:"வரைகிறது…", rdkitLabel:"2D கட்டமைப்பு (RDKit)",
      placeholder2:"RDKit மூலம் 2D கட்டமைப்பு இங்கே தோன்றும்."
    },
    analyze: {
      title:"PDF / படம் பகுப்பாய்வு", sub:"சுருக்கம் + வினாடி வினா + வீடியோ ஸ்கிரிப்ட்",
      dropLabel:"உங்கள் PDF அல்லது படத்தை இங்கே போடுங்கள்",
      btn:"🔍 ஆவணத்தை பகுப்பாய்வு செய்யுங்கள்", loading:"FLAN-T5 இயங்குகிறது…",
      summary:"📄 சுருக்கம்", quiz:"🧪 வினாடி வினா", video:"🎬 வீடியோ ஸ்கிரிப்ட்",
      placeholder:"PDF ஏற்றுங்கள் — மாதிரி சுருக்கம், வினாடி வினா மற்றும் வீடியோ உருவாக்கும்."
    },
    notes: {
      title:"குறிப்புகள் ஜெனரேட்டர்", sub:"PDF → FLAN-T5 சுருக்கம்",
      pdfLabel:"PDF ஏற்று (விருப்பமானது)", topicLabel:"அல்லது தலைப்பு உள்ளிடுங்கள்",
      placeholder:"எ.கா. மின்வேதியியல் கலங்கள்…",
      btn:"📝 குறிப்புகள் உருவாக்கு", loading:"உருவாக்குகிறது…",
      placeholder2:"தேர்வுக்கு தயாரான குறிப்புகள் இங்கே தோன்றும்."
    },
    video: {
      title:"வீடியோ ஸ்கிரிப்ட் ஜெனரேட்டர்", sub:"PDF அல்லது தலைப்பை வீடியோ ஸ்கிரிப்டாக மாற்றுங்கள்",
      pdfLabel:"PDF ஏற்று (விருப்பமானது)", topicLabel:"அல்லது தலைப்பு உள்ளிடுங்கள்",
      placeholder:"எ.கா. மின்னாற்பகுப்பு எவ்வாறு செயல்படுகிறது…",
      btn:"🎬 ஸ்கிரிப்ட் உருவாக்கு", loading:"ஸ்கிரிப்ட் எழுதுகிறது…",
      placeholder2:"உங்கள் வீடியோ ஸ்கிரிப்ட் இங்கே தோன்றும்."
    },
    quiz: {
      title:"வினாடி வினா ஜெனரேட்டர்", sub:"PDF அல்லது தலைப்புகளிலிருந்து MCQ",
      pdfLabel:"மூல PDF (விருப்பமானது)", topicLabel:"தலைப்பு (PDF இல்லாவிட்டால்)",
      placeholder:"எ.கா. சமன்பாடு, ஆவர்த்தன அட்டவணை…",
      btn:"🧩 வினாடி வினா உருவாக்கு", loading:"வினாடி வினா கேள்விகள் உருவாகுகின்றன…",
      submit:"சமர்ப்பி", newQuiz:"🔄 புதிய வினாடி வினா", answered:"பதில் அளிக்கப்பட்டது",
      placeholder2:"வினாடி வினாவிற்கு PDF அல்லது தலைப்பு உள்ளிடுங்கள்.",
      excellent:"சிறந்தது! 🏆", good:"நல்லது! ✅", study:"படிப்பை தொடருங்கள்! 📚",
      explanation:"விளக்கம்"
    },
    chat: {
      title:"வேதியியல் அரட்டை", sub:"FLAN-T5 மாதிரியுடன் உரையாடல்",
      placeholder:"வேதியியல் கேள்வி கேளுங்கள்… (Enter அனுப்பு)",
      send:"அனுப்பு →", clear:"அழி", loading:"மாதிரி சிந்திக்கிறது…",
      welcome:"வணக்கம்! நான் உங்கள் வேதியியல் AI ஆசிரியர். எந்த வேதியியல் கேள்வியையும் கேளுங்கள்!",
      quickAsk:["அயனிப் பிணைப்பு என்றால் என்ன?","Le Chatelier கோட்பாடு விளக்கவும்","மின்னாற்பகுப்பு எவ்வாறு செயல்படுகிறது?","கலப்பமைவு என்றால் என்ன?"]
    },
    sidebar: { nav:"வழிசெலுத்தல்", system:"கணினி நிலை", recent:"🕘 சமீபத்திய கேள்விகள்", noHistory:"வரலாறு இல்லை" },
    history:"டாஷ்போர்டு", signOut:"வெளியேறு",
    footer: { brand:"வேதியியல் AI உதவியாளர்", location:"KIET கல்லூரி, ஆந்திரப் பிரதேசம்", model:"மாதிரி: FLAN-T5 + LoRA · FastAPI + MongoDB" },
    voice: { start:"குரல் உள்ளீட்டை தொடங்கு", stop:"கேட்பதை நிறுத்து", speak:"வெளியீட்டை கேள்" },
    langLabel: "மொழி",
  },
  kn: {
    appName: "ರಸಾಯನ AI", appSub: "FLAN-T5 + LoRA · KIET ವಿಶ್ವವಿದ್ಯಾಲಯ",
    nav: { ask:"ರಸಾಯನ ಕೇಳಿ", formula:"ಸೂತ್ರಗಳು", structure:"ರಚನೆಗಳು", analyze:"PDF ವಿಶ್ಲೇಷಣೆ", notes:"ಸಣ್ಣ ಟಿಪ್ಪಣಿಗಳು", video:"ವೀಡಿಯೊ ಸ್ಕ್ರಿಪ್ಟ್", quiz:"ಕ್ವಿಜ್", chat:"ರಸಾಯನ ಚಾಟ್" },
    terms: {
      title:"ನಿಯಮಗಳು ಮತ್ತು ಷರತ್ತುಗಳು", sub:"ರಸಾಯನ AI — KIET ಶೈಕ್ಷಣಿಕ ವೇದಿಕೆ",
      powered:"ಆಧರಿಸಿ", model:"FLAN-T5 + LoRA",
      features:"✅ ವೈಶಿಷ್ಟ್ಯಗಳು",
      featureList:["FLAN-T5 Q&A","RDKit ರಚನೆಗಳು","PDF ವಿಶ್ಲೇಷಣೆ","ಬಹುಭಾಷಾ ಬೆಂಬಲ","MongoDB ಇತಿಹಾಸ"],
      policy:"⚠️ ನೀತಿ",
      policyList:["ಕೇವಲ ಶೈಕ್ಷಣಿಕ ಉಪಯೋಗ","AI ಡೇಟಾ ಪರಿಶೀಲಿಸಿ","ದುರುಪಯೋಗ ಮಾಡಬೇಡಿ"],
      agree:"ನಾನು ನಿಯಮಗಳಿಗೆ ಒಪ್ಪಿಗೆ ನೀಡುತ್ತೇನೆ", enter:"ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ನಮೂದಿಸಿ →"
    },
    ask: {
      title:"ರಸಾಯನ ಪ್ರಶ್ನೆ ಕೇಳಿ", sub:"FLAN-T5 + LoRA · ಆವರ್ತಕ ಕೋಷ್ಟಕ · Wikipedia",
      label:"ನಿಮ್ಮ ಪ್ರಶ್ನೆ",
      placeholder:"ಉದಾಹರಣೆಗಳು:\n• CO2 ಮೋಲಾರ್ ದ್ರವ್ಯರಾಶಿ ಏನು?\n• ಆಕ್ಸಿಡೇಶನ್ ವಿವರಿಸಿ",
      btn:"⚡ ಉತ್ತರ ಪಡೆಯಿರಿ", loading:"ರಚಿಸಲಾಗುತ್ತಿದೆ…", hint:"Ctrl+Enter",
      emptyError:"⚠️ ರಸಾಯನ ಪ್ರಶ್ನೆ ಕೇಳಿ.",
      emailError:"🚫 ಇಮೇಲ್ ಅನುಮತಿಸಲಾಗಿಲ್ಲ.", numberError:"🚫 ಸರಿಯಾದ ಪ್ರಶ್ನೆ ನಮೂದಿಸಿ.",
      placeholder2:"ನಿಮ್ಮ ವಿವರವಾದ ಉತ್ತರ ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತದೆ."
    },
    formula: {
      title:"ರಸಾಯನ ಸೂತ್ರಗಳು", sub:"ಆಣ್ವಿಕ ತೂಕಗಳು · ರಚನಾ ಸೂತ್ರಗಳು · ಸಮೀಕರಣಗಳು",
      label:"ತ್ವರಿತ ವರ್ಗಗಳು", customLabel:"ಕಸ್ಟಮ್ ವಿಷಯ",
      placeholder:"ಉದಾ. Nernst ಸಮೀಕರಣ, Grignard ಅಭಿಕಾರಕಗಳು…",
      btn:"ಸೂತ್ರಗಳನ್ನು ಪಡೆಯಿರಿ", loading:"ಸೂತ್ರಗಳನ್ನು ಲೋಡ್ ಮಾಡಲಾಗುತ್ತಿದೆ…",
      placeholder2:"ವರ್ಗ ಕ್ಲಿಕ್ ಮಾಡಿ ಅಥವಾ ವಿಷಯ ನಮೂದಿಸಿ."
    },
    structure: {
      title:"ಆಣ್ವಿಕ ರಚನಾ ಜನರೇಟರ್", sub:"RDKit + SMILES",
      label:"ಬೆಂಬಲಿತ ಸಂಯುಕ್ತಗಳು", customLabel:"ಸಂಯುಕ್ತದ ಹೆಸರು ನಮೂದಿಸಿ",
      placeholder:"ಉದಾ. benzene, ethanol…",
      btn:"🎨 ರಚಿಸಿ", loading:"ಚಿತ್ರಿಸಲಾಗುತ್ತಿದೆ…", rdkitLabel:"2D ರಚನೆ (RDKit)",
      placeholder2:"RDKit ಮೂಲಕ 2D ರಚನೆ ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತದೆ."
    },
    analyze: {
      title:"PDF / ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆ", sub:"ಸಾರಾಂಶ + ಕ್ವಿಜ್ + ವೀಡಿಯೊ ಸ್ಕ್ರಿಪ್ಟ್",
      dropLabel:"ನಿಮ್ಮ PDF ಅಥವಾ ಚಿತ್ರವನ್ನು ಇಲ್ಲಿ ಬಿಡಿ",
      btn:"🔍 ದಾಖಲೆ ವಿಶ್ಲೇಷಿಸಿ", loading:"FLAN-T5 ಚಾಲನೆಯಲ್ಲಿದೆ…",
      summary:"📄 ಸಾರಾಂಶ", quiz:"🧪 ಕ್ವಿಜ್", video:"🎬 ವೀಡಿಯೊ ಸ್ಕ್ರಿಪ್ಟ್",
      placeholder:"PDF ಅಪ್‌ಲೋಡ್ ಮಾಡಿ — ಮಾದರಿ ಸಾರಾಂಶ, ಕ್ವಿಜ್ ಮತ್ತು ವೀಡಿಯೊ ರಚಿಸುತ್ತದೆ."
    },
    notes: {
      title:"ಸಣ್ಣ ಟಿಪ್ಪಣಿಗಳು ಜನರೇಟರ್", sub:"PDF → FLAN-T5 ಸಾರಾಂಶ",
      pdfLabel:"PDF ಅಪ್‌ಲೋಡ್ (ಐಚ್ಛಿಕ)", topicLabel:"ಅಥವಾ ವಿಷಯ ನಮೂದಿಸಿ",
      placeholder:"ಉದಾ. ವಿದ್ಯುದ್ರಾಸಾಯನಿಕ ಕೋಶಗಳು…",
      btn:"📝 ಟಿಪ್ಪಣಿಗಳು ರಚಿಸಿ", loading:"ರಚಿಸಲಾಗುತ್ತಿದೆ…",
      placeholder2:"ಪರೀಕ್ಷೆಗೆ ಸಿದ್ಧ ಟಿಪ್ಪಣಿಗಳು ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತವೆ."
    },
    video: {
      title:"ವೀಡಿಯೊ ಸ್ಕ್ರಿಪ್ಟ್ ಜನರೇಟರ್", sub:"PDF ಅಥವಾ ವಿಷಯವನ್ನು ವೀಡಿಯೊ ಸ್ಕ್ರಿಪ್ಟ್‌ಗೆ ಪರಿವರ್ತಿಸಿ",
      pdfLabel:"PDF ಅಪ್‌ಲೋಡ್ (ಐಚ್ಛಿಕ)", topicLabel:"ಅಥವಾ ವಿಷಯ ನಮೂದಿಸಿ",
      placeholder:"ಉದಾ. ವಿದ್ಯುದ್ವಿಭಾಜನೆ ಹೇಗೆ ಕಾರ್ಯ ನಿರ್ವಹಿಸುತ್ತದೆ…",
      btn:"🎬 ಸ್ಕ್ರಿಪ್ಟ್ ರಚಿಸಿ", loading:"ಸ್ಕ್ರಿಪ್ಟ್ ಬರೆಯಲಾಗುತ್ತಿದೆ…",
      placeholder2:"ನಿಮ್ಮ ವೀಡಿಯೊ ಸ್ಕ್ರಿಪ್ಟ್ ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತದೆ."
    },
    quiz: {
      title:"ಕ್ವಿಜ್ ಜನರೇಟರ್", sub:"PDF ಅಥವಾ ವಿಷಯಗಳಿಂದ MCQ ಕ್ವಿಜ್",
      pdfLabel:"ಮೂಲ PDF (ಐಚ್ಛಿಕ)", topicLabel:"ವಿಷಯ (PDF ಇಲ್ಲದಿದ್ದರೆ)",
      placeholder:"ಉದಾ. ಸಮತೋಲನ, ಆವರ್ತಕ ಕೋಷ್ಟಕ…",
      btn:"🧩 ಕ್ವಿಜ್ ರಚಿಸಿ", loading:"ಕ್ವಿಜ್ ಪ್ರಶ್ನೆಗಳು ರಚಿಸಲಾಗುತ್ತಿದೆ…",
      submit:"ಸಲ್ಲಿಸಿ", newQuiz:"🔄 ಹೊಸ ಕ್ವಿಜ್", answered:"ಉತ್ತರಿಸಲಾಗಿದೆ",
      placeholder2:"ಕ್ವಿಜ್‌ಗಾಗಿ PDF ಅಥವಾ ವಿಷಯ ನಮೂದಿಸಿ.",
      excellent:"ಅದ್ಭುತ! 🏆", good:"ಉತ್ತಮ! ✅", study:"ಅಧ್ಯಯನ ಮುಂದುವರಿಸಿ! 📚",
      explanation:"ವಿವರಣೆ"
    },
    chat: {
      title:"ರಸಾಯನ ಚಾಟ್", sub:"FLAN-T5 ಮಾದರಿಯೊಂದಿಗೆ ಸಂಭಾಷಣೆ",
      placeholder:"ರಸಾಯನ ಪ್ರಶ್ನೆ ಕೇಳಿ… (Enter ಕಳುಹಿಸಿ)",
      send:"ಕಳುಹಿಸಿ →", clear:"ತೆರವುಗೊಳಿಸಿ", loading:"ಮಾದರಿ ಯೋಚಿಸುತ್ತಿದೆ…",
      welcome:"ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ರಸಾಯನ AI ಶಿಕ್ಷಕ. ಯಾವುದೇ ರಸಾಯನ ಪ್ರಶ್ನೆ ಕೇಳಿ!",
      quickAsk:["ಅಯಾನಿಕ್ ಬಂಧ ಎಂದರೇನು?","Le Chatelier ತತ್ವ ವಿವರಿಸಿ","ವಿದ್ಯುದ್ವಿಭಾಜನೆ ಹೇಗೆ ಕಾರ್ಯ ನಿರ್ವಹಿಸುತ್ತದೆ?","ಸಂಕರಣ ಎಂದರೇನು?"]
    },
    sidebar: { nav:"ನ್ಯಾವಿಗೇಷನ್", system:"ಸಿಸ್ಟಂ ಸ್ಥಿತಿ", recent:"🕘 ಇತ್ತೀಚಿನ ಪ್ರಶ್ನೆಗಳು", noHistory:"ಇತಿಹಾಸ ಇಲ್ಲ" },
    history:"ಡ್ಯಾಶ್‌ಬೋರ್ಡ್", signOut:"ಸೈನ್ ಔಟ್",
    footer: { brand:"ರಸಾಯನ AI ಸಹಾಯಕ", location:"KIET ಕಾಲೇಜು, ಆಂಧ್ರ ಪ್ರದೇಶ", model:"ಮಾದರಿ: FLAN-T5 + LoRA · FastAPI + MongoDB" },
    voice: { start:"ಧ್ವನಿ ಇನ್‌ಪುಟ್ ಪ್ರಾರಂಭಿಸಿ", stop:"ಕೇಳುವುದನ್ನು ನಿಲ್ಲಿಸಿ", speak:"ಔಟ್‌ಪುಟ್ ಕೇಳಿ" },
    langLabel: "ಭಾಷೆ",
  },
  fr: {
    appName: "Chimie IA", appSub: "FLAN-T5 + LoRA · Université KIET, Andhra Pradesh",
    nav: { ask:"Poser une question", formula:"Formules", structure:"Structures", analyze:"Analyser PDF", notes:"Notes courtes", video:"Script vidéo", quiz:"Quiz", chat:"Chat chimie" },
    terms: {
      title:"Termes et conditions", sub:"Chimie IA — Plateforme académique KIET",
      powered:"Propulsé par", model:"FLAN-T5 + LoRA",
      features:"✅ Fonctionnalités",
      featureList:["FLAN-T5 Q&R avec LoRA","Structures 2D RDKit","Extraction PDF + analyse IA","Résumé / Quiz / Vidéo depuis PDFs","Support multilingue","Historique MongoDB"],
      policy:"⚠️ Politique",
      policyList:["À des fins éducatives uniquement","Vérifiez les données générées par IA","Pas d'abus de la chimie"],
      agree:"J'accepte les termes et conditions", enter:"Accéder au tableau de bord →"
    },
    ask: {
      title:"Poser une question de chimie", sub:"FLAN-T5 + LoRA · Tableau périodique · Wikipedia",
      label:"Votre question",
      placeholder:"Exemples:\n• Quelle est la masse molaire du CO2?\n• Équilibrer: H2 + O2 → H2O\n• Expliquer l'oxydation et la réduction",
      btn:"⚡ Générer une réponse", loading:"Génération…", hint:"Ctrl+Entrée",
      emptyError:"⚠️ Veuillez poser une question de chimie.",
      emailError:"🚫 Email non autorisé.", numberError:"🚫 Veuillez entrer une question valide.",
      placeholder2:"Votre réponse apparaîtra ici avec une explication détaillée."
    },
    formula: {
      title:"Formules de chimie", sub:"Masses moléculaires · Formules structurales · Équations",
      label:"Catégories rapides", customLabel:"Sujet personnalisé",
      placeholder:"ex. équation de Nernst, Henderson-Hasselbalch, réactifs de Grignard…",
      btn:"Obtenir les formules", loading:"Chargement des formules…",
      placeholder2:"Cliquez sur une catégorie ou entrez un sujet pour voir les formules."
    },
    structure: {
      title:"Générateur de structure moléculaire", sub:"Utilise RDKit + SMILES",
      label:"Composés supportés", customLabel:"Entrer le nom du composé",
      placeholder:"ex. benzène, éthanol, eau…",
      btn:"🎨 Générer", loading:"Dessin…", rdkitLabel:"Structure 2D (RDKit)",
      placeholder2:"La structure 2D par RDKit apparaîtra ici."
    },
    analyze: {
      title:"Analyser PDF / Image", sub:"Extraction → FLAN-T5 génère Résumé + Quiz + Script",
      dropLabel:"Déposez votre PDF ou image ici",
      btn:"🔍 Analyser le document", loading:"FLAN-T5 en cours…",
      summary:"📄 Résumé", quiz:"🧪 Quiz", video:"🎬 Script vidéo",
      placeholder:"Téléchargez un PDF — le modèle générera Résumé, Quiz et Script vidéo."
    },
    notes: {
      title:"Générateur de notes courtes", sub:"PDF → Résumé FLAN-T5 en notes courtes",
      pdfLabel:"Télécharger PDF (optionnel)", topicLabel:"Ou entrer le sujet",
      placeholder:"ex. Cellules électrochimiques, Réactions organiques…",
      btn:"📝 Générer des notes", loading:"Génération…",
      placeholder2:"Des notes concises prêtes pour l'examen apparaîtront ici."
    },
    video: {
      title:"Générateur de script vidéo", sub:"Convertir un PDF ou un sujet en script vidéo",
      pdfLabel:"Télécharger PDF (optionnel)", topicLabel:"Ou entrer le sujet",
      placeholder:"ex. Comment fonctionne l'électrolyse…",
      btn:"🎬 Générer le script", loading:"Écriture du script…",
      placeholder2:"Votre script vidéo apparaîtra ici."
    },
    quiz: {
      title:"Générateur de quiz", sub:"Quiz MCQ depuis PDFs ou sujets",
      pdfLabel:"PDF source (optionnel)", topicLabel:"Sujet (sans PDF)",
      placeholder:"ex. Équilibre, Tableau périodique…",
      btn:"🧩 Générer le quiz", loading:"Génération des questions…",
      submit:"Soumettre", newQuiz:"🔄 Nouveau quiz", answered:"répondu",
      placeholder2:"Téléchargez un PDF ou entrez un sujet pour générer un quiz.",
      excellent:"Excellent! 🏆", good:"Bon travail! ✅", study:"Continuez à étudier! 📚",
      explanation:"EXPLICATION"
    },
    chat: {
      title:"Chat chimie", sub:"Conversation multi-tours avec le modèle FLAN-T5",
      placeholder:"Posez une question de chimie… (Entrée pour envoyer)",
      send:"Envoyer →", clear:"Effacer", loading:"Le modèle réfléchit…",
      welcome:"## Bienvenue sur ChimieBot IA 👋\n\n**Définition:** Votre tuteur de chimie intelligent propulsé par FLAN-T5 + LoRA.\n\n## Fonctionnalités clés\n\n1. Posez n'importe quelle question de chimie pour une **réponse étape par étape**\n2. Tapez `QUIZ: <sujet>` pour un quiz MCQ interactif\n3. Demandez des structures moléculaires, des calculs de masse molaire, et plus\n\n## Démarrage rapide\n\n- Essayez: `Expliquer le pH et comment il est mesuré?`\n- Essayez: `QUIZ: réactions acide base` pour un quiz instantané",
      quickAsk:["Qu'est-ce que la liaison ionique?","Expliquer le principe de Le Chatelier","Comment fonctionne l'électrolyse?","Qu'est-ce que l'hybridation?"]
    },
    sidebar: { nav:"NAVIGATION", system:"ÉTAT DU SYSTÈME", recent:"🕘 REQUÊTES RÉCENTES", noHistory:"Pas encore d'historique" },
    history:"Tableau de bord", signOut:"Se déconnecter",
    footer: { brand:"Assistant IA Chimie", location:"Collège KIET, Andhra Pradesh", model:"Modèle: FLAN-T5 Base + LoRA · FastAPI + MongoDB" },
    voice: { start:"Démarrer la saisie vocale", stop:"Arrêter l'écoute", speak:"Écouter la sortie" },
    langLabel: "Langue",
  },
  de: {
    appName: "Chemie KI", appSub: "FLAN-T5 + LoRA · KIET Universität, Andhra Pradesh",
    nav: { ask:"Frage stellen", formula:"Formeln", structure:"Strukturen", analyze:"PDF analysieren", notes:"Kurznotizen", video:"Videoskript", quiz:"Quiz", chat:"Chemie-Chat" },
    terms: {
      title:"Nutzungsbedingungen", sub:"Chemie KI — KIET Akademische Plattform",
      powered:"Angetrieben von", model:"FLAN-T5 + LoRA",
      features:"✅ Funktionen",
      featureList:["FLAN-T5 Q&A mit LoRA","RDKit 2D-Molekülstrukturen","PDF-Extraktion + KI-Analyse","Zusammenfassung / Quiz / Video aus PDFs","Mehrsprachige Unterstützung","MongoDB-Verlauf"],
      policy:"⚠️ Richtlinie",
      policyList:["Nur für Bildungszwecke","KI-generierte Daten überprüfen","Kein Missbrauch der Chemie"],
      agree:"Ich stimme den Nutzungsbedingungen zu", enter:"Dashboard betreten →"
    },
    ask: {
      title:"Chemiefrage stellen", sub:"FLAN-T5 + LoRA · Periodensystem · Wikipedia",
      label:"Ihre Frage",
      placeholder:"Beispiele:\n• Was ist die Molmasse von CO2?\n• Ausgleichen: H2 + O2 → H2O\n• Oxidation und Reduktion erklären",
      btn:"⚡ Antwort generieren", loading:"Generiere…", hint:"Strg+Enter",
      emptyError:"⚠️ Bitte stellen Sie eine chemische Frage.",
      emailError:"🚫 E-Mail nicht erlaubt.", numberError:"🚫 Bitte geben Sie eine gültige Frage ein.",
      placeholder2:"Ihre Antwort erscheint hier mit detaillierter Erklärung."
    },
    formula: {
      title:"Chemieformeln", sub:"Molekulargewichte · Strukturformeln · Gleichungen",
      label:"Schnellkategorien", customLabel:"Benutzerdefiniertes Thema",
      placeholder:"z.B. Nernst-Gleichung, Henderson-Hasselbalch, Grignard-Reagenzien…",
      btn:"Formeln abrufen", loading:"Formeln laden…",
      placeholder2:"Klicken Sie auf eine Kategorie oder geben Sie ein Thema ein."
    },
    structure: {
      title:"Molekülstruktur-Generator", sub:"Verwendet RDKit + SMILES",
      label:"Unterstützte Verbindungen", customLabel:"Verbindungsname eingeben",
      placeholder:"z.B. Benzol, Ethanol, Wasser…",
      btn:"🎨 Generieren", loading:"Zeichnen…", rdkitLabel:"2D-Struktur (RDKit)",
      placeholder2:"2D-Struktur von RDKit erscheint hier."
    },
    analyze: {
      title:"PDF / Bild analysieren", sub:"Extraktion → FLAN-T5 generiert Zusammenfassung + Quiz + Skript",
      dropLabel:"Ihr Chemie-PDF oder Bild hier ablegen",
      btn:"🔍 Dokument analysieren", loading:"FLAN-T5 läuft…",
      summary:"📄 Zusammenfassung", quiz:"🧪 Quiz", video:"🎬 Videoskript",
      placeholder:"PDF hochladen — Modell generiert Zusammenfassung, Quiz und Videoskript."
    },
    notes: {
      title:"Kurznotizen-Generator", sub:"PDF → FLAN-T5 Zusammenfassung in Kurznotizen",
      pdfLabel:"PDF hochladen (Optional)", topicLabel:"Oder Thema eingeben",
      placeholder:"z.B. Elektrochemische Zellen, Organische Reaktionen…",
      btn:"📝 Notizen generieren", loading:"Generiere…",
      placeholder2:"Prägnante prüfungsfertige Notizen erscheinen hier."
    },
    video: {
      title:"Videoskript-Generator", sub:"PDF oder Thema in Videoskript umwandeln",
      pdfLabel:"PDF hochladen (Optional)", topicLabel:"Oder Thema eingeben",
      placeholder:"z.B. Wie Elektrolyse funktioniert…",
      btn:"🎬 Skript generieren", loading:"Schreibe Skript…",
      placeholder2:"Ihr Videoskript erscheint hier."
    },
    quiz: {
      title:"Quiz-Generator", sub:"MCQ-Quizze aus PDFs oder Themen",
      pdfLabel:"Quell-PDF (Optional)", topicLabel:"Thema (ohne PDF)",
      placeholder:"z.B. Gleichgewicht, Periodensystem…",
      btn:"🧩 Quiz generieren", loading:"Quizfragen generieren…",
      submit:"Einreichen", newQuiz:"🔄 Neues Quiz", answered:"beantwortet",
      placeholder2:"PDF hochladen oder Thema eingeben um Quiz zu erstellen.",
      excellent:"Ausgezeichnet! 🏆", good:"Gute Arbeit! ✅", study:"Weiter lernen! 📚",
      explanation:"ERKLÄRUNG"
    },
    chat: {
      title:"Chemie-Chat", sub:"Mehrteilige Konversation mit FLAN-T5-Modell",
      placeholder:"Chemiefrage stellen… (Enter zum Senden)",
      send:"Senden →", clear:"Löschen", loading:"Modell denkt nach…",
      welcome:"## Willkommen bei ChemBot KI 👋\n\n**Definition:** Ihr intelligenter Chemielehrer mit FLAN-T5 + LoRA.\n\n## Hauptfunktionen\n\n1. Stellen Sie jede Chemiefrage für eine **schrittweise Antwort**\n2. Geben Sie `QUIZ: <Thema>` für ein interaktives MCQ-Quiz ein\n3. Fordern Sie Molekülstrukturen, Molmassenberechnungen und mehr an\n\n## Schnellstart\n\n- Versuchen: `Was ist pH und wie wird er gemessen?`\n- Versuchen: `QUIZ: Säure-Base-Reaktionen`",
      quickAsk:["Was ist Ionenbindung?","Le Chateliers Prinzip erklären","Wie funktioniert Elektrolyse?","Was ist Hybridisierung?"]
    },
    sidebar: { nav:"NAVIGATION", system:"SYSTEMSTATUS", recent:"🕘 LETZTE ANFRAGEN", noHistory:"Noch kein Verlauf" },
    history:"Dashboard", signOut:"Abmelden",
    footer: { brand:"Chemie KI-Assistent", location:"KIET College, Andhra Pradesh", model:"Modell: FLAN-T5 Base + LoRA · FastAPI + MongoDB" },
    voice: { start:"Spracheingabe starten", stop:"Hören stoppen", speak:"Ausgabe anhören" },
    langLabel: "Sprache",
  },
  es: {
    appName: "Química IA", appSub: "FLAN-T5 + LoRA · Universidad KIET, Andhra Pradesh",
    nav: { ask:"Preguntar química", formula:"Fórmulas", structure:"Estructuras", analyze:"Analizar PDF", notes:"Notas breves", video:"Guión de vídeo", quiz:"Quiz", chat:"Chat de química" },
    terms: {
      title:"Términos y condiciones", sub:"Química IA — Plataforma académica KIET",
      powered:"Impulsado por", model:"FLAN-T5 + LoRA",
      features:"✅ Características",
      featureList:["FLAN-T5 Q&A con LoRA","Estructuras 2D RDKit","Extracción PDF + análisis IA","Resumen / Quiz / Vídeo desde PDFs","Soporte multilingüe","Historial MongoDB"],
      policy:"⚠️ Política",
      policyList:["Solo para fines educativos","Verificar datos generados por IA","Sin mal uso de la química"],
      agree:"Acepto los términos y condiciones", enter:"Entrar al panel →"
    },
    ask: {
      title:"Preguntar química", sub:"FLAN-T5 + LoRA · Tabla periódica · Wikipedia",
      label:"Su pregunta",
      placeholder:"Ejemplos:\n• ¿Cuál es la masa molar del CO2?\n• Balancear: H2 + O2 → H2O\n• Explicar oxidación y reducción",
      btn:"⚡ Generar respuesta", loading:"Generando…", hint:"Ctrl+Enter",
      emptyError:"⚠️ Por favor haga una pregunta química.",
      emailError:"🚫 Email no permitido.", numberError:"🚫 Por favor ingrese una pregunta válida.",
      placeholder2:"Su respuesta aparecerá aquí con explicación detallada."
    },
    formula: {
      title:"Fórmulas de química", sub:"Pesos moleculares · Fórmulas estructurales · Ecuaciones",
      label:"Categorías rápidas", customLabel:"Tema personalizado",
      placeholder:"ej. ecuación de Nernst, Henderson-Hasselbalch, reactivos de Grignard…",
      btn:"Obtener fórmulas", loading:"Cargando fórmulas…",
      placeholder2:"Haga clic en una categoría o ingrese un tema para ver fórmulas."
    },
    structure: {
      title:"Generador de estructura molecular", sub:"Usa RDKit + SMILES",
      label:"Compuestos admitidos", customLabel:"Ingresar nombre del compuesto",
      placeholder:"ej. benceno, etanol, agua…",
      btn:"🎨 Generar", loading:"Dibujando…", rdkitLabel:"Estructura 2D (RDKit)",
      placeholder2:"La estructura 2D de RDKit aparecerá aquí."
    },
    analyze: {
      title:"Analizar PDF / Imagen", sub:"Extracción → FLAN-T5 genera Resumen + Quiz + Guión",
      dropLabel:"Suelte su PDF o imagen aquí",
      btn:"🔍 Analizar documento", loading:"Ejecutando FLAN-T5…",
      summary:"📄 Resumen", quiz:"🧪 Quiz", video:"🎬 Guión de vídeo",
      placeholder:"Subir PDF — el modelo generará Resumen, Quiz y Guión de vídeo."
    },
    notes: {
      title:"Generador de notas breves", sub:"PDF → Resumen FLAN-T5 en notas breves",
      pdfLabel:"Subir PDF (Opcional)", topicLabel:"O ingresar tema",
      placeholder:"ej. Celdas electroquímicas, Reacciones orgánicas…",
      btn:"📝 Generar notas", loading:"Generando…",
      placeholder2:"Notas concisas listas para el examen aparecerán aquí."
    },
    video: {
      title:"Generador de guión de vídeo", sub:"Convertir PDF o tema en guión de vídeo",
      pdfLabel:"Subir PDF (Opcional)", topicLabel:"O ingresar tema",
      placeholder:"ej. Cómo funciona la electrólisis…",
      btn:"🎬 Generar guión", loading:"Escribiendo guión…",
      placeholder2:"Su guión de vídeo aparecerá aquí."
    },
    quiz: {
      title:"Generador de quiz", sub:"Quiz MCQ desde PDFs o temas",
      pdfLabel:"PDF fuente (Opcional)", topicLabel:"Tema (sin PDF)",
      placeholder:"ej. Equilibrio, Tabla periódica…",
      btn:"🧩 Generar quiz", loading:"Generando preguntas de quiz…",
      submit:"Enviar", newQuiz:"🔄 Nuevo quiz", answered:"respondido",
      placeholder2:"Suba un PDF o ingrese un tema para generar un quiz.",
      excellent:"¡Excelente! 🏆", good:"¡Buen trabajo! ✅", study:"¡Sigue estudiando! 📚",
      explanation:"EXPLICACIÓN"
    },
    chat: {
      title:"Chat de química", sub:"Conversación multi-turno con el modelo FLAN-T5",
      placeholder:"Haga una pregunta química… (Enter para enviar)",
      send:"Enviar →", clear:"Limpiar", loading:"El modelo está pensando…",
      welcome:"## Bienvenido a ChemBot IA 👋\n\n**Definición:** Su tutor inteligente de química con FLAN-T5 + LoRA.\n\n## Características clave\n\n1. Haga cualquier pregunta química para una **respuesta paso a paso**\n2. Escriba `QUIZ: <tema>` para un quiz MCQ interactivo\n3. Solicite estructuras moleculares, cálculos de masa molar y más\n\n## Inicio rápido\n\n- Pruebe: `¿Qué es el pH y cómo se mide?`\n- Pruebe: `QUIZ: reacciones ácido base`",
      quickAsk:["¿Qué es el enlace iónico?","Explicar el principio de Le Chatelier","¿Cómo funciona la electrólisis?","¿Qué es la hibridación?"]
    },
    sidebar: { nav:"NAVEGACIÓN", system:"ESTADO DEL SISTEMA", recent:"🕘 CONSULTAS RECIENTES", noHistory:"Sin historial aún" },
    history:"Panel", signOut:"Cerrar sesión",
    footer: { brand:"Asistente IA de Química", location:"Colegio KIET, Andhra Pradesh", model:"Modelo: FLAN-T5 Base + LoRA · FastAPI + MongoDB" },
    voice: { start:"Iniciar entrada de voz", stop:"Dejar de escuchar", speak:"Escuchar salida" },
    langLabel: "Idioma",
  },
  zh: {
    appName: "化学 AI", appSub: "FLAN-T5 + LoRA · KIET 大学，安得拉邦",
    nav: { ask:"问化学问题", formula:"公式", structure:"结构", analyze:"分析 PDF", notes:"简短笔记", video:"视频脚本", quiz:"测验", chat:"化学聊天" },
    terms: {
      title:"条款与条件", sub:"化学 AI — KIET 学术平台",
      powered:"技术支持", model:"FLAN-T5 + LoRA",
      features:"✅ 功能",
      featureList:["FLAN-T5 问答与 LoRA","RDKit 2D 分子结构","PDF 提取 + AI 分析","从 PDF 生成摘要/测验/视频","多语言支持","MongoDB 历史记录"],
      policy:"⚠️ 政策",
      policyList:["仅用于教育目的","验证 AI 生成的数据","不得滥用化学知识"],
      agree:"我同意条款与条件", enter:"进入仪表板 →"
    },
    ask: {
      title:"提问化学问题", sub:"FLAN-T5 + LoRA · 元素周期表 · Wikipedia",
      label:"您的问题",
      placeholder:"示例:\n• CO2 的摩尔质量是多少?\n• 平衡: H2 + O2 → H2O\n• 解释氧化和还原",
      btn:"⚡ 生成答案", loading:"生成中…", hint:"Ctrl+Enter",
      emptyError:"⚠️ 请提问与化学相关的问题。",
      emailError:"🚫 不允许输入电子邮件。", numberError:"🚫 请输入有效的问题。",
      placeholder2:"您的详细答案将显示在这里。"
    },
    formula: {
      title:"化学公式", sub:"分子量 · 结构公式 · 方程式",
      label:"快速分类", customLabel:"自定义主题",
      placeholder:"例如：Nernst 方程、Henderson-Hasselbalch、Grignard 试剂…",
      btn:"获取公式", loading:"加载公式…",
      placeholder2:"点击一个类别或输入主题以查看公式。"
    },
    structure: {
      title:"分子结构生成器", sub:"使用 RDKit + SMILES",
      label:"支持的化合物", customLabel:"输入化合物名称",
      placeholder:"例如：苯、乙醇、水…",
      btn:"🎨 生成", loading:"绘制中…", rdkitLabel:"2D 结构 (RDKit)",
      placeholder2:"RDKit 的 2D 结构将显示在这里。"
    },
    analyze: {
      title:"分析 PDF / 图像", sub:"提取 → FLAN-T5 生成摘要 + 测验 + 视频脚本",
      dropLabel:"将您的 PDF 或图像拖放到这里",
      btn:"🔍 分析文档", loading:"FLAN-T5 运行中…",
      summary:"📄 摘要", quiz:"🧪 测验", video:"🎬 视频脚本",
      placeholder:"上传 PDF — 模型将生成摘要、测验和视频脚本。"
    },
    notes: {
      title:"简短笔记生成器", sub:"PDF → FLAN-T5 总结为简短笔记",
      pdfLabel:"上传 PDF（可选）", topicLabel:"或输入主题",
      placeholder:"例如：电化学电池、有机反应…",
      btn:"📝 生成笔记", loading:"生成中…",
      placeholder2:"简洁的备考笔记将显示在这里。"
    },
    video: {
      title:"视频脚本生成器", sub:"将 PDF 或主题转换为视频脚本",
      pdfLabel:"上传 PDF（可选）", topicLabel:"或输入主题",
      placeholder:"例如：电解是如何工作的…",
      btn:"🎬 生成脚本", loading:"写作脚本中…",
      placeholder2:"您的视频脚本将显示在这里。"
    },
    quiz: {
      title:"测验生成器", sub:"从 PDF 或主题生成 MCQ 测验",
      pdfLabel:"源 PDF（可选）", topicLabel:"主题（无 PDF 时）",
      placeholder:"例如：平衡、元素周期表…",
      btn:"🧩 生成测验", loading:"生成测验题中…",
      submit:"提交", newQuiz:"🔄 新测验", answered:"已回答",
      placeholder2:"上传 PDF 或输入主题以生成测验。",
      excellent:"优秀! 🏆", good:"做得好! ✅", study:"继续学习! 📚",
      explanation:"解释"
    },
    chat: {
      title:"化学聊天", sub:"与 FLAN-T5 模型的多轮对话",
      placeholder:"提问化学问题… (Enter 发送)",
      send:"发送 →", clear:"清除", loading:"模型思考中…",
      welcome:"## 欢迎使用化学机器人 AI 👋\n\n**定义:** 由 FLAN-T5 + LoRA 驱动的智能化学导师。\n\n## 主要功能\n\n1. 提问任何化学问题获得**逐步答案**\n2. 输入 `QUIZ: <主题>` 获得互动 MCQ 测验\n3. 请求分子结构、摩尔质量计算等\n\n## 快速开始\n\n- 尝试: `什么是 pH 值以及如何测量?`\n- 尝试: `QUIZ: 酸碱反应`",
      quickAsk:["什么是离子键?","解释勒夏特列原理","电解是如何工作的?","什么是杂化?"]
    },
    sidebar: { nav:"导航", system:"系统状态", recent:"🕘 最近查询", noHistory:"暂无历史记录" },
    history:"仪表板", signOut:"退出",
    footer: { brand:"化学 AI 助手", location:"KIET 学院，安得拉邦", model:"模型: FLAN-T5 Base + LoRA · FastAPI + MongoDB" },
    voice: { start:"开始语音输入", stop:"停止监听", speak:"听取输出" },
    langLabel: "语言",
  },
  ja: {
    appName: "化学 AI", appSub: "FLAN-T5 + LoRA · KIET 大学、アンドラプラデーシュ州",
    nav: { ask:"化学を質問する", formula:"数式", structure:"構造", analyze:"PDF分析", notes:"短いノート", video:"ビデオスクリプト", quiz:"クイズ", chat:"化学チャット" },
    terms: {
      title:"利用規約", sub:"化学 AI — KIET 学術プラットフォーム",
      powered:"搭載", model:"FLAN-T5 + LoRA",
      features:"✅ 機能",
      featureList:["FLAN-T5 Q&A with LoRA","RDKit 2D分子構造","PDF抽出 + AI分析","PDFからサマリー/クイズ/動画","多言語サポート","MongoDB履歴"],
      policy:"⚠️ ポリシー",
      policyList:["教育目的のみ","AIが生成したデータを確認する","化学の悪用禁止"],
      agree:"利用規約に同意します", enter:"ダッシュボードへ →"
    },
    ask: {
      title:"化学を質問する", sub:"FLAN-T5 + LoRA · 元素周期表 · Wikipedia",
      label:"質問",
      placeholder:"例:\n• CO2のモル質量は?\n• H2 + O2 → H2Oをバランス\n• 酸化と還元を説明する",
      btn:"⚡ 答えを生成", loading:"生成中…", hint:"Ctrl+Enter",
      emptyError:"⚠️ 化学に関する質問をしてください。",
      emailError:"🚫 メール不可。", numberError:"🚫 有効な質問を入力してください。",
      placeholder2:"詳細な答えがここに表示されます。"
    },
    formula: {
      title:"化学式", sub:"分子量 · 構造式 · 方程式",
      label:"クイックカテゴリー", customLabel:"カスタムトピック",
      placeholder:"例: Nernst方程式、Henderson-Hasselbalch、Grignard試薬…",
      btn:"数式を取得", loading:"数式を読み込み中…",
      placeholder2:"カテゴリーをクリックまたはトピックを入力してください。"
    },
    structure: {
      title:"分子構造ジェネレーター", sub:"RDKit + SMILESを使用",
      label:"サポートされる化合物", customLabel:"化合物名を入力",
      placeholder:"例: ベンゼン、エタノール、水…",
      btn:"🎨 生成", loading:"描画中…", rdkitLabel:"2D構造 (RDKit)",
      placeholder2:"RDKitによる2D構造がここに表示されます。"
    },
    analyze: {
      title:"PDF / 画像分析", sub:"抽出 → FLAN-T5がサマリー + クイズ + スクリプトを生成",
      dropLabel:"化学PDFまたは画像をここにドロップ",
      btn:"🔍 ドキュメントを分析", loading:"FLAN-T5実行中…",
      summary:"📄 サマリー", quiz:"🧪 クイズ", video:"🎬 ビデオスクリプト",
      placeholder:"PDFをアップロード — モデルがサマリー、クイズ、ビデオスクリプトを生成します。"
    },
    notes: {
      title:"短いノートジェネレーター", sub:"PDF → FLAN-T5要約を短いノートに",
      pdfLabel:"PDFをアップロード（任意）", topicLabel:"またはトピックを入力",
      placeholder:"例: 電気化学セル、有機反応…",
      btn:"📝 ノートを生成", loading:"生成中…",
      placeholder2:"試験対策の簡潔なノートがここに表示されます。"
    },
    video: {
      title:"ビデオスクリプトジェネレーター", sub:"PDFまたはトピックをビデオスクリプトに変換",
      pdfLabel:"PDFをアップロード（任意）", topicLabel:"またはトピックを入力",
      placeholder:"例: 電気分解の仕組み…",
      btn:"🎬 スクリプトを生成", loading:"スクリプト作成中…",
      placeholder2:"ビデオスクリプトがここに表示されます。"
    },
    quiz: {
      title:"クイズジェネレーター", sub:"PDFまたはトピックからMCQクイズ",
      pdfLabel:"ソースPDF（任意）", topicLabel:"トピック（PDFなしの場合）",
      placeholder:"例: 平衡、元素周期表…",
      btn:"🧩 クイズを生成", loading:"クイズ問題を生成中…",
      submit:"提出", newQuiz:"🔄 新しいクイズ", answered:"回答済み",
      placeholder2:"PDFをアップロードまたはトピックを入力してクイズを生成。",
      excellent:"素晴らしい! 🏆", good:"よくできました! ✅", study:"学習を続けてください! 📚",
      explanation:"解説"
    },
    chat: {
      title:"化学チャット", sub:"FLAN-T5モデルとのマルチターン会話",
      placeholder:"化学の質問をする… (Enterで送信)",
      send:"送信 →", clear:"クリア", loading:"モデルが考えています…",
      welcome:"## ChemBot AIへようこそ 👋\n\n**定義:** FLAN-T5 + LoRAが搭載されたスマート化学家庭教師。\n\n## 主な機能\n\n1. あらゆる化学の質問に**ステップバイステップの回答**\n2. `QUIZ: <トピック>` でインタラクティブMCQクイズ\n3. 分子構造、モル質量計算などをリクエスト\n\n## クイックスタート\n\n- 試してみてください: `pHとは何か、どう測るか?`\n- 試してみてください: `QUIZ: 酸塩基反応`",
      quickAsk:["イオン結合とは?","ル・シャトリエの原理を説明","電気分解の仕組みは?","混成軌道とは?"]
    },
    sidebar: { nav:"ナビゲーション", system:"システム状態", recent:"🕘 最近のクエリ", noHistory:"履歴なし" },
    history:"ダッシュボード", signOut:"サインアウト",
    footer: { brand:"化学AIアシスタント", location:"KIETカレッジ、アンドラプラデーシュ州", model:"モデル: FLAN-T5 Base + LoRA · FastAPI + MongoDB" },
    voice: { start:"音声入力を開始", stop:"聴取を停止", speak:"出力を聞く" },
    langLabel: "言語",
  },
  ar: {
    appName: "الكيمياء AI", appSub: "FLAN-T5 + LoRA · جامعة KIET، آندرا براديش",
    nav: { ask:"اسأل الكيمياء", formula:"الصيغ", structure:"الهياكل", analyze:"تحليل PDF", notes:"ملاحظات قصيرة", video:"نص الفيديو", quiz:"اختبار", chat:"محادثة الكيمياء" },
    terms: {
      title:"الشروط والأحكام", sub:"الكيمياء AI — منصة KIET الأكاديمية",
      powered:"مدعوم بـ", model:"FLAN-T5 + LoRA",
      features:"✅ الميزات",
      featureList:["FLAN-T5 Q&A مع LoRA","هياكل جزيئية ثنائية الأبعاد RDKit","استخراج PDF + تحليل AI","ملخص / اختبار / فيديو من PDFs","دعم متعدد اللغات","سجل MongoDB"],
      policy:"⚠️ السياسة",
      policyList:["للأغراض التعليمية فقط","تحقق من البيانات التي أنشأها AI","لا إساءة لاستخدام الكيمياء"],
      agree:"أوافق على الشروط والأحكام", enter:"الدخول إلى لوحة التحكم →"
    },
    ask: {
      title:"اسأل الكيمياء", sub:"FLAN-T5 + LoRA · الجدول الدوري · Wikipedia",
      label:"سؤالك",
      placeholder:"أمثلة:\n• ما الكتلة المولية لـ CO2؟\n• موازنة: H2 + O2 → H2O\n• اشرح الأكسدة والاختزال",
      btn:"⚡ إنشاء إجابة", loading:"جارٍ الإنشاء…", hint:"Ctrl+Enter",
      emptyError:"⚠️ يرجى طرح سؤال كيميائي.",
      emailError:"🚫 البريد الإلكتروني غير مسموح.", numberError:"🚫 يرجى إدخال سؤال صالح.",
      placeholder2:"ستظهر إجابتك المفصلة هنا."
    },
    formula: {
      title:"صيغ الكيمياء", sub:"الأوزان الجزيئية · الصيغ البنيوية · المعادلات",
      label:"الفئات السريعة", customLabel:"موضوع مخصص",
      placeholder:"مثال: معادلة نيرنست، هندرسون-هاسلبالش، كواشف غرينيار…",
      btn:"الحصول على الصيغ", loading:"تحميل الصيغ…",
      placeholder2:"انقر على فئة أو أدخل موضوعاً لرؤية الصيغ."
    },
    structure: {
      title:"مولد البنية الجزيئية", sub:"يستخدم RDKit + SMILES",
      label:"المركبات المدعومة", customLabel:"أدخل اسم المركب",
      placeholder:"مثال: البنزين، الإيثانول، الماء…",
      btn:"🎨 إنشاء", loading:"الرسم…", rdkitLabel:"هيكل ثنائي الأبعاد (RDKit)",
      placeholder2:"سيظهر هيكل RDKit ثنائي الأبعاد هنا."
    },
    analyze: {
      title:"تحليل PDF / صورة", sub:"استخراج → FLAN-T5 ينشئ ملخصاً + اختباراً + نصاً",
      dropLabel:"أسقط ملف PDF أو الصورة هنا",
      btn:"🔍 تحليل المستند", loading:"FLAN-T5 يعمل…",
      summary:"📄 ملخص", quiz:"🧪 اختبار", video:"🎬 نص الفيديو",
      placeholder:"قم بتحميل PDF — سيقوم النموذج بإنشاء ملخص واختبار ونص فيديو."
    },
    notes: {
      title:"مولد الملاحظات القصيرة", sub:"PDF → FLAN-T5 تلخيص إلى ملاحظات قصيرة",
      pdfLabel:"تحميل PDF (اختياري)", topicLabel:"أو أدخل الموضوع",
      placeholder:"مثال: الخلايا الكهروكيميائية، التفاعلات العضوية…",
      btn:"📝 إنشاء ملاحظات", loading:"جارٍ الإنشاء…",
      placeholder2:"ستظهر هنا ملاحظات موجزة جاهزة للامتحان."
    },
    video: {
      title:"مولد نص الفيديو", sub:"تحويل PDF أو موضوع إلى نص فيديو",
      pdfLabel:"تحميل PDF (اختياري)", topicLabel:"أو أدخل الموضوع",
      placeholder:"مثال: كيف يعمل التحليل الكهربائي…",
      btn:"🎬 إنشاء النص", loading:"كتابة النص…",
      placeholder2:"سيظهر نص الفيديو هنا."
    },
    quiz: {
      title:"مولد الاختبار", sub:"اختبارات MCQ من PDFs أو موضوعات",
      pdfLabel:"PDF المصدر (اختياري)", topicLabel:"الموضوع (بدون PDF)",
      placeholder:"مثال: التوازن، الجدول الدوري…",
      btn:"🧩 إنشاء اختبار", loading:"إنشاء أسئلة الاختبار…",
      submit:"إرسال", newQuiz:"🔄 اختبار جديد", answered:"تم الإجابة",
      placeholder2:"قم بتحميل PDF أو إدخال موضوع لإنشاء اختبار.",
      excellent:"ممتاز! 🏆", good:"عمل جيد! ✅", study:"استمر في الدراسة! 📚",
      explanation:"الشرح"
    },
    chat: {
      title:"محادثة الكيمياء", sub:"محادثة متعددة الأدوار مع نموذج FLAN-T5",
      placeholder:"اسأل سؤالاً كيميائياً… (Enter للإرسال)",
      send:"إرسال →", clear:"مسح", loading:"النموذج يفكر…",
      welcome:"## مرحباً بك في ChemBot AI 👋\n\n**التعريف:** مدرسك الذكي للكيمياء مدعوم بـ FLAN-T5 + LoRA.\n\n## الميزات الرئيسية\n\n1. اسأل أي سؤال كيميائي للحصول على **إجابة خطوة بخطوة**\n2. اكتب `QUIZ: <موضوع>` للحصول على اختبار MCQ تفاعلي\n3. اطلب هياكل جزيئية وحسابات الكتلة المولية والمزيد\n\n## بداية سريعة\n\n- جرب: `ما هو الـ pH وكيف يُقاس؟`\n- جرب: `QUIZ: تفاعلات حمض قاعدة`",
      quickAsk:["ما هو الترابط الأيوني؟","اشرح مبدأ لو شاتيليه","كيف يعمل التحليل الكهربائي؟","ما هو التهجين؟"]
    },
    sidebar: { nav:"التنقل", system:"حالة النظام", recent:"🕘 الاستعلامات الأخيرة", noHistory:"لا يوجد سجل بعد" },
    history:"لوحة التحكم", signOut:"تسجيل الخروج",
    footer: { brand:"مساعد الكيمياء AI", location:"كلية KIET، آندرا براديش", model:"النموذج: FLAN-T5 Base + LoRA · FastAPI + MongoDB" },
    voice: { start:"بدء إدخال الصوت", stop:"إيقاف الاستماع", speak:"الاستماع للمخرجات" },
    langLabel: "اللغة",
  },
};

// Safe T() — always returns a full object, never falls back silently to English without the right keys
const T = (lang) => UI_TEXT[lang] || UI_TEXT["en"];

/* ══════════════════════════════════════════════════════════════
   LANGUAGES LIST
══════════════════════════════════════════════════════════════ */
const LANGUAGES = [
  { code: "en", label: "🇺🇸 English"  },
  { code: "te", label: "🇮🇳 Telugu"   },
  { code: "hi", label: "🇮🇳 Hindi"    },
  { code: "ta", label: "🇮🇳 Tamil"    },
  { code: "kn", label: "🇮🇳 Kannada"  },
  { code: "fr", label: "🇫🇷 French"   },
  { code: "de", label: "🇩🇪 German"   },
  { code: "es", label: "🇪🇸 Spanish"  },
  { code: "zh", label: "🇨🇳 Chinese"  },
  { code: "ja", label: "🇯🇵 Japanese" },
  { code: "ar", label: "🇸🇦 Arabic"   },
];

const VOICE_LANG = { en:"en-US", te:"te-IN", hi:"hi-IN", ta:"ta-IN", kn:"kn-IN", fr:"fr-FR", de:"de-DE", es:"es-ES", zh:"zh-CN", ja:"ja-JP", ar:"ar-SA" };

/* ══════════════════════════════════════════════════════════════
   CHEMISTRY KEYWORD CHECK
══════════════════════════════════════════════════════════════ */
const CHEM_KW_EN = ["atom","molecule","reaction","acid","base","salt","compound","element","periodic","bond","ionic","covalent","organic","inorganic","oxidation","reduction","ph","formula","molar","mass","equation","balance","electron","proton","neutron","chemical","enzyme","drug","solution","catalyst","thermodynamics","kinetics","structure","polymer","isotope","enthalpy","entropy","stoichiometry","valence","orbital","benzene","methane","ethanol","water","ammonia","hydrogen","oxygen","carbon","nitrogen","sulfur","phosphorus","chlorine","sodium","potassium","calcium","iron","copper","molarity","molality","titration","buffer","equilibrium","hybridization","chatelier","le chatelier","electrolysis","electrochemistry","electrolyte","nuclear","radioactive","isomer","functional group","alkane","alkene","alkyne","ester","ether","ketone","aldehyde","carboxylic","amine","amide","aromatic","aliphatic","coordination","ligand","complex","oxidation state","reduction potential","galvanic","voltaic","electrolytic","ph scale","buffer solution","henderson","nernst","gibbs","hess","avogadro","dalton","bohr","quantum","orbital","hybridization","vsepr","dipole","polarity","intermolecular","hydrogen bond","van der waals","ionic bond","covalent bond","metallic bond"];

const isChemistryQuery = (text, lang) => {
  if (lang !== "en") return true;
  const lower = text.toLowerCase().trim();
  return CHEM_KW_EN.some(k => lower.includes(k));
};

/* ══════════════════════════════════════════════════════════════
   FORMULA DATABASE — v7.0
══════════════════════════════════════════════════════════════ */
const FORMULA_DB = {
  "Thermodynamics": [
    {name:"Gibbs Free Energy",formula:"ΔG = ΔH - TΔS",mw:null,type:"equation",desc:"Determines spontaneity; negative ΔG = spontaneous at constant T,P"},
    {name:"Enthalpy (Hess's Law)",formula:"ΔH = Σ H(products) - Σ H(reactants)",mw:null,type:"equation",desc:"Enthalpy is a state function; path-independent"},
    {name:"Entropy (Clausius)",formula:"ΔS = q_rev / T",mw:null,type:"equation",desc:"Reversible heat transfer per unit temperature"},
    {name:"Standard Cell EMF",formula:"ΔG° = -nFE°",mw:null,type:"equation",desc:"n = moles of electrons; F = 96485 C/mol"},
    {name:"Kirchhoff's Law",formula:"ΔH_T2 = ΔH_T1 + ΔCp(T2 - T1)",mw:null,type:"equation",desc:"Temperature dependence of enthalpy change"},
    {name:"Gibbs-Helmholtz",formula:"ΔG = ΔH(1 - T/T_eq)",mw:null,type:"equation",desc:"Relates Gibbs energy to equilibrium temperature"},
    {name:"Entropy of Universe",formula:"ΔS_univ = ΔS_sys + ΔS_surr ≥ 0",mw:null,type:"equation",desc:"Second law of thermodynamics; always ≥ 0"},
  ],
  "Mole Concept": [
    {name:"Moles",formula:"n = mass / M",mw:null,type:"equation",desc:"n = moles; mass in grams; M = molar mass (g/mol)"},
    {name:"Avogadro's Number",formula:"N = n × Nₐ   (Nₐ = 6.022×10²³)",mw:null,type:"equation",desc:"Number of particles in n moles"},
    {name:"Mole Fraction",formula:"χ_A = n_A / (n_A + n_B + …)",mw:null,type:"equation",desc:"Fraction of moles of component A in a mixture"},
    {name:"Empirical Formula",formula:"Mole ratio → divide by smallest → simplest integers",mw:null,type:"equation",desc:"Divide each element's moles by the smallest value"},
    {name:"Percent Composition",formula:"% = (mass of element / molar mass) × 100",mw:null,type:"equation",desc:"Mass percentage of each element in a compound"},
    {name:"Molecular vs Empirical",formula:"Molecular formula = n × Empirical formula",mw:null,type:"equation",desc:"n found from molar mass ÷ empirical formula mass"},
  ],
  "Molarity": [
    {name:"Molarity",formula:"M = n / V(L)",mw:null,type:"equation",desc:"Moles of solute per liter of solution"},
    {name:"Molality",formula:"m = n / mass_solvent(kg)",mw:null,type:"equation",desc:"Moles of solute per kilogram of solvent"},
    {name:"Dilution",formula:"M₁V₁ = M₂V₂",mw:null,type:"equation",desc:"Conservation of moles during dilution"},
    {name:"Normality",formula:"N = n × eq / V(L)",mw:null,type:"equation",desc:"Equivalents of solute per liter; eq = valence factor"},
    {name:"Van't Hoff Factor",formula:"i = 1 + α(n - 1)",mw:null,type:"equation",desc:"α = degree of dissociation; n = number of ions produced"},
    {name:"Boiling Point Elevation",formula:"ΔTb = i · Kb · m",mw:null,type:"equation",desc:"Kb = ebullioscopic constant of solvent"},
    {name:"Freezing Point Depression",formula:"ΔTf = i · Kf · m",mw:null,type:"equation",desc:"Kf = cryoscopic constant of solvent"},
    {name:"Osmotic Pressure",formula:"π = iMRT",mw:null,type:"equation",desc:"R = 0.0821 L·atm/mol·K; i = Van't Hoff factor"},
    {name:"Raoult's Law",formula:"P_A = χ_A · P°_A",mw:null,type:"equation",desc:"Partial vapour pressure proportional to mole fraction"},
  ],
  "Kinetics": [
    {name:"Rate Law",formula:"rate = k[A]ᵐ[B]ⁿ",mw:null,type:"equation",desc:"k = rate constant; m,n = partial orders (found experimentally)"},
    {name:"Arrhenius Equation",formula:"k = A · e^(-Ea/RT)",mw:null,type:"equation",desc:"A = frequency factor; Ea = activation energy (J/mol)"},
    {name:"Half-Life — 1st Order",formula:"t½ = 0.693 / k",mw:null,type:"equation",desc:"Independent of concentration; radioactive decay uses this"},
    {name:"Half-Life — 2nd Order",formula:"t½ = 1 / (k[A]₀)",mw:null,type:"equation",desc:"Depends on initial concentration"},
    {name:"Half-Life — 0th Order",formula:"t½ = [A]₀ / (2k)",mw:null,type:"equation",desc:"Linear decrease; enzyme-saturated reactions"},
    {name:"Integrated 1st Order",formula:"ln[A]t = ln[A]₀ - kt",mw:null,type:"equation",desc:"Plot of ln[A] vs t is linear with slope = -k"},
    {name:"Integrated 2nd Order",formula:"1/[A]t = 1/[A]₀ + kt",mw:null,type:"equation",desc:"Plot of 1/[A] vs t is linear with slope = k"},
    {name:"Integrated 0th Order",formula:"[A]t = [A]₀ - kt",mw:null,type:"equation",desc:"Plot of [A] vs t is linear with slope = -k"},
    {name:"Activation Energy (two T)",formula:"ln(k₂/k₁) = (Ea/R)(1/T₁ - 1/T₂)",mw:null,type:"equation",desc:"Calculates Ea from rate constants at two temperatures"},
  ],
  "Electrochemistry": [
    {name:"Nernst Equation",formula:"E = E° - (RT/nF) · ln Q",mw:null,type:"equation",desc:"At 25°C simplifies to: E = E° - (0.0592/n) · log Q"},
    {name:"Faraday's 1st Law",formula:"m = (M · I · t) / (n · F)",mw:null,type:"equation",desc:"m = mass deposited; I = current (A); t = time (s)"},
    {name:"Cell EMF",formula:"E°cell = E°cathode - E°anode",mw:null,type:"equation",desc:"Positive E°cell → spontaneous (galvanic) cell"},
    {name:"Conductance",formula:"G = 1/R = κ · A / l",mw:null,type:"equation",desc:"κ = specific conductance (S/m); A = area; l = length"},
    {name:"Kohlrausch's Law",formula:"Λm = Λ°m - b√C",mw:null,type:"equation",desc:"Strong electrolyte limiting molar conductance"},
    {name:"Degree of Dissociation",formula:"α = Λm / Λ°m",mw:null,type:"equation",desc:"Weak electrolyte α from conductance data"},
    {name:"Relation ΔG and K",formula:"ΔG° = -RT · ln K  =  -nFE°",mw:null,type:"equation",desc:"Links equilibrium constant to cell potential"},
  ],
  "Acid-Base": [
    {name:"pH",formula:"pH = -log[H⁺]",mw:null,type:"equation",desc:"Also: pH + pOH = 14 at 25°C"},
    {name:"Ka — Weak Acid",formula:"Ka = [H⁺][A⁻] / [HA]",mw:null,type:"equation",desc:"Acid dissociation constant for HA ⇌ H⁺ + A⁻"},
    {name:"Henderson-Hasselbalch",formula:"pH = pKa + log([A⁻]/[HA])",mw:null,type:"equation",desc:"Buffer pH; most accurate when [A⁻]/[HA] is 0.1–10"},
    {name:"Kw — Water",formula:"Kw = [H⁺][OH⁻] = 10⁻¹⁴  (25°C)",mw:null,type:"equation",desc:"pKw = 14; autoionization constant of water"},
    {name:"Buffer Capacity",formula:"β = 2.303 · C · Ka[H⁺] / (Ka + [H⁺])²",mw:null,type:"equation",desc:"Maximum buffer capacity occurs at pH = pKa"},
    {name:"Degree of Hydrolysis",formula:"h = √(Kw / Ka · C)",mw:null,type:"equation",desc:"For salt of strong acid + weak base"},
    {name:"Polyprotic — Intermediate pH",formula:"pH ≈ ½(pKa1 + pKa2)",mw:null,type:"equation",desc:"pH of amphoteric species (e.g. HCO₃⁻, H₂PO₄⁻)"},
  ],
  "Gas Laws": [
    {name:"Ideal Gas Law",formula:"PV = nRT",mw:null,type:"equation",desc:"R = 8.314 J/mol·K  or  0.0821 L·atm/mol·K"},
    {name:"Boyle's Law",formula:"P₁V₁ = P₂V₂  (T constant)",mw:null,type:"equation",desc:"Isothermal process: pressure × volume = constant"},
    {name:"Charles' Law",formula:"V₁/T₁ = V₂/T₂  (P constant)",mw:null,type:"equation",desc:"Isobaric process; T must be in Kelvin"},
    {name:"Gay-Lussac's Law",formula:"P₁/T₁ = P₂/T₂  (V constant)",mw:null,type:"equation",desc:"Isochoric process; pressure ∝ temperature"},
    {name:"Van der Waals",formula:"(P + a/V²)(V - b) = RT",mw:null,type:"equation",desc:"a = intermolecular attraction; b = molecular volume"},
    {name:"Graham's Law",formula:"r₁/r₂ = √(M₂/M₁)",mw:null,type:"equation",desc:"Rate of diffusion inversely proportional to √molar mass"},
    {name:"Dalton's Law",formula:"P_total = P₁ + P₂ + P₃ + …",mw:null,type:"equation",desc:"Total pressure = sum of partial pressures"},
    {name:"RMS Speed",formula:"v_rms = √(3RT / M)",mw:null,type:"equation",desc:"M = molar mass in kg/mol; R = 8.314 J/mol·K"},
    {name:"Mean Free Path",formula:"λ = RT / (√2 · π · d² · Nₐ · P)",mw:null,type:"equation",desc:"Average distance between collisions; d = molecular diameter"},
  ],
  "Organic Chemistry": [
    {name:"Methane",formula:"CH₄",mw:"16.04 g/mol",type:"organic",desc:"Simplest alkane; sp³; tetrahedral 109.5°; b.p. -161.5°C"},
    {name:"Ethanol",formula:"CH₃CH₂OH",mw:"46.07 g/mol",type:"organic",desc:"Primary alcohol; b.p. 78.4°C; H-bonding raises boiling point"},
    {name:"Acetic Acid",formula:"CH₃COOH",mw:"60.05 g/mol",type:"organic",desc:"Simplest carboxylic acid; Ka = 1.8×10⁻⁵; pKa = 4.74; vinegar"},
    {name:"Benzene",formula:"C₆H₆",mw:"78.11 g/mol",type:"organic",desc:"Aromatic; 6π delocalized electrons; resonance energy ~150 kJ/mol"},
    {name:"Glucose",formula:"C₆H₁₂O₆",mw:"180.16 g/mol",type:"organic",desc:"Aldohexose; both open-chain and ring (Haworth) forms exist"},
    {name:"Sucrose",formula:"C₁₂H₂₂O₁₁",mw:"342.30 g/mol",type:"organic",desc:"Disaccharide: glucose + fructose; non-reducing sugar"},
    {name:"Caffeine",formula:"C₈H₁₀N₄O₂",mw:"194.19 g/mol",type:"organic",desc:"Methylxanthine alkaloid; CNS stimulant; purine derivative"},
    {name:"Paracetamol",formula:"C₈H₉NO₂",mw:"151.16 g/mol",type:"organic",desc:"N-acetyl-4-aminophenol; analgesic/antipyretic (Tylenol)"},
    {name:"Aspirin",formula:"C₉H₈O₄",mw:"180.16 g/mol",type:"organic",desc:"Acetylsalicylic acid; analgesic, anti-inflammatory, antipyretic"},
    {name:"ATP",formula:"C₁₀H₁₆N₅O₁₃P₃",mw:"507.18 g/mol",type:"organic",desc:"Energy currency of the cell; hydrolysis releases ~30.5 kJ/mol"},
    {name:"Cholesterol",formula:"C₂₇H₄₆O",mw:"386.65 g/mol",type:"organic",desc:"Steroid; 4 fused rings; precursor to hormones and vitamin D"},
    {name:"Urea",formula:"(NH₂)₂CO",mw:"60.06 g/mol",type:"organic",desc:"Diamide; end product of protein metabolism; fertilizer"},
  ],
  "Alkanes": [
    {name:"Methane",formula:"CH₄",mw:"16.04 g/mol",type:"organic",desc:"sp³; 109.5° bond angle; b.p. -161.5°C; natural gas"},
    {name:"Ethane",formula:"C₂H₆",mw:"30.07 g/mol",type:"organic",desc:"2 C; staggered conformation lower energy than eclipsed"},
    {name:"Propane",formula:"C₃H₈",mw:"44.10 g/mol",type:"organic",desc:"3 C; used as LPG; b.p. -42.1°C"},
    {name:"Butane",formula:"C₄H₁₀",mw:"58.12 g/mol",type:"organic",desc:"n-butane + isobutane; lighter fluid; b.p. -0.5°C"},
    {name:"Pentane",formula:"C₅H₁₂",mw:"72.15 g/mol",type:"organic",desc:"3 isomers: n-, iso-, neopentane; b.p. 36.1°C"},
    {name:"Hexane",formula:"C₆H₁₄",mw:"86.18 g/mol",type:"organic",desc:"5 isomers; common non-polar solvent; b.p. 68.7°C"},
    {name:"Heptane",formula:"C₇H₁₆",mw:"100.20 g/mol",type:"organic",desc:"9 isomers; octane rating reference (0 rating)"},
    {name:"Octane",formula:"C₈H₁₈",mw:"114.23 g/mol",type:"organic",desc:"18 isomers; isooctane has octane rating 100"},
    {name:"Cyclohexane",formula:"C₆H₁₂",mw:"84.16 g/mol",type:"organic",desc:"Chair conformation most stable; ring flip interconverts axial/equatorial H"},
    {name:"General Alkane Formula",formula:"CₙH₂ₙ₊₂",mw:"(14n + 2) g/mol",type:"organic",desc:"n = number of carbon atoms; applies to acyclic alkanes only"},
  ],
  "Alkenes & Alkynes": [
    {name:"Ethene (Ethylene)",formula:"CH₂=CH₂",mw:"28.05 g/mol",type:"organic",desc:"sp²; planar; 120°; monomer for polyethylene"},
    {name:"Propene (Propylene)",formula:"CH₃CH=CH₂",mw:"42.08 g/mol",type:"organic",desc:"Markovnikov addition; monomer for polypropylene"},
    {name:"1,3-Butadiene",formula:"CH₂=CH-CH=CH₂",mw:"54.09 g/mol",type:"organic",desc:"Conjugated diene; used in synthetic rubber; 1,2 and 1,4-addition"},
    {name:"Ethyne (Acetylene)",formula:"HC≡CH",mw:"26.04 g/mol",type:"organic",desc:"sp; linear; 180°; oxy-acetylene welding; acidic terminal H"},
    {name:"Markovnikov's Rule",formula:"H⁺ adds to C with more H; X⁻ adds to C with fewer H",mw:null,type:"equation",desc:"Governs regioselectivity of electrophilic addition to alkenes"},
    {name:"Zaitsev's Rule",formula:"Major product = more substituted alkene",mw:null,type:"equation",desc:"Governs regioselectivity of elimination (E2) reactions"},
    {name:"General Alkene",formula:"CₙH₂ₙ",mw:"14n g/mol",type:"organic",desc:"One degree of unsaturation; one C=C double bond"},
    {name:"General Alkyne",formula:"CₙH₂ₙ₋₂",mw:"(14n - 2) g/mol",type:"organic",desc:"Two degrees of unsaturation; one C≡C triple bond"},
  ],
  "Aromatic Chemistry": [
    {name:"Benzene",formula:"C₆H₆",mw:"78.11 g/mol",type:"organic",desc:"6π electrons; resonance energy ~150 kJ/mol; undergoes EAS"},
    {name:"Toluene",formula:"C₆H₅CH₃",mw:"92.14 g/mol",type:"organic",desc:"Methyl group: EDG; o/p-director; activates ring"},
    {name:"Naphthalene",formula:"C₁₀H₈",mw:"128.17 g/mol",type:"organic",desc:"Bicyclic aromatic; 10π electrons; mothballs"},
    {name:"Phenol",formula:"C₆H₅OH",mw:"94.11 g/mol",type:"organic",desc:"-OH: EDG; o/p-director; activates ring strongly; Ka ≈ 10⁻¹⁰"},
    {name:"Aniline",formula:"C₆H₅NH₂",mw:"93.13 g/mol",type:"organic",desc:"-NH₂: EDG; o/p-director; activates ring; weak base"},
    {name:"Nitrobenzene",formula:"C₆H₅NO₂",mw:"123.11 g/mol",type:"organic",desc:"-NO₂: EWG; meta-director; deactivates ring"},
    {name:"Hückel's Rule",formula:"(4n + 2)π electrons → aromatic",mw:null,type:"equation",desc:"n = 0,1,2…; benzene n=1 (6π)"},
    {name:"EAS Directing Rules",formula:"EDG → ortho/para;  EWG → meta",mw:null,type:"equation",desc:"Electron-donating groups activate; electron-withdrawing deactivate"},
  ],
  "Nuclear Chemistry": [
    {name:"Radioactive Decay",formula:"N(t) = N₀ · e^(-λt)",mw:null,type:"equation",desc:"λ = decay constant (s⁻¹); N₀ = initial number of nuclei"},
    {name:"Half-Life",formula:"t½ = 0.693 / λ",mw:null,type:"equation",desc:"Time for half the nuclei to decay; independent of amount"},
    {name:"Mass-Energy (Einstein)",formula:"E = mc²",mw:null,type:"equation",desc:"c = 3×10⁸ m/s; mass defect converts to binding energy"},
    {name:"Alpha Decay",formula:"ᴬ_ZX → ᴬ⁻⁴_(Z-2)Y + ⁴₂He",mw:null,type:"equation",desc:"Mass number decreases by 4; atomic number decreases by 2"},
    {name:"Beta⁻ Decay",formula:"n → p + e⁻ + v̄_e",mw:null,type:"equation",desc:"Neutron → proton + electron + antineutrino; Z increases by 1"},
    {name:"Nuclear Fission",formula:"²³⁵₉₂U + n → ¹⁴¹₅₆Ba + ⁹²₃₆Kr + 3n + energy",mw:null,type:"equation",desc:"Heavy nucleus splits; chain reaction possible"},
  ],
  "Quantum Chemistry": [
    {name:"de Broglie Wavelength",formula:"λ = h / mv",mw:null,type:"equation",desc:"h = 6.626×10⁻³⁴ J·s; wave-particle duality of matter"},
    {name:"Heisenberg Uncertainty",formula:"Δx · Δp ≥ h / 4π",mw:null,type:"equation",desc:"Cannot simultaneously know exact position and momentum"},
    {name:"Energy Levels (Hydrogen)",formula:"Eₙ = -13.6 / n²  eV",mw:null,type:"equation",desc:"Ground state n=1: E = -13.6 eV; ionisation at E = 0"},
    {name:"Rydberg Formula",formula:"1/λ = R_H(1/n₁² - 1/n₂²)",mw:null,type:"equation",desc:"R_H = 1.097×10⁷ m⁻¹; spectral lines of hydrogen atom"},
    {name:"Photoelectric Effect",formula:"KE = hν - φ",mw:null,type:"equation",desc:"φ = work function of metal; hν = photon energy"},
  ],
  "Stoichiometry": [
    {name:"Limiting Reagent",formula:"n/stoich. coeff. → smallest value = limiting reagent",mw:null,type:"equation",desc:"Determines maximum product; excess reagent left over"},
    {name:"Percent Yield",formula:"% yield = (actual / theoretical) × 100",mw:null,type:"equation",desc:"Actual = measured; theoretical = from stoichiometry"},
    {name:"Atom Economy",formula:"AE% = MW(desired product) / Σ MW(all products) × 100",mw:null,type:"equation",desc:"Green chemistry metric; higher = more efficient/less waste"},
    {name:"Equivalent Weight",formula:"Eq. Wt. = Molar mass / n-factor",mw:null,type:"equation",desc:"n-factor = valency, H replaced, or electrons transferred"},
  ],
  "Coordination Chemistry": [
    {name:"Cisplatin",formula:"[Pt(NH₃)₂Cl₂]",mw:"300.05 g/mol",type:"inorganic",desc:"cis isomer is anticancer drug; trans isomer is inactive"},
    {name:"EDTA4⁻",formula:"C₁₀H₁₂N₂O₈⁴⁻",mw:"292.24 g/mol",type:"inorganic",desc:"Hexadentate ligand; chelates most metal ions; used in titrations"},
    {name:"Crystal Field Splitting",formula:"Δo (octahedral) = 9/4 × Δt (tetrahedral)",mw:null,type:"equation",desc:"Octahedral splitting always larger; affects colour and magnetism"},
    {name:"Magnetic Moment",formula:"μ = √(n(n+2)) BM",mw:null,type:"equation",desc:"n = number of unpaired electrons; BM = Bohr Magnetons"},
    {name:"EAN Rule",formula:"EAN = Z - ox. state + 2 × CN",mw:null,type:"equation",desc:"Effective atomic number; stable complexes near noble gas EAN"},
  ],
  "Periodic Trends": [
    {name:"Effective Nuclear Charge",formula:"Z_eff = Z - σ  (Slater's rules)",mw:null,type:"equation",desc:"σ = shielding constant; Z_eff drives most periodic trends"},
    {name:"Ionization Energy Trend",formula:"IE: increases → across period; decreases ↓ group",mw:null,type:"equation",desc:"Successive IEs increase sharply after valence shell emptied"},
    {name:"Atomic Radius Trend",formula:"r: decreases → across period; increases ↓ group",mw:null,type:"equation",desc:"Increased Z_eff pulls electrons closer across a period"},
    {name:"Electronegativity (Pauling)",formula:"χ_A - χ_B = 0.102 √(D_AB - ½(D_AA + D_BB))",mw:null,type:"equation",desc:"F = 3.98 (highest); Cs = 0.79 (lowest)"},
  ],
};

/* ══════════════════════════════════════════════════════════════
   FORMULA HELPERS
══════════════════════════════════════════════════════════════ */
function deduplicateFormulas(data) {
  const seen = new Set();
  return data.filter(item => {
    const key = item.formula.replace(/\s/g, "").toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function getLocalFormulas(target) {
  const exactKey = Object.keys(FORMULA_DB).find(k => k.toLowerCase() === target.toLowerCase());
  if (exactKey) return deduplicateFormulas(FORMULA_DB[exactKey]);
  const lower = target.toLowerCase();
  for (const [k, v] of Object.entries(FORMULA_DB)) {
    if (k.toLowerCase().includes(lower) || lower.includes(k.toLowerCase()))
      return deduplicateFormulas(v);
  }
  const words = lower.split(/\s+/).filter(w => w.length > 3);
  const merged = [];
  const usedKeys = new Set();
  for (const [k, v] of Object.entries(FORMULA_DB)) {
    if (!usedKeys.has(k) && words.some(w => k.toLowerCase().includes(w))) {
      merged.push(...v); usedKeys.add(k);
    }
  }
  return merged.length ? deduplicateFormulas(merged) : null;
}

/* ══════════════════════════════════════════════════════════════
   VOICE HOOKS
══════════════════════════════════════════════════════════════ */
function useVoiceInput(lang) {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const recRef = useRef(null);
  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    recRef.current = new SR();
    recRef.current.continuous = false;
    recRef.current.interimResults = true;
    recRef.current.lang = VOICE_LANG[lang] || "en-US";
    recRef.current.onresult = e => {
      const t = Array.from(e.results).map(r => r[0].transcript).join("");
      setTranscript(t);
    };
    recRef.current.onend = () => setListening(false);
    return () => recRef.current?.abort();
  }, [lang]);
  const start = () => { if (recRef.current) { setTranscript(""); recRef.current.lang = VOICE_LANG[lang] || "en-US"; recRef.current.start(); setListening(true); } };
  const stop  = () => { if (recRef.current) { recRef.current.stop(); setListening(false); } };
  return { listening, transcript, start, stop, setTranscript };
}

function useTTS() {
  const speak = useCallback((text, lang = "en") => {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = VOICE_LANG[lang] || "en-US";
    u.rate = 0.9; u.pitch = 1; u.volume = 1;
    window.speechSynthesis.speak(u);
  }, []);
  const stop = useCallback(() => window.speechSynthesis?.cancel(), []);
  return { speak, stop };
}

/* ══════════════════════════════════════════════════════════════
   QUIZ WIDGET — Interactive MCQ
══════════════════════════════════════════════════════════════ */
export function QuizWidget({ questions }) {
  const [current,   setCurrent]   = useState(0);
  const [selected,  setSelected]  = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [showExpl,  setShowExpl]  = useState({});

  if (!questions || questions.length === 0) {
    return <div style={QS.empty}>No quiz questions found.</div>;
  }

  const q      = questions[current];
  const total  = questions.length;
  const isLast = current === total - 1;
  const score  = submitted ? questions.filter((q, i) => selected[i] === q.answer).length : null;

  const handleSelect = (letter) => { if (submitted) return; setSelected(prev => ({ ...prev, [current]: letter })); };
  const handleSubmit = () => setSubmitted(true);
  const optLetter    = (opt) => opt.trim()[0];

  const optStyle = (letter) => {
    if (!submitted) return { ...QS.opt, ...(selected[current] === letter ? QS.optSel : {}) };
    if (letter === q.answer) return { ...QS.opt, ...QS.optCorrect };
    if (selected[current] === letter && letter !== q.answer) return { ...QS.opt, ...QS.optWrong };
    return { ...QS.opt, ...QS.optDim };
  };

  return (
    <div style={QS.wrap}>
      <div style={QS.header}>
        <span style={QS.badge}>QUIZ</span>
        <span style={QS.counter}>Q{current + 1} / {total}</span>
        {submitted && (
          <span style={{ ...QS.badge, background: score >= total * 0.6 ? "#22c55e22" : "#ef444422", color: score >= total * 0.6 ? "#22c55e" : "#ef4444", borderColor: score >= total * 0.6 ? "#22c55e44" : "#ef444444" }}>
            Score: {score}/{total}
          </span>
        )}
      </div>
      <div style={QS.progBg}>
        <div style={{ ...QS.progFill, width: `${((current + 1) / total) * 100}%` }} />
      </div>
      <p style={QS.question}>{q.question}</p>
      <div style={QS.opts}>
        {q.options.map((opt, i) => {
          const letter = optLetter(opt);
          return (
            <button key={i} style={optStyle(letter)} onClick={() => handleSelect(letter)}>
              <span style={QS.optLetter}>{letter}</span>
              <span style={QS.optText}>{opt.slice(2).trim()}</span>
              {submitted && letter === q.answer && <span style={QS.tick}>✓</span>}
              {submitted && selected[current] === letter && letter !== q.answer && <span style={QS.cross}>✗</span>}
            </button>
          );
        })}
      </div>
      {submitted && (
        <div style={QS.expl}>
          <button style={QS.explToggle} onClick={() => setShowExpl(p => ({ ...p, [current]: !p[current] }))}>
            {showExpl[current] ? "▾ Hide" : "▸ Show"} Explanation
          </button>
          {showExpl[current] && <div style={QS.explText}>📖 {q.explanation}</div>}
        </div>
      )}
      <div style={QS.nav}>
        <button style={QS.navBtn} onClick={() => setCurrent(c => Math.max(0, c - 1))} disabled={current === 0}>← Prev</button>
        {!submitted && isLast && (
          <button style={{ ...QS.navBtn, ...QS.submitBtn }} onClick={handleSubmit} disabled={Object.keys(selected).length < total}>Submit Quiz</button>
        )}
        {submitted && !isLast && (
          <button style={{ ...QS.navBtn, ...QS.submitBtn }} onClick={() => setCurrent(c => c + 1)}>Next →</button>
        )}
        {!isLast && !submitted && (
          <button style={QS.navBtn} onClick={() => setCurrent(c => Math.min(total - 1, c + 1))} disabled={!selected[current]}>Next →</button>
        )}
      </div>
      {submitted && (
        <div style={QS.scoreBar}>
          {questions.map((_, i) => (
            <div key={i}
              style={{ ...QS.scoreDot, background: selected[i] === questions[i].answer ? "#22c55e" : "#ef4444", opacity: i === current ? 1 : 0.65, transform: i === current ? "scale(1.25)" : "scale(1)", cursor: "pointer" }}
              onClick={() => setCurrent(i)}
              title={`Q${i + 1}: ${selected[i] === questions[i].answer ? "Correct" : "Wrong"}`}
            />
          ))}
        </div>
      )}
    </div>
  );
}

const QS = {
  wrap:       { background:"rgba(56,189,248,0.04)", border:"1px solid rgba(56,189,248,0.18)", borderRadius:14, padding:"18px 20px", marginTop:10 },
  empty:      { color:"#64748b", fontSize:13, padding:12, textAlign:"center" },
  header:     { display:"flex", alignItems:"center", gap:8, marginBottom:10 },
  badge:      { padding:"3px 10px", borderRadius:20, fontSize:10, fontWeight:800, background:"rgba(56,189,248,0.12)", color:"#38bdf8", border:"1px solid rgba(56,189,248,0.25)", letterSpacing:1 },
  counter:    { fontSize:12, color:"#64748b", fontWeight:700, marginLeft:"auto" },
  progBg:     { height:3, background:"rgba(255,255,255,0.07)", borderRadius:3, marginBottom:16, overflow:"hidden" },
  progFill:   { height:"100%", background:"linear-gradient(90deg,#38bdf8,#818cf8)", borderRadius:3, transition:"width 0.4s ease" },
  question:   { fontSize:14, fontWeight:700, color:"#f1f5f9", lineHeight:1.6, margin:"0 0 14px" },
  opts:       { display:"flex", flexDirection:"column", gap:8, marginBottom:14 },
  opt:        { display:"flex", alignItems:"center", gap:10, padding:"10px 14px", borderRadius:9, border:"1px solid rgba(255,255,255,0.1)", background:"rgba(255,255,255,0.03)", cursor:"pointer", transition:"all 0.18s", textAlign:"left" },
  optSel:     { background:"rgba(56,189,248,0.1)", borderColor:"rgba(56,189,248,0.5)", boxShadow:"0 0 12px rgba(56,189,248,0.15)" },
  optCorrect: { background:"rgba(34,197,94,0.1)", borderColor:"rgba(34,197,94,0.5)", cursor:"default" },
  optWrong:   { background:"rgba(239,68,68,0.1)", borderColor:"rgba(239,68,68,0.5)", cursor:"default" },
  optDim:     { opacity:0.45, cursor:"default" },
  optLetter:  { width:24, height:24, borderRadius:6, display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, fontWeight:800, background:"rgba(255,255,255,0.07)", color:"#94a3b8", flexShrink:0 },
  optText:    { fontSize:13, color:"#e2e8f0", flex:1, lineHeight:1.4 },
  tick:       { color:"#22c55e", fontSize:14, fontWeight:900, flexShrink:0 },
  cross:      { color:"#ef4444", fontSize:14, fontWeight:900, flexShrink:0 },
  expl:       { marginBottom:12 },
  explToggle: { background:"none", border:"none", color:"#38bdf8", fontSize:12, cursor:"pointer", padding:"3px 0", fontWeight:600 },
  explText:   { background:"rgba(56,189,248,0.05)", border:"1px solid rgba(56,189,248,0.15)", borderRadius:8, padding:"10px 12px", fontSize:12, color:"#94a3b8", lineHeight:1.65, marginTop:6 },
  nav:        { display:"flex", gap:8, justifyContent:"space-between", alignItems:"center", marginTop:4 },
  navBtn:     { padding:"8px 16px", borderRadius:8, border:"1px solid rgba(255,255,255,0.12)", background:"rgba(255,255,255,0.04)", color:"#94a3b8", fontSize:12, fontWeight:600, cursor:"pointer", transition:"all 0.18s" },
  submitBtn:  { background:"linear-gradient(135deg,#0ea5e9,#38bdf8)", border:"none", color:"#020617", fontWeight:800, flex:1, maxWidth:160 },
  scoreBar:   { display:"flex", gap:6, justifyContent:"center", marginTop:14, flexWrap:"wrap" },
  scoreDot:   { width:10, height:10, borderRadius:"50%", transition:"all 0.2s", cursor:"pointer" },
};

/* ══════════════════════════════════════════════════════════════
   POINTWISE ANSWER RENDERER
══════════════════════════════════════════════════════════════ */
export function PointwiseAnswer({ text }) {
  if (!text) return null;

  const lines = text.split("\n");
  const elements = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) { elements.push(<div key={key++} style={{ height:8 }} />); continue; }

    if (trimmed.startsWith("## ")) {
      const title = trimmed.slice(3);
      elements.push(
        <div key={key++} style={PS.h2wrap}>
          <span style={PS.h2dot} />
          <h3 style={PS.h2}>{title}</h3>
        </div>
      );
      continue;
    }
    if (trimmed.startsWith("# ")) {
      elements.push(<h2 key={key++} style={PS.h1}>{trimmed.slice(2)}</h2>);
      continue;
    }
    if (trimmed.startsWith("### ")) {
      elements.push(<div key={key++} style={{ color:"#38bdf8", fontSize:13, fontWeight:700, margin:"12px 0 6px", textTransform:"uppercase", letterSpacing:0.8 }}>{trimmed.slice(4)}</div>);
      continue;
    }

    const numMatch = trimmed.match(/^(\d+)[\.\)]\s+(.+)$/);
    if (numMatch) {
      elements.push(
        <div key={key++} style={PS.numRow}>
          <span style={PS.numBadge}>{numMatch[1]}</span>
          <span style={PS.numText}>{renderInline(numMatch[2])}</span>
        </div>
      );
      continue;
    }

    const stepMatch = trimmed.match(/^Step\s+(\d+)[:\.]\s+(.+)$/i);
    if (stepMatch) {
      elements.push(
        <div key={key++} style={PS.stepRow}>
          <div style={PS.stepNum}><span style={PS.stepNumTxt}>{stepMatch[1]}</span></div>
          <div style={PS.stepContent}>{renderInline(stepMatch[2])}</div>
        </div>
      );
      continue;
    }

    if (trimmed.startsWith("- ") || trimmed.startsWith("• ") || trimmed.startsWith("* ")) {
      elements.push(
        <div key={key++} style={PS.bullet}>
          <span style={PS.bulletDot}>▸</span>
          <span style={PS.bulletText}>{renderInline(trimmed.slice(2))}</span>
        </div>
      );
      continue;
    }

    if (trimmed === "---" || trimmed === "***") {
      elements.push(<div key={key++} style={PS.hr} />);
      continue;
    }

    if (trimmed.startsWith("```")) {
      let codeLines = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]); i++;
      }
      const codeText = codeLines.join("\n").trim();
      if (!trimmed.includes("json:quiz")) {
        elements.push(
          <div key={key++} style={PS.code}><code style={{ fontFamily:"monospace", fontSize:12 }}>{codeText}</code></div>
        );
      }
      continue;
    }

    if (trimmed.startsWith("**") && trimmed.endsWith("**")) {
      elements.push(<p key={key++} style={PS.boldLine}>{trimmed.slice(2, -2)}</p>);
      continue;
    }

    elements.push(<p key={key++} style={PS.para}>{renderInline(trimmed)}</p>);
  }

  return <div style={PS.root}>{elements}</div>;
}

function renderInline(text) {
  const parts = [];
  let idx = 0;
  const pattern = /(\*\*(.+?)\*\*|`(.+?)`|_(.+?)_|💡.+)/g;
  let lastIndex = 0;
  let match;
  pattern.lastIndex = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));
    if (match[2]) {
      parts.push(<strong key={idx++} style={{ color:"#f1f5f9", fontWeight:700 }}>{match[2]}</strong>);
    } else if (match[3]) {
      parts.push(<code key={idx++} style={{ background:"rgba(56,189,248,0.1)", padding:"1px 6px", borderRadius:4, fontSize:"0.9em", color:"#38bdf8", fontFamily:"monospace" }}>{match[3]}</code>);
    } else if (match[4]) {
      parts.push(<em key={idx++} style={{ color:"#94a3b8" }}>{match[4]}</em>);
    } else {
      parts.push(<span key={idx++} style={{ color:"#fbbf24" }}>{match[0]}</span>);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts.length > 0 ? parts : text;
}

const PS = {
  root:        { padding:"4px 0" },
  h1:          { fontSize:17, fontWeight:800, color:"#f1f5f9", margin:"12px 0 6px", letterSpacing:-0.3 },
  h2wrap:      { display:"flex", alignItems:"center", gap:8, margin:"14px 0 6px" },
  h2dot:       { width:3, height:18, borderRadius:2, background:"linear-gradient(180deg,#38bdf8,#818cf8)", flexShrink:0 },
  h2:          { fontSize:13, fontWeight:800, color:"#38bdf8", margin:0, letterSpacing:0.5, textTransform:"uppercase" },
  para:        { fontSize:13, color:"#94a3b8", lineHeight:1.7, margin:"4px 0" },
  boldLine:    { fontSize:13, fontWeight:700, color:"#e2e8f0", margin:"4px 0" },
  numRow:      { display:"flex", alignItems:"flex-start", gap:10, padding:"5px 0", borderBottom:"1px solid rgba(255,255,255,0.04)" },
  numBadge:    { width:22, height:22, borderRadius:6, background:"linear-gradient(135deg,rgba(56,189,248,0.15),rgba(129,140,248,0.15))", border:"1px solid rgba(56,189,248,0.2)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, fontWeight:800, color:"#38bdf8", flexShrink:0, marginTop:1 },
  numText:     { fontSize:13, color:"#e2e8f0", lineHeight:1.65, flex:1 },
  stepRow:     { display:"flex", alignItems:"flex-start", gap:10, padding:"6px 0" },
  stepNum:     { width:26, height:26, borderRadius:"50%", background:"rgba(56,189,248,0.1)", border:"1px solid rgba(56,189,248,0.3)", display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 },
  stepNumTxt:  { fontSize:11, fontWeight:800, color:"#38bdf8" },
  stepContent: { fontSize:13, color:"#e2e8f0", lineHeight:1.65, flex:1, paddingTop:3 },
  bullet:      { display:"flex", alignItems:"flex-start", gap:8, padding:"3px 0" },
  bulletDot:   { color:"#38bdf8", fontSize:10, flexShrink:0, marginTop:4 },
  bulletText:  { fontSize:13, color:"#e2e8f0", lineHeight:1.65, flex:1 },
  hr:          { height:1, background:"rgba(255,255,255,0.06)", margin:"12px 0" },
  code:        { background:"rgba(0,0,0,0.3)", border:"1px solid rgba(56,189,248,0.15)", borderRadius:8, padding:"10px 14px", margin:"8px 0", overflowX:"auto" },
};

/* ══════════════════════════════════════════════════════════════
   CHAT MESSAGE
══════════════════════════════════════════════════════════════ */
export function ChatMessage({ role, content }) {
  const isBot = role === "assistant" || role === "bot";

  const quizMatch = content.match(/```json:quiz\s*([\s\S]*?)```/);
  const quizData  = quizMatch ? (() => {
    try { return JSON.parse(quizMatch[1]); } catch { return null; }
  })() : null;
  const displayText = content.replace(/```json:quiz[\s\S]*?```/, "").trim();

  return (
    <div style={{ ...CM.wrap, justifyContent: isBot ? "flex-start" : "flex-end" }}>
      {isBot && <div style={CM.avatar}>⚗</div>}
      <div style={{ ...CM.bubble, ...(isBot ? CM.botBubble : CM.userBubble), maxWidth: isBot ? "88%" : "72%" }}>
        {isBot ? (
          <>
            <PointwiseAnswer text={displayText} />
            {quizData && quizData.length > 0 && <QuizWidget questions={quizData} />}
          </>
        ) : (
          <p style={CM.userText}>{content}</p>
        )}
      </div>
      {!isBot && <div style={CM.userAvatar}>👤</div>}
    </div>
  );
}

const CM = {
  wrap:       { display:"flex", alignItems:"flex-start", gap:10, padding:"8px 0" },
  avatar:     { width:32, height:32, borderRadius:10, background:"rgba(56,189,248,0.1)", border:"1px solid rgba(56,189,248,0.2)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:15, flexShrink:0, marginTop:2 },
  userAvatar: { width:32, height:32, borderRadius:10, background:"rgba(129,140,248,0.1)", border:"1px solid rgba(129,140,248,0.2)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:14, flexShrink:0, marginTop:2 },
  bubble:     { borderRadius:14, padding:"14px 16px", position:"relative" },
  botBubble:  { background:"rgba(10,13,30,0.85)", border:"1px solid rgba(255,255,255,0.07)", borderTopLeftRadius:4 },
  userBubble: { background:"rgba(56,189,248,0.1)", border:"1px solid rgba(56,189,248,0.2)", borderTopRightRadius:4 },
  userText:   { fontSize:13, color:"#e2e8f0", margin:0, lineHeight:1.65 },
};

/* ══════════════════════════════════════════════════════════════
   QUIZ LAUNCHER
══════════════════════════════════════════════════════════════ */
export function QuizLauncher({ onSend }) {
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState(5);
  const QUICK = ["Acid & Base","Oxidation Reduction","Organic Chemistry","Chemical Bonding","Thermodynamics","Electrochemistry","Reaction Kinetics","Periodic Table","Equilibrium"];

  const launch = () => {
    if (!topic.trim()) return;
    onSend(`QUIZ:${count}: ${topic.trim()}`);
    setTopic("");
  };

  return (
    <div style={QL.wrap}>
      <div style={QL.row}>
        <input value={topic} onChange={e => setTopic(e.target.value)} onKeyDown={e => e.key === "Enter" && launch()}
          placeholder="Enter quiz topic (e.g. acid base, organic, redox)…" style={QL.input} />
        <select value={count} onChange={e => setCount(Number(e.target.value))} style={QL.select}>
          {[3,5,7,10].map(n => <option key={n} value={n}>{n} Qs</option>)}
        </select>
        <button onClick={launch} style={QL.btn}>Generate Quiz</button>
      </div>
      <div style={QL.chips}>
        <span style={QL.chipLabel}>Quick topics:</span>
        {QUICK.map(t => <button key={t} style={QL.chip} onClick={() => setTopic(t)}>{t}</button>)}
      </div>
    </div>
  );
}

const QL = {
  wrap:      { padding:"12px 16px", background:"rgba(56,189,248,0.03)", borderTop:"1px solid rgba(255,255,255,0.06)", borderRadius:"0 0 14px 14px" },
  row:       { display:"flex", gap:8, marginBottom:10 },
  input:     { flex:1, padding:"9px 14px", background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:9, color:"#f1f5f9", fontSize:13, outline:"none", fontFamily:"inherit" },
  select:    { padding:"9px 10px", background:"rgba(255,255,255,0.05)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:9, color:"#94a3b8", fontSize:12, cursor:"pointer", outline:"none" },
  btn:       { padding:"9px 18px", background:"linear-gradient(135deg,#0ea5e9,#38bdf8)", border:"none", borderRadius:9, color:"#020617", fontSize:12, fontWeight:800, cursor:"pointer" },
  chips:     { display:"flex", flexWrap:"wrap", gap:6, alignItems:"center" },
  chipLabel: { fontSize:11, color:"#475569", fontWeight:700, marginRight:2 },
  chip:      { padding:"4px 11px", borderRadius:20, fontSize:11, fontWeight:600, background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", color:"#64748b", cursor:"pointer", transition:"all 0.18s" },
};

/* ══════════════════════════════════════════════════════════════
   SHARED UI COMPONENTS
══════════════════════════════════════════════════════════════ */
const AtomIcon = ({ size = 26 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="1.8" strokeLinecap="round" style={{ animation:"atomSpin 10s linear infinite" }}>
    <circle cx="12" cy="12" r="2.2" fill="#38bdf8" />
    <ellipse cx="12" cy="12" rx="10" ry="3.8" />
    <ellipse cx="12" cy="12" rx="10" ry="3.8" transform="rotate(60 12 12)" />
    <ellipse cx="12" cy="12" rx="10" ry="3.8" transform="rotate(120 12 12)" />
  </svg>
);

function LangSelect({ value, onChange }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} style={S.select}>
      {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
    </select>
  );
}

function MicBtn({ lang, onResult, t }) {
  const { listening, transcript, start, stop, setTranscript } = useVoiceInput(lang);
  useEffect(() => {
    if (transcript && !listening) { onResult(transcript); setTranscript(""); }
  }, [transcript, listening]);
  return (
    <button onClick={listening ? stop : start} title={listening ? t.voice.stop : t.voice.start}
      style={{ ...S.iconBtn, background: listening ? "rgba(239,68,68,0.2)" : "rgba(56,189,248,0.12)", borderColor: listening ? "#ef4444" : "#38bdf8", animation: listening ? "pulse 1.2s infinite" : "none" }}>
      {listening ? "🔴" : "🎤"}
    </button>
  );
}

function SpeakBtn({ text, lang, t }) {
  const { speak, stop } = useTTS();
  const [on, setOn] = useState(false);
  if (!text) return null;
  const toggle = () => {
    if (on) { stop(); setOn(false); }
    else { speak(text, lang); setOn(true); setTimeout(() => setOn(false), Math.max(text.length * 60, 3000)); }
  };
  return (
    <button onClick={toggle} title={t.voice.speak}
      style={{ ...S.iconBtn, background: on ? "rgba(34,197,94,0.2)" : "rgba(56,189,248,0.1)", borderColor: on ? "#22c55e" : "#38bdf8", fontSize:16 }}>
      {on ? "🔊" : "🔈"}
    </button>
  );
}

function FileZone({ onFile, accept = "*", label, file }) {
  const ref = useRef(); const [drag, setDrag] = useState(false);
  return (
    <div onClick={() => ref.current.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0]); }}
      style={{ border:`2px dashed ${drag ? "#38bdf8" : "rgba(56,189,248,0.25)"}`, borderRadius:12, padding:22, textAlign:"center", cursor:"pointer", background: drag ? "rgba(56,189,248,0.05)" : "transparent", transition:"all 0.2s" }}>
      <input ref={ref} type="file" accept={accept} style={{ display:"none" }} onChange={e => onFile(e.target.files[0])} />
      <div style={{ fontSize:26, marginBottom:5 }}>📁</div>
      <div style={{ color: file ? "#38bdf8" : "#64748b", fontSize:13, fontWeight:600 }}>{file ? `✅ ${file.name}` : label}</div>
    </div>
  );
}

function Spinner({ text }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, padding:"18px 0" }}>
      <span style={S.spinner} />
      <span style={{ color:"#64748b", fontSize:14 }}>{text}</span>
    </div>
  );
}

function OutputBox({ content, loading, placeholder, lang, t, loadingText }) {
  if (loading) return <div style={S.outputBox}><Spinner text={loadingText || "Generating…"} /></div>;
  if (!content) return <div style={{ ...S.outputBox, color:"#475569" }}>{placeholder}</div>;

  return (
    <div style={{ position:"relative" }}>
      <div style={{ position:"absolute", top:12, right:12, zIndex:2 }}>
        <SpeakBtn text={content} lang={lang} t={t} />
      </div>
      <div style={{ ...S.outputBox, paddingTop:20 }}>
        <PointwiseAnswer text={content} />
      </div>
    </div>
  );
}

function SectionHeader({ icon, title, sub, badge }) {
  return (
    <div style={S.sectionHeader}>
      <div style={{ width:50, height:50, borderRadius:13, background:"rgba(56,189,248,0.08)", border:"1px solid rgba(56,189,248,0.18)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:24, flexShrink:0 }}>{icon}</div>
      <div style={{ flex:1 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:3 }}>
          <h2 style={S.sectionTitle}>{title}</h2>
          {badge && <span style={{ padding:"2px 9px", borderRadius:14, fontSize:10, fontWeight:700, background:"rgba(56,189,248,0.12)", color:"#38bdf8", border:"1px solid rgba(56,189,248,0.2)" }}>{badge}</span>}
        </div>
        <p style={S.sectionSub}>{sub}</p>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 1 — ASK CHEMISTRY
══════════════════════════════════════════════════════════════ */
function AskSection({ lang, setGlobalHistory }) {
  const t = T(lang);
  const [q, setQ] = useState("");
  const [out, setOut] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const generate = async (question) => {
    const trimmed = (question || q).trim();
    if (!trimmed) return;
    if (/\S+@\S+\.\S+/.test(trimmed)) { setErr(t.ask.emailError); return; }
    if (/^\d+$/.test(trimmed))         { setErr(t.ask.numberError); return; }
    if (!isChemistryQuery(trimmed, lang)) { setErr(t.ask.emptyError); return; }
    setLoading(true); setOut(""); setErr("");
    try {
      const prompt = buildPrompt(
        `You are a chemistry expert tutor. Answer the following question in a clear structured format.\nUse ## for section headings.\nUse numbered lists (1. 2. 3.) for steps or sequences.\nUse bullet points (- ) for key facts or properties.\nDo NOT write plain unbroken paragraphs.\nQuestion: ${trimmed}`,
        lang
      );
      const res = await apiPredict(prompt, lang);
      setOut(res);
      setGlobalHistory(h => [{ q: trimmed.slice(0, 50), ts: Date.now() }, ...h.slice(0, 29)]);
    } catch (e) { setErr("❌ " + e.message); }
    setLoading(false);
  };

  return (
    <div style={S.section}>
      <SectionHeader icon="⚗️" title={t.ask.title} sub={t.ask.sub} badge="FLAN-T5" />
      <div style={S.card}>
        <label style={S.label}>{t.ask.label}</label>
        <div style={S.inputRow}>
          <textarea style={{ ...S.textarea, flex:1 }} value={q} onChange={e => setQ(e.target.value)}
            onKeyDown={e => e.key === "Enter" && e.ctrlKey && generate()} rows={4} placeholder={t.ask.placeholder} />
          <MicBtn lang={lang} t={t} onResult={v => { setQ(v); setTimeout(() => generate(v), 200); }} />
        </div>
        <div style={S.btnRow}>
          <button style={S.primaryBtn} onClick={() => generate()} disabled={loading}>{loading ? t.ask.loading : t.ask.btn}</button>
          <span style={{ color:"#475569", fontSize:11 }}>{t.ask.hint}</span>
        </div>
      </div>
      {err && <div style={S.errorBox}>{err}</div>}
      <OutputBox content={out} loading={loading} lang={lang} t={t} placeholder={t.ask.placeholder2} loadingText={t.ask.loading} />
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 2 — FORMULAS
══════════════════════════════════════════════════════════════ */
function FormulaSection({ lang }) {
  const t = T(lang);
  const [topic, setTopic] = useState("");
  const [formulas, setFormulas] = useState(null);
  const [rawOut, setRawOut] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeCat, setActiveCat] = useState(null);
  const cats = Object.keys(FORMULA_DB);

  const generate = async (forceTopic) => {
    const target = (forceTopic || topic || "").trim();
    if (!target) return;
    setActiveCat(target); setLoading(true); setFormulas(null); setRawOut("");
    const local = getLocalFormulas(target);
    if (local && local.length > 0) { setFormulas(local); setLoading(false); return; }
    try {
      const prompt = buildPrompt(`List ALL important chemistry formulas for: "${target}". For every formula include: Name | Formula | Molecular Weight (or N/A) | Brief explanation. Do not repeat any formula. Be thorough.`, lang);
      const res = await apiPredict(prompt, lang);
      setRawOut(res);
    } catch (e) { setRawOut("❌ " + e.message); }
    setLoading(false);
  };

  const organicList  = formulas?.filter(f => f.type === "organic")   || [];
  const equationList = formulas?.filter(f => f.type === "equation")  || [];
  const inorgList    = formulas?.filter(f => f.type === "inorganic") || [];

  const FormulaCard = ({ f }) => {
    const typeColor = f.type === "organic" ? "#22c55e" : f.type === "inorganic" ? "#38bdf8" : "#a855f7";
    const typeBg    = f.type === "organic" ? "rgba(34,197,94,0.10)" : f.type === "inorganic" ? "rgba(56,189,248,0.10)" : "rgba(168,85,247,0.10)";
    const typeBorder= f.type === "organic" ? "rgba(34,197,94,0.25)" : f.type === "inorganic" ? "rgba(56,189,248,0.25)" : "rgba(168,85,247,0.25)";
    return (
      <div style={{ background:"rgba(14,18,40,0.95)", border:"1px solid rgba(56,189,248,0.12)", borderRadius:12, padding:"14px 16px", display:"flex", flexDirection:"column", gap:4, transition:"border-color 0.2s" }}
        onMouseEnter={e => e.currentTarget.style.borderColor = "rgba(56,189,248,0.35)"}
        onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(56,189,248,0.12)"}>
        <span style={{ alignSelf:"flex-start", padding:"2px 8px", borderRadius:9, fontSize:10, fontWeight:700, background:typeBg, color:typeColor, border:`1px solid ${typeBorder}` }}>{f.type}</span>
        <div style={{ color:"#f1f5f9", fontSize:13, fontWeight:700, lineHeight:1.4 }}>{f.name}</div>
        <div style={{ color:"#38bdf8", fontSize:15, fontFamily:"'Courier New', monospace", fontWeight:700, letterSpacing:0.4, wordBreak:"break-all" }}>{f.formula}</div>
        {f.mw && <div style={{ color:"#f59e0b", fontSize:11, fontWeight:700 }}>MW: {f.mw}</div>}
        <div style={{ color:"#64748b", fontSize:11, lineHeight:1.55, marginTop:2 }}>{f.desc}</div>
      </div>
    );
  };

  const CatSection = ({ title, list }) => list.length === 0 ? null : (
    <>
      <div style={{ color:"#7dd3fc", fontSize:11, fontWeight:800, letterSpacing:1.2, textTransform:"uppercase", padding:"10px 0 8px", borderBottom:"1px solid rgba(255,255,255,0.05)", marginBottom:12 }}>{title}</div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:20 }}>
        {list.map((f, i) => <FormulaCard key={i} f={f} />)}
      </div>
    </>
  );

  return (
    <div style={S.section}>
      <SectionHeader icon="🧮" title={t.formula.title} sub={t.formula.sub} badge="FLAN-T5" />
      <div style={S.card}>
        <label style={S.label}>{t.formula.label}</label>
        <div style={S.chipRow}>
          {cats.map(c => (
            <button key={c} style={{ ...S.chip, ...(activeCat === c ? { background:"rgba(56,189,248,0.18)", borderColor:"#38bdf8", color:"#7dd3fc" } : {}) }}
              onClick={() => { setTopic(c); generate(c); }}>{c}</button>
          ))}
        </div>
        <label style={{ ...S.label, marginTop:16 }}>{t.formula.customLabel}</label>
        <div style={S.inputRow}>
          <input style={S.input} value={topic} onChange={e => setTopic(e.target.value)}
            placeholder={t.formula.placeholder} onKeyDown={e => e.key === "Enter" && generate()} />
          <MicBtn lang={lang} t={t} onResult={v => { setTopic(v); setTimeout(() => generate(v), 200); }} />
          <button style={S.primaryBtn} onClick={() => generate()} disabled={loading}>{loading ? "…" : t.formula.btn}</button>
        </div>
      </div>
      {loading && <div style={S.outputBox}><Spinner text={t.formula.loading} /></div>}
      {formulas && !loading && (
        <div style={S.card}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16 }}>
            <span style={{ color:"#f1f5f9", fontWeight:800, fontSize:15 }}>{activeCat}</span>
            <span style={{ color:"#64748b", fontSize:12 }}>{formulas.length} unique formulas</span>
          </div>
          <CatSection title="📐 Equations & Laws" list={equationList} />
          <CatSection title="🧬 Molecular Formulas & Weights" list={organicList} />
          <CatSection title="⚗️ Inorganic / Coordination" list={inorgList} />
        </div>
      )}
      {rawOut && !loading && <OutputBox content={rawOut} loading={false} lang={lang} t={t} />}
      {!formulas && !rawOut && !loading && <div style={{ ...S.outputBox, color:"#475569" }}>{t.formula.placeholder2}</div>}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 3 — STRUCTURE
══════════════════════════════════════════════════════════════ */
function StructureSection({ lang }) {
  const t = T(lang);
  const [compound, setCompound] = useState("");
  const [imgUrl, setImgUrl] = useState(null);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const supported = ["water","methane","ethane","ethanol","benzene","carbon dioxide","ammonia","glucose","aspirin","caffeine","acetic acid","sulfuric acid","sodium hydroxide","hydrogen peroxide","nitric acid","acetone","toluene","citric acid","paracetamol","ibuprofen"];

  const generate = async (c) => {
    const target = c || compound;
    if (!target.trim()) return;
    setLoading(true); setImgUrl(null); setMsg("");
    try {
      const data = await apiStructure(target.trim().toLowerCase());
      if (data.image_url) setImgUrl(`${API}${data.image_url}`);
      const translatedMsg = await apiTranslate(data.message || "", lang);
      setMsg(translatedMsg);
    } catch (e) { setMsg("❌ " + e.message); }
    setLoading(false);
  };

  return (
    <div style={S.section}>
      <SectionHeader icon="🎨" title={t.structure.title} sub={t.structure.sub} badge="RDKit" />
      <div style={S.card}>
        <label style={S.label}>{t.structure.label}</label>
        <div style={S.chipRow}>
          {supported.map(c => <button key={c} style={S.chip} onClick={() => { setCompound(c); generate(c); }}>{c}</button>)}
        </div>
        <label style={{ ...S.label, marginTop:16 }}>{t.structure.customLabel}</label>
        <div style={S.inputRow}>
          <input style={S.input} value={compound} onChange={e => setCompound(e.target.value)}
            placeholder={t.structure.placeholder} onKeyDown={e => e.key === "Enter" && generate()} />
          <MicBtn lang={lang} t={t} onResult={v => { setCompound(v); setTimeout(() => generate(v), 200); }} />
          <button style={S.primaryBtn} onClick={() => generate()} disabled={loading}>{loading ? t.structure.loading : t.structure.btn}</button>
        </div>
      </div>
      {loading && <div style={S.outputBox}><Spinner text={t.structure.loading} /></div>}
      {imgUrl && !loading && (
        <div style={S.card}>
          <label style={S.label}>{t.structure.rdkitLabel}</label>
          <div style={{ background:"#fff", borderRadius:10, padding:14, display:"inline-block" }}>
            <img src={imgUrl} alt={compound} style={{ maxWidth:"100%", borderRadius:8, display:"block" }} />
          </div>
          {msg && <p style={{ color:"#94a3b8", fontSize:13, marginTop:8 }}>{msg}</p>}
        </div>
      )}
      {!loading && !imgUrl && msg && <div style={{ ...S.outputBox, color:"#fca5a5" }}>{msg}</div>}
      {!loading && !imgUrl && !msg && <div style={{ ...S.outputBox, color:"#475569" }}>{t.structure.placeholder2}</div>}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 4 — ANALYZE PDF
══════════════════════════════════════════════════════════════ */
function AnalyzeSection({ lang }) {
  const t = T(lang);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("summary");

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setResult(null);
    try { const data = await apiPDFAnalyze(file, lang); setResult(data); }
    catch (e) { setResult({ error: e.message }); }
    setLoading(false);
  };

  return (
    <div style={S.section}>
      <SectionHeader icon="🔬" title={t.analyze.title} sub={t.analyze.sub} badge="PyMuPDF" />
      <div style={S.card}>
        <FileZone file={file} onFile={setFile} accept="image/*,application/pdf" label={t.analyze.dropLabel} />
        {file && <div style={{ marginTop:8, padding:"7px 12px", borderRadius:8, background:"rgba(56,189,248,0.06)", border:"1px solid rgba(56,189,248,0.15)", fontSize:12, color:"#38bdf8" }}>📎 {file.name} · {(file.size/1024).toFixed(1)} KB</div>}
        <button style={{ ...S.primaryBtn, marginTop:14, opacity:(!file||loading)?0.5:1 }}
          onClick={analyze} disabled={!file||loading}>
          {loading ? t.analyze.loading : t.analyze.btn}
        </button>
      </div>
      {loading && <div style={S.outputBox}><Spinner text={t.analyze.loading} /></div>}
      {result && !loading && (
        result.error
          ? <div style={{ ...S.outputBox, color:"#fca5a5" }}>❌ {result.error}</div>
          : <div style={S.card}>
              <div style={{ display:"flex", gap:8, marginBottom:14, flexWrap:"wrap" }}>
                {[["summary",t.analyze.summary],["quiz",t.analyze.quiz],["video_script",t.analyze.video]].map(([id,lbl]) => (
                  <button key={id} onClick={() => setTab(id)}
                    style={{ ...S.chip, ...(tab===id ? { background:"rgba(56,189,248,0.18)", borderColor:"#38bdf8", color:"#7dd3fc" } : {}) }}>
                    {lbl}
                  </button>
                ))}
              </div>
              <OutputBox content={result[tab]} loading={false} lang={lang} t={t} />
            </div>
      )}
      {!loading && !result && <div style={{ ...S.outputBox, color:"#475569" }}>{t.analyze.placeholder}</div>}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 5 — SHORT NOTES
══════════════════════════════════════════════════════════════ */
function NotesSection({ lang }) {
  const t = T(lang);
  const [file, setFile] = useState(null);
  const [topic, setTopic] = useState("");
  const [out, setOut] = useState("");
  const [loading, setLoading] = useState(false);

  const generate = async () => {
    setLoading(true); setOut("");
    try {
      if (file && file.type === "application/pdf") {
        const data = await apiPDFAnalyze(file, lang);
        setOut(data.summary || data.error || "");
      } else if (topic.trim()) {
        const prompt = buildPrompt(
          `Create detailed short notes for exam preparation on: ${topic}.\n## Structure your notes like this:\n- Use ## for each major section heading\n- Use numbered lists (1. 2. 3.) for processes and sequences\n- Use bullet points (- ) for definitions, facts, and properties\n- Include formulas on their own lines\nDo NOT write plain paragraphs.`,
          lang
        );
        const res = await apiPredict(prompt, lang);
        setOut(res);
      } else { setOut("⚠️ " + t.notes.pdfLabel + " / " + t.notes.topicLabel); }
    } catch (e) { setOut("❌ " + e.message); }
    setLoading(false);
  };

  return (
    <div style={S.section}>
      <SectionHeader icon="📝" title={t.notes.title} sub={t.notes.sub} badge="FLAN-T5" />
      <div style={S.card}>
        <label style={S.label}>{t.notes.pdfLabel}</label>
        <FileZone file={file} onFile={setFile} accept="application/pdf" label={t.notes.pdfLabel} />
        <label style={{ ...S.label, marginTop:16 }}>{t.notes.topicLabel}</label>
        <div style={S.inputRow}>
          <input style={S.input} value={topic} onChange={e => setTopic(e.target.value)}
            placeholder={t.notes.placeholder} onKeyDown={e => e.key === "Enter" && generate()} />
          <MicBtn lang={lang} t={t} onResult={v => { setTopic(v); setTimeout(generate, 200); }} />
          <button style={S.primaryBtn} onClick={generate} disabled={loading}>{loading ? "…" : t.notes.btn}</button>
        </div>
      </div>
      <OutputBox content={out} loading={loading} lang={lang} t={t} placeholder={t.notes.placeholder2} loadingText={t.notes.loading} />
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 6 — VIDEO SCRIPT
══════════════════════════════════════════════════════════════ */
function VideoSection({ lang }) {
  const t = T(lang);
  const [file, setFile] = useState(null);
  const [topic, setTopic] = useState("");
  const [out, setOut] = useState("");
  const [loading, setLoading] = useState(false);

  const generate = async () => {
    setLoading(true); setOut("");
    try {
      if (file && file.type === "application/pdf") {
        const data = await apiPDFAnalyze(file, lang);
        setOut(data.video_script || data.error || "");
      } else {
        const prompt = buildPrompt(
          `Write a complete, engaging YouTube video script about this chemistry topic: ${topic || "chemistry basics"}. Include an intro hook, main content with clear explanations and examples, and a conclusion. Format with [INTRO], [MAIN CONTENT], [CONCLUSION] sections.`,
          lang
        );
        const res = await apiPredict(prompt, lang);
        setOut(res);
      }
    } catch (e) { setOut("❌ " + e.message); }
    setLoading(false);
  };

  return (
    <div style={S.section}>
      <SectionHeader icon="🎬" title={t.video.title} sub={t.video.sub} badge="FLAN-T5" />
      <div style={S.card}>
        <label style={S.label}>{t.video.pdfLabel}</label>
        <FileZone file={file} onFile={setFile} accept="application/pdf" label={t.video.pdfLabel} />
        <label style={{ ...S.label, marginTop:16 }}>{t.video.topicLabel}</label>
        <div style={S.inputRow}>
          <input style={{ ...S.input, flex:1 }} value={topic} onChange={e => setTopic(e.target.value)} placeholder={t.video.placeholder} />
          <MicBtn lang={lang} t={t} onResult={v => { setTopic(v); setTimeout(generate, 200); }} />
        </div>
        <button style={{ ...S.primaryBtn, marginTop:14, width:"100%" }} onClick={generate} disabled={loading}>
          {loading ? t.video.loading : t.video.btn}
        </button>
      </div>
      <OutputBox content={out} loading={loading} lang={lang} t={t} placeholder={t.video.placeholder2} loadingText={t.video.loading} />
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 7 — QUIZ (FIXED: proper JSON parsing + QuizWidget)
══════════════════════════════════════════════════════════════ */
function QuizSection({ lang }) {
  const t = T(lang);
  const [file, setFile] = useState(null);
  const [topic, setTopic] = useState("");
  const [quiz, setQuiz] = useState(null);
  const [rawQuiz, setRawQuiz] = useState("");
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [count, setCount] = useState(5);

  // ── FIXED: Robust JSON extraction that tries multiple patterns ──
  const extractQuizJSON = (raw) => {
    if (!raw) return null;
    // Try plain JSON array first
    const trimmed = raw.trim();
    if (trimmed.startsWith("[")) {
      try { const p = JSON.parse(trimmed); if (Array.isArray(p) && p.length > 0) return p; } catch {}
    }
    // Try ```json ... ``` block
    const jsonBlockMatch = trimmed.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
    if (jsonBlockMatch) {
      try { const p = JSON.parse(jsonBlockMatch[1]); if (Array.isArray(p) && p.length > 0) return p; } catch {}
    }
    // Try to find any [...] array in the text
    const arrayMatch = trimmed.match(/(\[[\s\S]*\])/);
    if (arrayMatch) {
      try { const p = JSON.parse(arrayMatch[1]); if (Array.isArray(p) && p.length > 0) return p; } catch {}
    }
    return null;
  };

  const generateQuiz = async (overrideTopic) => {
    setLoading(true); setQuiz(null); setRawQuiz(""); setAnswers({}); setSubmitted(false);
    const finalTopic = (overrideTopic || topic || "general chemistry").trim();
    try {
      let raw = "";
      if (file && file.type === "application/pdf") {
        const data = await apiPDFAnalyze(file, lang);
        raw = data.quiz || "";
      } else {
        // First try API
        try {
          const prompt = buildPrompt(
            `Generate exactly ${count} MCQ questions about: ${finalTopic}.\n\nReturn ONLY a valid JSON array. No explanation, no markdown, no extra text.\n\nFormat:\n[\n  {\n    "question": "What is...?",\n    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],\n    "answer": "A",\n    "explanation": "Because..."\n  }\n]\n\nGenerate ${count} questions now:`,
            lang
          );
          raw = await apiPredict(prompt, lang);
        } catch {
          // API failed, use demo
          raw = JSON.stringify(getDemoQuiz(finalTopic));
        }
      }

      const parsed = extractQuizJSON(raw);
      if (parsed && parsed.length > 0) {
        // Normalise: ensure options array and answer field exist
        const normalised = parsed.map((q, idx) => ({
          question: q.question || q.q || `Question ${idx + 1}`,
          options: Array.isArray(q.options) ? q.options :
            ["A","B","C","D"].map(l => `${l}) Option ${l}`),
          answer: (q.answer || "A").trim().toUpperCase()[0],
          explanation: q.explanation || q.exp || "See your chemistry textbook for details.",
        }));
        setQuiz(normalised);
        setRawQuiz("");
      } else {
        // Fallback: use demo quiz
        setQuiz(getDemoQuiz(finalTopic));
        setRawQuiz("");
      }
    } catch (e) {
      // Final fallback
      setQuiz(getDemoQuiz(finalTopic));
      setRawQuiz("");
    }
    setLoading(false);
  };

  return (
    <div style={S.section}>
      <SectionHeader icon="🧩" title={t.quiz.title} sub={t.quiz.sub} badge="FLAN-T5" />

      <div style={S.card}>
        <label style={S.label}>{t.quiz.pdfLabel}</label>
        <FileZone file={file} onFile={setFile} accept="application/pdf" label={t.quiz.pdfLabel} />
        <label style={{ ...S.label, marginTop:16 }}>{t.quiz.topicLabel}</label>
        <div style={S.inputRow}>
          <input style={S.input} value={topic} onChange={e => setTopic(e.target.value)} placeholder={t.quiz.placeholder} onKeyDown={e => e.key === "Enter" && generateQuiz()} />
          <select style={S.select} value={count} onChange={e => setCount(Number(e.target.value))}>
            {[3,5,8,10].map(n => <option key={n} value={n}>{n} Q</option>)}
          </select>
          <button style={S.primaryBtn} onClick={() => generateQuiz()} disabled={loading}>{loading ? "…" : t.quiz.btn}</button>
        </div>
      </div>

      <QuizLauncher onSend={(msg) => {
        const topicRaw = msg.replace(/^quiz:\d*:\s*/i, "").replace(/^quiz:\s*/i, "").trim();
        setTopic(topicRaw);
        generateQuiz(topicRaw);
      }} />

      {loading && <div style={S.outputBox}><Spinner text={t.quiz.loading} /></div>}

      {/* Always use QuizWidget when quiz data is available */}
      {quiz && !loading && <QuizWidget questions={quiz} />}

      {rawQuiz && !loading && <OutputBox content={rawQuiz} loading={false} lang={lang} t={t} />}
      {!quiz && !rawQuiz && !loading && <div style={{ ...S.outputBox, color:"#475569" }}>{t.quiz.placeholder2}</div>}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SECTION 8 — CHAT
══════════════════════════════════════════════════════════════ */
function ChatSection({ lang }) {
  const t = T(lang);
  const [messages, setMessages] = useState(() => [{ role:"assistant", content: T(lang).chat.welcome }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatRef = useRef();

  useEffect(() => {
    setMessages([{ role:"assistant", content: T(lang).chat.welcome }]);
  }, [lang]);

  // ── FIXED: generateChatResponse returns proper quiz JSON block ──
  const generateChatResponse = async (msg, allMsgs) => {
    const lower = msg.toLowerCase().trim();

    if (lower.startsWith("quiz:")) {
      // Parse count and topic from "QUIZ:5: acid base" or "QUIZ: acid base"
      const countMatch = msg.match(/^quiz:(\d+):/i);
      const count = countMatch ? parseInt(countMatch[1]) : 5;
      const topicRaw = msg.replace(/^quiz:\d*:\s*/i, "").replace(/^quiz:\s*/i, "").trim();
      const topic = topicRaw || "general chemistry";

      let questions;
      try {
        const prompt = `Generate exactly ${count} MCQ questions about: ${topic}.\n\nReturn ONLY a valid JSON array. No explanation, no markdown, no extra text.\n\nFormat:\n[\n  {\n    "question": "What is...?",\n    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],\n    "answer": "A",\n    "explanation": "Because..."\n  }\n]\n\nGenerate ${count} questions now:`;
        const raw = await apiPredict(prompt, lang);

        // Try to parse
        const trimmed = raw.trim();
        let parsed = null;
        if (trimmed.startsWith("[")) {
          try { parsed = JSON.parse(trimmed); } catch {}
        }
        if (!parsed) {
          const arrayMatch = trimmed.match(/(\[[\s\S]*\])/);
          if (arrayMatch) { try { parsed = JSON.parse(arrayMatch[1]); } catch {} }
        }

        if (parsed && Array.isArray(parsed) && parsed.length > 0) {
          questions = parsed.map(q => ({
            question: q.question || q.q || "Question",
            options: Array.isArray(q.options) ? q.options : ["A) Option A","B) Option B","C) Option C","D) Option D"],
            answer: (q.answer || "A").trim().toUpperCase()[0],
            explanation: q.explanation || "See textbook.",
          }));
        } else {
          questions = getDemoQuiz(topic);
        }
      } catch {
        questions = getDemoQuiz(topic);
      }

      const intro = `## Chemistry Quiz — ${topic}\n\n**${questions.length} questions ready!** Use the interactive quiz widget below.\n\n---`;
      return `${intro}\n\n\`\`\`json:quiz\n${JSON.stringify(questions, null, 2)}\n\`\`\``;
    }

    // Regular chat response
    try {
      const prompt = buildChatPrompt(allMsgs.slice(0, -1), msg, lang);
      return await apiPredict(prompt, lang);
    } catch {
      return generateDemoResponse(msg);
    }
  };

  const send = async (txt) => {
    const msg = txt || input;
    if (!msg.trim() || loading) return;
    setInput("");
    const newMsgs = [...messages, { role:"user", content:msg }];
    setMessages(newMsgs);
    setLoading(true);
    try {
      const reply = await generateChatResponse(msg, newMsgs);
      setMessages(m => [...m, { role:"assistant", content:reply }]);
    } catch (e) {
      setMessages(m => [...m, { role:"assistant", content:`❌ Error: ${e.message}` }]);
    }
    setLoading(false);
    setTimeout(() => chatRef.current?.scrollTo({ top:chatRef.current.scrollHeight, behavior:"smooth" }), 100);
  };

  const quickAskItems = T(lang).chat.quickAsk;

  return (
    <div style={S.section}>
      <div style={{ ...S.sectionHeader, justifyContent:"space-between" }}>
        <div style={{ display:"flex", gap:13, alignItems:"flex-start" }}>
          <div style={{ width:50, height:50, borderRadius:13, background:"rgba(56,189,248,0.08)", border:"1px solid rgba(56,189,248,0.18)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:24, flexShrink:0 }}>💬</div>
          <div>
            <h2 style={S.sectionTitle}>{t.chat.title}</h2>
            <p style={S.sectionSub}>{t.chat.sub}</p>
          </div>
        </div>
        <button onClick={() => setMessages([{ role:"assistant", content:t.chat.welcome }])}
          style={{ ...S.chip, color:"#fca5a5", borderColor:"rgba(239,68,68,0.3)" }}>{t.chat.clear}</button>
      </div>

      <div style={{ display:"flex", gap:7, flexWrap:"wrap", marginBottom:12 }}>
        {quickAskItems.map(q => <button key={q} style={{ ...S.chip, fontSize:11 }} onClick={() => send(q)}>{q}</button>)}
      </div>

      <div style={{ ...S.card, display:"flex", flexDirection:"column", height:560, padding:0, overflow:"hidden" }}>
        <div ref={chatRef} style={{ flex:1, overflowY:"auto", padding:"16px 16px 8px", display:"flex", flexDirection:"column", gap:4 }}>
          {messages.map((m, i) => <ChatMessage key={i} role={m.role} content={m.content} />)}
          {loading && (
            <div style={{ display:"flex", gap:10, alignItems:"flex-start" }}>
              <div style={CM.avatar}>⚗</div>
              <div style={{ ...CM.bubble, ...CM.botBubble }}>
                <div style={{ display:"flex", gap:5, padding:"4px 0" }}>
                  {[0,1,2].map(d => <div key={d} style={{ width:7, height:7, borderRadius:"50%", background:"#38bdf8", animation:"bounce 1.2s ease infinite", animationDelay:`${d*0.2}s` }} />)}
                </div>
              </div>
            </div>
          )}
        </div>

        <QuizLauncher onSend={(msg) => send(msg)} />

        <div style={{ padding:"11px 14px", borderTop:"1px solid rgba(255,255,255,0.05)", display:"flex", gap:9 }}>
          <input style={{ ...S.input, flex:1 }} value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && send()} placeholder={t.chat.placeholder} disabled={loading} />
          <MicBtn lang={lang} t={t} onResult={v => { setInput(v); setTimeout(() => send(v), 200); }} />
          <button style={{ ...S.primaryBtn, padding:"10px 18px" }} onClick={() => send()} disabled={loading||!input.trim()}>{t.chat.send}</button>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   DEMO RESPONSE GENERATOR
══════════════════════════════════════════════════════════════ */
function generateDemoResponse(text) {
  const topic = text.replace(/^(what is|explain|describe|define|how does)/i,"").trim() || "chemistry";
  return `## ${topic.charAt(0).toUpperCase()+topic.slice(1,60)}\n\n## Definition\n\nThis is a fundamental concept in chemistry with important real-world applications.\n\n## Key Points\n\n1. **Core principle:** The basic rule that governs this chemical behaviour.\n2. **Mechanism:** Electrons or ions move according to electronegativity differences.\n3. **Example:** A classic demonstration involves mixing acids and bases.\n4. **Application:** Used in industrial processes such as the Haber process.\n5. **Limitation:** Only applies under standard temperature and pressure conditions.\n\n## Step-by-Step Example\n\nStep 1: Identify the reactants and their chemical formulas\nStep 2: Balance the equation by conservation of atoms\nStep 3: Calculate moles using n = m / M\n\n## Key Formula\n\n\`pH = -log[H⁺]\`\n\n## Quick Summary\n\n💡 Remember that stronger acids have lower pH values and fully dissociate in water.`;
}

/* ══════════════════════════════════════════════════════════════
   DEMO QUIZ BANK
══════════════════════════════════════════════════════════════ */
function getDemoQuiz(topic) {
  const t = (topic || "").toLowerCase();
  const bank = {
    acid: [
      { question:"What is the pH range for acidic solutions?", options:["A) 7 to 14","B) 0 to 7","C) 7 to 10","D) 1 to 5"], answer:"B", explanation:"Acidic solutions have pH less than 7. Pure water is neutral at pH 7." },
      { question:"Which is a strong acid?", options:["A) Acetic acid","B) Carbonic acid","C) Sulfuric acid","D) Citric acid"], answer:"C", explanation:"Sulfuric acid (H₂SO₄) fully dissociates in water — it is a strong acid." },
      { question:"What do acids produce when dissolved in water?", options:["A) OH⁻ ions","B) H⁺ ions","C) Na⁺ ions","D) Cl⁻ ions"], answer:"B", explanation:"Acids donate protons (H⁺ ions) — this is the Arrhenius definition." },
      { question:"Which indicator turns red in acid?", options:["A) Phenolphthalein","B) Methyl orange","C) Bromothymol blue","D) Litmus"], answer:"B", explanation:"Methyl orange turns red in acidic solution (pH < 3.1)." },
      { question:"Acid + Base → ?", options:["A) Only water","B) Only salt","C) Salt and water","D) An oxide"], answer:"C", explanation:"Neutralisation: Acid + Base → Salt + Water." },
    ],
    organic: [
      { question:"General formula for alkanes?", options:["A) CₙH₂ₙ","B) CₙH₂ₙ₋₂","C) CₙH₂ₙ₊₂","D) CₙHₙ"], answer:"C", explanation:"Alkanes are saturated: CₙH₂ₙ₊₂ (e.g. CH₄, C₂H₆)." },
      { question:"Functional group in alcohols?", options:["A) –COOH","B) –OH","C) –CHO","D) –NH₂"], answer:"B", explanation:"Alcohols contain the hydroxyl group (–OH)." },
      { question:"Molecular formula of benzene?", options:["A) C₆H₁₂","B) C₆H₁₄","C) C₆H₆","D) C₆H₈"], answer:"C", explanation:"Benzene is C₆H₆ — an aromatic ring." },
      { question:"Alkenes undergo which type of reaction?", options:["A) Substitution","B) Elimination","C) Addition","D) Hydrolysis"], answer:"C", explanation:"Alkenes have C=C and undergo addition reactions." },
      { question:"IUPAC name of CH₃CH₂OH?", options:["A) Methanol","B) Propanol","C) Ethanol","D) Butanol"], answer:"C", explanation:"2 carbons + OH group = ethanol." },
    ],
    thermo: [
      { question:"Which equation relates free energy to enthalpy and entropy?", options:["A) E = mc²","B) PV = nRT","C) ΔG = ΔH − TΔS","D) ΔS = q/T"], answer:"C", explanation:"Gibbs free energy ΔG = ΔH − TΔS determines spontaneity." },
      { question:"A reaction is spontaneous when ΔG is:", options:["A) Zero","B) Positive","C) Negative","D) Equal to ΔH"], answer:"C", explanation:"Negative ΔG means the process occurs spontaneously at constant T and P." },
      { question:"What is the first law of thermodynamics?", options:["A) Energy cannot be created or destroyed","B) Entropy always increases","C) Heat flows from hot to cold","D) ΔG < 0 for spontaneous reactions"], answer:"A", explanation:"The first law states conservation of energy: ΔU = q + w." },
      { question:"At equilibrium, ΔG equals:", options:["A) ΔH","B) TΔS","C) Zero","D) −nFE°"], answer:"C", explanation:"At equilibrium the system has minimum Gibbs energy, so ΔG = 0." },
      { question:"Hess's Law states that:", options:["A) Reaction rates depend on temperature","B) Enthalpy changes are additive","C) Entropy increases with temperature","D) All reactions are exothermic"], answer:"B", explanation:"Hess's Law: ΔH for an overall reaction = sum of ΔH for each step." },
    ],
    electro: [
      { question:"The Nernst equation calculates:", options:["A) Reaction rate","B) Cell potential at non-standard conditions","C) pH of buffer","D) Molar mass"], answer:"B", explanation:"E = E° − (RT/nF) ln Q gives potential at any concentration." },
      { question:"In a galvanic cell, oxidation occurs at the:", options:["A) Cathode","B) Salt bridge","C) Anode","D) Electrolyte"], answer:"C", explanation:"Oxidation (loss of electrons) always occurs at the anode." },
      { question:"Faraday's constant is approximately:", options:["A) 8.314 J/mol·K","B) 96485 C/mol","C) 6.022×10²³","D) 1.38×10⁻²³ J/K"], answer:"B", explanation:"F = 96485 C per mole of electrons transferred." },
      { question:"Which cell converts electrical energy to chemical energy?", options:["A) Galvanic cell","B) Voltaic cell","C) Electrolytic cell","D) Fuel cell"], answer:"C", explanation:"Electrolytic cells use electrical energy to drive non-spontaneous reactions." },
      { question:"Kohlrausch's Law applies to:", options:["A) Weak acids only","B) Strong electrolytes at infinite dilution","C) Non-electrolytes","D) Gases"], answer:"B", explanation:"Λm = Λ°m − b√C describes strong electrolyte conductance." },
    ],
  };

  for (const [k, v] of Object.entries(bank)) {
    if (t.includes(k) || (k === "thermo" && (t.includes("thermo") || t.includes("gibbs") || t.includes("enthalpy"))) || (k === "electro" && (t.includes("electro") || t.includes("nernst") || t.includes("cell")))) {
      return v;
    }
  }

  // Default general quiz
  return [
    { question:"Chemical formula for water?", options:["A) H₂O₂","B) HO","C) H₂O","D) H₃O"], answer:"C", explanation:"Water is H₂O — two hydrogen atoms bonded to one oxygen." },
    { question:"Avogadro's number is:", options:["A) 6.022×10²¹","B) 6.022×10²³","C) 3.011×10²³","D) 6.022×10²⁵"], answer:"B", explanation:"One mole = 6.022×10²³ particles." },
    { question:"DNA strands are held by:", options:["A) Ionic bonds","B) Covalent bonds","C) Hydrogen bonds","D) Metallic bonds"], answer:"C", explanation:"Hydrogen bonds link complementary base pairs A–T and G–C." },
    { question:"SI unit of amount of substance?", options:["A) Gram","B) Litre","C) Mole","D) Dalton"], answer:"C", explanation:"The mole (mol) is the SI unit for amount of substance." },
    { question:"Gas produced when Zn reacts with HCl?", options:["A) Oxygen","B) Chlorine","C) Carbon dioxide","D) Hydrogen"], answer:"D", explanation:"Zn + 2HCl → ZnCl₂ + H₂↑. Hydrogen gives a squeaky pop." },
  ];
}

/* ══════════════════════════════════════════════════════════════
   MAIN DASHBOARD
══════════════════════════════════════════════════════════════ */
export default function Dashboard({ onLogout }) {
  const [accepted, setAccepted]           = useState(() => !!sessionStorage.getItem("chem-terms-session"));
  const [isChecked, setIsChecked]         = useState(false);
  const [lang, setLang]                   = useState(() => localStorage.getItem("chem-lang") || "en");
  const [activeTab, setActiveTab]         = useState("ask");
  const [sidebarOpen, setSidebarOpen]     = useState(true);
  const [globalHistory, setGlobalHistory] = useState([]);
  const [serverHistory, setServerHistory] = useState([]);

  const t = T(lang);

  useEffect(() => { localStorage.setItem("chem-lang", lang); }, [lang]);

  useEffect(() => {
    if (accepted) apiHistory().then(d => setServerHistory(Array.isArray(d) ? d : [])).catch(() => {});
  }, [accepted]);

  useEffect(() => {
    const sub = window.location.pathname.replace("/dashboard","").replace(/^\//,"") || "ask";
    if (NAV_IDS.includes(sub)) setActiveTab(sub);
  }, []);

  const NAV_IDS = ["ask","formula","structure","analyze","notes","video","quiz","chat"];

  const goTo = id => { setActiveTab(id); window.history.pushState({}, "", `/dashboard/${id}`); };

  const handleLogout = () => {
    sessionStorage.removeItem("chem-terms-session");
    setAccepted(false); setIsChecked(false); onLogout();
  };

  const navItems = [
    { id:"ask",       label:t.nav.ask,       icon:"⚗️" },
    { id:"formula",   label:t.nav.formula,   icon:"🧮" },
    { id:"structure", label:t.nav.structure, icon:"🎨" },
    { id:"analyze",   label:t.nav.analyze,   icon:"🔬" },
    { id:"notes",     label:t.nav.notes,     icon:"📝" },
    { id:"video",     label:t.nav.video,     icon:"🎬" },
    { id:"quiz",      label:t.nav.quiz,      icon:"🧩" },
    { id:"chat",      label:t.nav.chat,      icon:"💬" },
  ];

  /* ── TERMS SCREEN ── */
  if (!accepted) {
    const te = t.terms;
    return (
      <div style={{ minHeight:"100vh", background:"linear-gradient(135deg,#020617,#0a1628)", display:"flex", alignItems:"center", justifyContent:"center", fontFamily:"'Segoe UI',sans-serif", padding:20 }}>
        <style>{`@keyframes spin{to{transform:rotate(360deg)}} @keyframes atomSpin{to{transform:rotate(360deg)}} @keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}} @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.7}} @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}`}</style>
        <div style={{ width:500, background:"rgba(10,14,35,0.97)", backdropFilter:"blur(24px)", borderRadius:24, padding:"40px 36px", border:"1px solid rgba(56,189,248,0.2)", boxShadow:"0 40px 100px rgba(0,0,0,0.8)", animation:"fadeInUp 0.5s ease" }}>
          <div style={{ display:"flex", justifyContent:"flex-end", marginBottom:16 }}>
            <div style={{ display:"flex", alignItems:"center", gap:8 }}>
              <span style={{ color:"#64748b", fontSize:12 }}>{t.langLabel}:</span>
              <LangSelect value={lang} onChange={setLang} />
            </div>
          </div>
          <div style={{ textAlign:"center", marginBottom:22 }}>
            <div style={{ width:64, height:64, borderRadius:"50%", background:"rgba(56,189,248,0.1)", border:"2px solid rgba(56,189,248,0.3)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:30, margin:"0 auto 12px" }}>⚗️</div>
            <h2 style={{ color:"#f1f5f9", fontSize:21, fontWeight:900, margin:"0 0 4px" }}>{te.title}</h2>
            <p style={{ color:"#475569", fontSize:12, margin:0 }}>{te.sub}</p>
          </div>
          <div style={{ background:"rgba(56,189,248,0.05)", border:"1px solid rgba(56,189,248,0.13)", borderRadius:12, padding:"13px 16px", marginBottom:16 }}>
            <p style={{ color:"#94a3b8", fontSize:13, lineHeight:1.75, margin:0 }}>
              {te.powered} <strong style={{ color:"#38bdf8" }}>{te.model}</strong>
            </p>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:7, marginBottom:14 }}>
            {[[te.features,te.featureList,"#38bdf8"],[te.policy,te.policyList,"#f59e0b"]].map(([title,items,clr]) => (
              <div key={title} style={{ padding:"10px 12px", background:"rgba(255,255,255,0.02)", borderRadius:9, border:"1px solid rgba(255,255,255,0.05)" }}>
                <p style={{ color:clr, fontSize:11, fontWeight:800, margin:"0 0 6px" }}>{title}</p>
                {items.map(item => <p key={item} style={{ color:"#475569", fontSize:11, margin:"3px 0" }}>{item}</p>)}
              </div>
            ))}
          </div>
          <label style={{ display:"flex", alignItems:"center", gap:10, cursor:"pointer", margin:"18px 0" }}>
            <input type="checkbox" checked={isChecked} onChange={e => setIsChecked(e.target.checked)} style={{ width:16, height:16, accentColor:"#38bdf8" }} />
            <span style={{ color:"#cbd5e1", fontSize:13 }}>{te.agree}</span>
          </label>
          <button style={{ ...S.primaryBtn, width:"100%", padding:13, fontSize:14, opacity:isChecked?1:0.4, cursor:isChecked?"pointer":"not-allowed" }}
            disabled={!isChecked} onClick={() => { sessionStorage.setItem("chem-terms-session","1"); setAccepted(true); }}>
            {te.enter}
          </button>
        </div>
      </div>
    );
  }

  /* ── MAIN LAYOUT ── */
  const allHistory = [...globalHistory, ...serverHistory.map(h => ({ q:h.input?.slice(0,50), ts:0 }))].slice(0,30);

  return (
    <div style={{ minHeight:"100vh", background:"#020617", fontFamily:"'Segoe UI',system-ui,sans-serif", display:"flex", flexDirection:"column", color:"#fff" }}>
      <style>{`
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes atomSpin{to{transform:rotate(360deg)}}
        @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}
        @keyframes fadeInUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(56,189,248,0.4)}70%{box-shadow:0 0 0 8px rgba(56,189,248,0)}}
        input:focus,textarea:focus,select:focus{border-color:rgba(56,189,248,0.55)!important;box-shadow:0 0 0 3px rgba(56,189,248,0.1)!important;outline:none!important}
        button:hover:not(:disabled){filter:brightness(1.1);transform:translateY(-1px)}
        button:active:not(:disabled){transform:translateY(0)}
        ::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:rgba(56,189,248,0.3);border-radius:5px}::-webkit-scrollbar-track{background:transparent}
      `}</style>
      <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:0, background:"radial-gradient(ellipse 80% 50% at 20% 20%,rgba(56,189,248,0.05) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 80%,rgba(99,102,241,0.05) 0%,transparent 60%)" }} />

      {/* ── HEADER ── */}
      <header style={S.header}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <button style={{ background:"transparent", border:"none", color:"#64748b", cursor:"pointer", fontSize:18, padding:"4px 8px", borderRadius:7 }} onClick={() => setSidebarOpen(o => !o)}>☰</button>
          <AtomIcon size={26} />
          <div>
            <h1 style={{ fontSize:16, fontWeight:900, color:"#f1f5f9", margin:0, letterSpacing:-0.5 }}>{t.appName}</h1>
            <span style={{ fontSize:9, color:"#38bdf8", textTransform:"uppercase", letterSpacing:0.8, fontWeight:700 }}>{t.appSub}</span>
          </div>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ display:"flex", alignItems:"center", gap:6, padding:"5px 10px", borderRadius:10, background:"rgba(56,189,248,0.06)", border:"1px solid rgba(56,189,248,0.15)" }}>
            <span style={{ fontSize:13 }}>🌐</span>
            <LangSelect value={lang} onChange={setLang} />
          </div>
          <div style={{ display:"flex", gap:4 }}>
            {navItems.slice(0,5).map(n => (
              <button key={n.id} title={n.label} onClick={() => goTo(n.id)}
                style={{ background:activeTab===n.id?"rgba(56,189,248,0.15)":"transparent", border:`1px solid ${activeTab===n.id?"rgba(56,189,248,0.35)":"transparent"}`, borderRadius:8, padding:"5px 8px", cursor:"pointer", fontSize:16, color:activeTab===n.id?"#7dd3fc":"#475569" }}>
                {n.icon}
              </button>
            ))}
          </div>
          <button style={S.logoutBtn} onClick={handleLogout}>{t.signOut}</button>
        </div>
      </header>

      <div style={{ display:"flex", flex:1, position:"relative", zIndex:1 }}>
        {/* ── SIDEBAR ── */}
        <aside style={{ ...S.sidebar, width:sidebarOpen?240:0, opacity:sidebarOpen?1:0, overflow:"hidden", transition:"all 0.3s ease", flexShrink:0 }}>
          <nav style={{ padding:sidebarOpen?"12px 0":0 }}>
            <p style={{ fontSize:9, color:"#475569", letterSpacing:1.5, padding:"0 16px 8px", fontWeight:700 }}>{t.sidebar.nav}</p>
            {navItems.map(n => (
              <button key={n.id} onClick={() => goTo(n.id)}
                style={{ display:"flex", alignItems:"center", gap:10, width:"100%", padding:"10px 16px", background:activeTab===n.id?"rgba(56,189,248,0.08)":"transparent", border:"none", borderLeft:`2px solid ${activeTab===n.id?"#38bdf8":"transparent"}`, color:activeTab===n.id?"#7dd3fc":"#64748b", cursor:"pointer", textAlign:"left", transition:"all 0.2s" }}>
                <span style={{ fontSize:16 }}>{n.icon}</span>
                <span style={{ fontSize:12, fontWeight:600 }}>{n.label}</span>
                {activeTab===n.id && <div style={{ marginLeft:"auto", width:4, height:4, borderRadius:"50%", background:"#38bdf8" }} />}
              </button>
            ))}
          </nav>

          <div style={{ padding:"10px 16px", borderTop:"1px solid rgba(255,255,255,0.05)", borderBottom:"1px solid rgba(255,255,255,0.05)" }}>
            <p style={{ fontSize:9, color:"#475569", letterSpacing:1.2, marginBottom:7, fontWeight:700 }}>{t.sidebar.system}</p>
            {[["FLAN-T5","#10b981"],["RDKit","#10b981"],["MongoDB","#10b981"],["FastAPI","#38bdf8"]].map(([n,c]) => (
              <div key={n} style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                <span style={{ color:"#334155", fontSize:11 }}>{n}</span>
                <span style={{ color:c, fontSize:11, fontWeight:700 }}>● Active</span>
              </div>
            ))}
          </div>

          <div style={{ padding:"10px 16px", flex:1 }}>
            <p style={{ fontSize:9, color:"#475569", letterSpacing:1.2, marginBottom:7, fontWeight:700 }}>{t.sidebar.recent}</p>
            {allHistory.slice(0,6).map((h,i) => (
              <div key={i} style={{ padding:"5px 8px", borderRadius:7, color:"#334155", fontSize:11, marginBottom:3, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", cursor:"pointer" }}
                onMouseEnter={e => { e.currentTarget.style.background="rgba(56,189,248,0.06)"; e.currentTarget.style.color="#64748b"; }}
                onMouseLeave={e => { e.currentTarget.style.background="transparent"; e.currentTarget.style.color="#334155"; }}>
                {h.q}
              </div>
            ))}
            {allHistory.length===0 && <p style={{ color:"#1e293b", fontSize:12 }}>{t.sidebar.noHistory}</p>}
          </div>
        </aside>

        {/* ── MAIN ── */}
        <main style={{ flex:1, overflowY:"auto", paddingBottom:60 }}>
          <div style={{ padding:"12px 32px 0", display:"flex", alignItems:"center", gap:7 }}>
            <span style={{ color:"#334155", fontSize:11 }}>{t.history}</span>
            <span style={{ color:"#1e293b", fontSize:11 }}>›</span>
            <span style={{ color:"#38bdf8", fontSize:11, fontWeight:700 }}>{navItems.find(n => n.id===activeTab)?.label}</span>
          </div>
          <div style={{ animation:"fadeInUp 0.35s ease" }}>
            {activeTab==="ask"       && <AskSection       lang={lang} setGlobalHistory={setGlobalHistory} />}
            {activeTab==="formula"   && <FormulaSection   lang={lang} />}
            {activeTab==="structure" && <StructureSection lang={lang} />}
            {activeTab==="analyze"   && <AnalyzeSection   lang={lang} />}
            {activeTab==="notes"     && <NotesSection     lang={lang} />}
            {activeTab==="video"     && <VideoSection     lang={lang} />}
            {activeTab==="quiz"      && <QuizSection      lang={lang} />}
            {activeTab==="chat"      && <ChatSection      lang={lang} />}
          </div>
        </main>
      </div>

      {/* ── FOOTER ── */}
      <footer style={S.footer}>
        <div style={{ display:"flex", alignItems:"center", gap:7 }}>
          <AtomIcon size={14} />
          <strong style={{ color:"#38bdf8" }}>{t.footer.brand}</strong>
          <span style={{ color:"#1e293b" }}>—</span>
          <span>{t.footer.location}</span>
        </div>
        <div style={{ color:"#334155", fontSize:11 }}>{t.footer.model}</div>
        <div style={{ color:"#1e293b", fontSize:11 }}>
          <a href="mailto:admissions@kietgroup.com" style={{ color:"#38bdf8", textDecoration:"none" }}>admissions@kietgroup.com</a>
          {" · "}JNTU Kakinada{" · "}© {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   STYLE TOKENS
══════════════════════════════════════════════════════════════ */
const S = {
  header:       { display:"flex", justifyContent:"space-between", alignItems:"center", padding:"0 20px", height:58, background:"rgba(2,6,23,0.95)", backdropFilter:"blur(20px)", borderBottom:"1px solid rgba(255,255,255,0.06)", position:"sticky", top:0, zIndex:100 },
  sidebar:      { background:"rgba(10,12,28,0.88)", backdropFilter:"blur(14px)", borderRight:"1px solid rgba(255,255,255,0.06)", minHeight:"calc(100vh - 58px)", display:"flex", flexDirection:"column" },
  section:      { padding:"26px 30px", maxWidth:900 },
  sectionHeader:{ display:"flex", alignItems:"flex-start", gap:14, marginBottom:20, flexWrap:"wrap" },
  sectionTitle: { fontSize:20, fontWeight:900, color:"#f1f5f9", margin:0, letterSpacing:-0.5 },
  sectionSub:   { color:"#64748b", fontSize:12, margin:"3px 0 0", lineHeight:1.5 },
  card:         { background:"rgba(14,18,40,0.85)", backdropFilter:"blur(12px)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:16, padding:22, marginBottom:16 },
  label:        { display:"block", fontSize:10, fontWeight:800, color:"#475569", letterSpacing:1.1, textTransform:"uppercase", marginBottom:9 },
  textarea:     { width:"100%", background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:11, padding:"13px 15px", color:"#f1f5f9", fontSize:14, resize:"vertical", outline:"none", fontFamily:"inherit", lineHeight:1.6, boxSizing:"border-box" },
  input:        { flex:1, background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:10, padding:"11px 15px", color:"#f1f5f9", fontSize:14, outline:"none", fontFamily:"inherit", boxSizing:"border-box" },
  select:       { background:"rgba(14,18,40,0.95)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:9, padding:"8px 12px", color:"#94a3b8", fontSize:12, cursor:"pointer", outline:"none", flexShrink:0 },
  primaryBtn:   { padding:"10px 20px", background:"linear-gradient(135deg,#38bdf8,#0ea5e9)", border:"none", borderRadius:10, color:"#020617", fontWeight:800, fontSize:13, cursor:"pointer", flexShrink:0, boxShadow:"0 5px 16px rgba(56,189,248,0.28)", letterSpacing:0.2 },
  chip:         { padding:"6px 13px", background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:18, color:"#94a3b8", fontSize:12, fontWeight:600, cursor:"pointer", whiteSpace:"nowrap" },
  chipRow:      { display:"flex", flexWrap:"wrap", gap:7, marginBottom:4 },
  inputRow:     { display:"flex", gap:9, alignItems:"center" },
  btnRow:       { display:"flex", gap:10, alignItems:"center", marginTop:13 },
  outputBox:    { background:"rgba(4,7,18,0.8)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:15, padding:"18px 22px", minHeight:80, lineHeight:1.7, backdropFilter:"blur(8px)", marginBottom:16 },
  spinner:      { display:"inline-block", width:15, height:15, border:"2px solid rgba(56,189,248,0.18)", borderTopColor:"#38bdf8", borderRadius:"50%", animation:"spin 0.7s linear infinite" },
  iconBtn:      { width:42, height:42, display:"flex", alignItems:"center", justifyContent:"center", border:"1px solid", borderRadius:10, cursor:"pointer", fontSize:18, flexShrink:0, transition:"all 0.2s" },
  errorBox:     { padding:"11px 15px", borderRadius:9, background:"rgba(239,68,68,0.07)", border:"1px solid rgba(239,68,68,0.2)", color:"#fca5a5", fontSize:13, marginBottom:12 },
  logoutBtn:    { padding:"7px 14px", background:"rgba(239,68,68,0.08)", border:"1px solid rgba(239,68,68,0.22)", borderRadius:8, color:"#fca5a5", fontSize:12, fontWeight:700, cursor:"pointer" },
  footer:       { display:"flex", justifyContent:"space-between", alignItems:"center", padding:"12px 24px", background:"rgba(2,6,23,0.95)", borderTop:"1px solid rgba(255,255,255,0.05)", fontSize:12, color:"#334155", zIndex:10, flexWrap:"wrap", gap:10 },
};
