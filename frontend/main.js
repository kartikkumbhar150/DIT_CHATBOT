const apiBase = "/";
let listening = false;
const statusEl = document.getElementById("status");
const convEl = document.getElementById("conversation");
const btn = document.getElementById("toggle-listen");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const langSelect = document.getElementById("lang-select");
const sessionId = "session-" + Math.random().toString(36).slice(2, 9);

let finalTranscript = "";
let awake = false;
let ttsInterrupted = false;
let stopRequested = false;
let lastUserLang = "en"; // default language
let liveBubble = null;   // interim transcript bubble
const MAX_MESSAGES = 200; // trim conversation for performance

// ----------- Keyword Detection -----------
function containsWake(text) {
  return /\b(hey dit|hello dit|hello|hi|hi dit)\b/i.test(text);
}
function containsStop(text) {
  return /\b(stop|okay stop|ok stop|wait|wail)\b/i.test(text);
}
function containsFullStop(text) {
  return /\b(stop listening|exit)\b/i.test(text);
}

// ----------- Control Functions -----------
function stopSpeaking() {
  window.speechSynthesis.cancel();
  ttsInterrupted = true;
  awake = true;
  statusEl.textContent = "Stopped speaking. Ask me a new question...";

  // 🔑 Restart recognition after interrupt
  if (recognition && listening) {
    try { recognition.start(); } catch {}
  }
}

function stopAll() {
  if (recognition) recognition.stop();
  window.speechSynthesis.cancel();
  listening = false;
  awake = false;
  stopRequested = true;
  statusEl.textContent = "Stopped completely. Say 'hello' to wake again.";
}

// ----------- Speech Recognition Setup -----------
let recognition;
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = langSelect?.value || "en-IN";

  recognition.onstart = () => {
    statusEl.textContent = "Listening... Say hello to wake.";
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const transcript = event.results[i][0].transcript.trim().toLowerCase();

      // 🚨 Immediate stop detection
      if (containsStop(transcript)) {
        stopSpeaking();
        return;
      }
      if (containsFullStop(transcript)) {
        stopAll();
        return;
      }

      if (event.results[i].isFinal) {
        finalTranscript = transcript;

        if (liveBubble) {
          liveBubble.textContent = finalTranscript;
          liveBubble.classList.remove("live");
          liveBubble = null;
        } else {
          appendConversation(finalTranscript, "user");
        }

        // cancel old speech before handling new input
        stopSpeaking();
        handleFinalSpeech(finalTranscript);
      } else {
        interim += transcript;

        // 🚨 If user starts talking while bot is speaking → cut speech
        if (!ttsInterrupted && transcript.length > 2) {
          stopSpeaking();
        }

        if (!liveBubble) {
          liveBubble = document.createElement("div");
          liveBubble.classList.add("message", "user", "live");
          liveBubble.textContent = interim;
          convEl.appendChild(liveBubble);
        } else {
          liveBubble.textContent = interim;
        }
        convEl.scrollTop = convEl.scrollHeight;
      }
    }
  };

  recognition.onerror = () => {
    statusEl.textContent = "Speech recognition error.";
  };

  recognition.onend = () => {
    if (listening) {
      setTimeout(() => {
        try { recognition.start(); } catch {}
      }, 300);
    }
  };
} else {
  statusEl.textContent = "Speech Recognition not supported.";
}

// ----------- Language Dropdown -----------
if (langSelect) {
  langSelect.addEventListener("change", () => {
    if (recognition) recognition.lang = langSelect.value;
  });
}

// ----------- Translation Helpers -----------
async function translateText(text, sourceLang, targetLang) {
  try {
    const res = await fetch("https://libretranslate.de/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        q: text,
        source: sourceLang,
        target: targetLang,
        format: "text"
      }),
    });
    const data = await res.json();
    return data.translatedText || text;
  } catch {
    return text; // fallback
  }
}

async function translateToEnglish(text, sourceLang = "auto") {
  if (sourceLang && sourceLang.includes("-")) {
    sourceLang = sourceLang.split("-")[0];
  }
  lastUserLang = sourceLang || "en";
  return await translateText(text, sourceLang || "auto", "en");
}

// ----------- Speech Handling -----------
async function handleFinalSpeech(text) {
  if (containsFullStop(text)) {
    stopAll();
    return;
  }
  if (containsStop(text)) {
    stopSpeaking();
    return;
  }
  if (!awake) {
    if (containsWake(text)) {
      stopRequested = false;
      awake = true;
      statusEl.textContent = "Awake. Ask your question...";
      appendConversation("Yes, how can I help?", "bot");
      speak("Yes, how can I help?", "en");
    }
  } else {
    statusEl.textContent = "Processing...";
    let spokenLang = recognition.lang || langSelect.value || "en";
    if (spokenLang.includes("-")) spokenLang = spokenLang.split("-")[0];
    lastUserLang = spokenLang;

    const englishText = await translateToEnglish(text, spokenLang);
    handleBotResponse(sendQuery(englishText));
  }
}

// ----------- Backend Query -----------
async function sendQuery(q) {
  const res = await fetch(apiBase + "api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q, session_id: sessionId }),
  });
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { error: text || `Server returned ${res.status}` };
  }
  if (!res.ok) {
    throw new Error(body.detail || body.error || `Server returned ${res.status}`);
  }
  return body;
}

// ----------- Conversation UI -----------
function appendConversation(text, sender = "bot") {
  const d = document.createElement("div");
  d.classList.add("message", sender);
  if (sender === "bot") {
    try {
      const html = DOMPurify.sanitize(marked.parse(text || ""), {
        USE_PROFILES: { html: true },
      });
      d.innerHTML = html;
    } catch {
      d.textContent = text;
    }
  } else {
    d.textContent = text;
  }
  convEl.appendChild(d);
  convEl.scrollTop = convEl.scrollHeight;

  // trim messages
  if (convEl.children.length > MAX_MESSAGES) {
    convEl.removeChild(convEl.firstChild);
  }
}

// ----------- Speech Synthesis -----------
let voices = [];
window.speechSynthesis.onvoiceschanged = () => {
  try {
    voices = window.speechSynthesis.getVoices();
  } catch { voices = []; }
};

function speak(text, lang = "en") {
  if (!text) return;
  if (containsStop(text) || containsFullStop(text)) return;

  let cleanText = text
    .replace(/#+\s*/g, "")
    .replace(/[-*]\s+/g, "")
    .replace(/^\d+\.\s+/gm, "");

  ttsInterrupted = false;
  const parts = cleanText.split(/(?<=[.!?])\s+/).filter(Boolean);

  const speakNext = () => {
    if (ttsInterrupted || parts.length === 0) {
      if (recognition && listening) {
        try { recognition.start(); } catch {}
      }
      return;
    }

    const u = new SpeechSynthesisUtterance(parts.shift());
    const prefer = voices.find(v => v.lang.toLowerCase().startsWith(lang));
    if (prefer) u.voice = prefer;

    u.lang = (lang === "hi") ? "hi-IN" : (lang === "mr") ? "mr-IN" : "en-IN";

    u.onstart = () => {
      statusEl.textContent = "Speaking...";
      if (recognition && listening) {
        try { recognition.start(); } catch {}
      }
    };

    u.onend = () => {
      if (!window.speechSynthesis.speaking) {
        statusEl.textContent = "Ready for next question...";
        if (recognition && listening) {
          try { recognition.start(); } catch {}
        }
      }
      if (!ttsInterrupted) speakNext();
    };

    window.speechSynthesis.speak(u);
  };

  speakNext();
}

// ----------- Bot Response Handler -----------
function showTyping() {
  const typing = document.createElement("div");
  typing.classList.add("typing");
  typing.innerHTML = "<span></span><span></span><span></span>";
  convEl.appendChild(typing);
  convEl.scrollTop = convEl.scrollHeight;
  return typing;
}

function handleBotResponse(promise) {
  const typing = showTyping();
  promise
    .then(async (resp) => {
      typing.remove();
      if (stopRequested) {
        statusEl.textContent = "Stopped. Waiting for hello...";
        return;
      }

      let reply = resp.answer || "I couldn’t understand that.";

      if (lastUserLang !== "en") {
        try {
          reply = await translateText(reply, "en", lastUserLang);
        } catch { /* fallback */ }
      }

      appendConversation(reply, "bot");
      speak(reply, lastUserLang);
      awake = true;
      statusEl.textContent = "Ready for next question...";
    })
    .catch((err) => {
      typing.remove();
      if (!stopRequested) {
        statusEl.textContent = "Error contacting backend: " + err.message;
      }
    });
}

// ----------- Manual Toggle Button -----------
btn.addEventListener("click", () => {
  listening = !listening;
  if (listening && recognition) {
    recognition.start();
    btn.textContent = "Stop Listening";
  } else {
    if (recognition) recognition.stop();
    btn.textContent = "Start Listening";
    statusEl.textContent = "Stopped.";
  }
});

// ----------- Text Input Handler -----------
async function handleTextSubmit() {
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  statusEl.textContent = "Processing...";
  appendConversation(text, "user");

  stopSpeaking(); // cancel old speech if new query

  const englishText = await translateToEnglish(text, recognition?.lang || langSelect.value);
  handleBotResponse(sendQuery(englishText));
}
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleTextSubmit();
});
sendBtn.addEventListener("click", handleTextSubmit);

// ----------- Auto Start Listening -----------
window.addEventListener("load", () => {
  if (recognition) {
    listening = true;
    recognition.start();
    statusEl.textContent = "Listening... Say hello to wake.";
  }
});
