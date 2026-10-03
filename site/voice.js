// Gemini Live voice agent. Speak a task; the model can start a training run.
// Audio in is 16 kHz PCM. Audio out is 24 kHz PCM. The key never leaves this browser except on the Live socket.
(() => {
  const MODEL = "gemini-3.8-live";
  const $ = (s) => document.querySelector(s);
  const btn = document.querySelector("#mic");
  const prompt = $("#prompt");
  const keyIn = $("#rw-key");
  const keyRow = $("#key-row");
  const line = $("#route-line");
  if (!btn || !prompt) return;

  let ws = null;
  let mic = null;
  let capture = null;
  let playCtx = null;
  let nextAt = 0;
  let heard = "";

  const say = (text, bad = false) => {
    if (!line) return;
    line.classList.toggle("is-no", bad);
    line.textContent = text;
  };

  function b64(bytes) {
    let s = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) s += String.fromCharCode(...bytes.subarray(i, i + chunk));
    return btoa(s);
  }

  function pcm16k(float32, rate) {
    const ratio = rate / 16000;
    const n = Math.floor(float32.length / ratio);
    const out = new Int16Array(n);
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, float32[Math.floor(i * ratio)]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function play(b64pcm) {
    if (!playCtx) playCtx = new AudioContext({ sampleRate: 24000 });
    const bin = atob(b64pcm);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const pcm = new Int16Array(bytes.buffer);
    const buf = playCtx.createBuffer(1, pcm.length, 24000);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = playCtx.createBufferSource();
    src.buffer = buf;
    src.connect(playCtx.destination);
    const start = Math.max(playCtx.currentTime + 0.02, nextAt);
    src.start(start);
    nextAt = start + buf.duration;
  }

  function train(promptText, budget) {
    const text = String(promptText || "").replace(/\s+/g, " ").trim();
    if (text.length < 6) return { ok: false, error: "Need a short task sentence." };
    prompt.value = text.slice(0, 160);
    prompt.dispatchEvent(new Event("input", { bubbles: true }));
    const usd = Number(budget);
    if (window.Route && window.Route.BUDGETS.includes(usd)) {
      window.Route.budget = usd;
      document.querySelectorAll("#budget-seg [role=radio]").forEach((n, i) => {
        const on = window.Route.BUDGETS[i] === usd;
        n.classList.toggle("on", on);
        n.setAttribute("aria-checked", String(on));
      });
    }
    queueMicrotask(() => $("#composer").requestSubmit());
    return { ok: true, prompt: prompt.value, budget: window.Route ? window.Route.budget : usd };
  }

  function onTool(msg) {
    const calls = msg.toolCall?.functionCalls || [];
    if (!calls.length || ws?.readyState !== WebSocket.OPEN) return;
    const functionResponses = calls.map((fc) => {
      const args = fc.args || {};
      let result;
      if (fc.name === "start_training") result = train(args.prompt, args.budget);
      else if (fc.name === "set_budget") {
        const usd = Number(args.budget);
        if (window.Route && window.Route.BUDGETS.includes(usd)) window.Route.budget = usd;
        result = { ok: window.Route ? window.Route.BUDGETS.includes(usd) : false, budget: usd };
      }
      else result = { ok: false, error: "unknown tool" };
      return { id: fc.id, name: fc.name, response: { result } };
    });
    ws.send(JSON.stringify({ toolResponse: { functionResponses } }));
  }

  function onMessage(raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }
    if (msg.setupComplete) say("Listening. Say the task, then say train.");
    const content = msg.serverContent;
    if (content?.inputTranscription?.text) {
      heard = content.inputTranscription.text;
      if (!prompt.value || document.activeElement !== prompt) {
        prompt.value = heard.slice(0, 160);
        prompt.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }
    if (content?.outputTranscription?.text) say(content.outputTranscription.text);
    for (const part of content?.modelTurn?.parts || []) {
      const audio = part.inlineData || part.inline_data;
      if (audio?.data) play(audio.data);
    }
    if (msg.toolCall) onTool(msg);
  }

  async function start() {
    const key = (keyIn?.value || "").trim();
    if (!key) {
      if (keyRow) { keyRow.hidden = false; keyRow.classList.add("need"); }
      say("Add a Gemini API key, then press Voice again.", true);
      return;
    }
    playCtx = new AudioContext();
    await playCtx.resume();
    const url = `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=${encodeURIComponent(key)}`;
    ws = new WebSocket(url);
    ws.addEventListener("message", (ev) => {
      if (typeof ev.data === "string") onMessage(ev.data);
      else ev.data.text().then(onMessage);
    });
    ws.addEventListener("close", (ev) => {
      btn.classList.remove("on");
      btn.setAttribute("aria-pressed", "false");
      if (ev.code !== 1000) say(ev.reason || "Voice session closed.", true);
    });
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, { once: true });
      ws.addEventListener("error", () => reject(new Error("Could not open the Gemini Live session.")), { once: true });
    });
    ws.send(JSON.stringify({
      setup: {
        model: `models/${MODEL}`,
        generationConfig: {
          responseModalities: ["AUDIO"],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: "Puck" } } },
        },
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        systemInstruction: {
          parts: [{
            text: "You are the RoboHub voice agent for a low-cost SO-101 robot arm. The user speaks a manipulation task. " +
              "Call start_training when they want a demonstration filmed and trained. The prompt must be one short imperative sentence. " +
              "Budget is 2, 4, 6, 8, or 10. Use 2 unless they name a higher budget. Confirm in one short sentence. Do not invent gate scores.",
          }],
        },
        tools: [{
          functionDeclarations: [
            {
              name: "start_training",
              description: "Film the task with Gemini Veo and start the RoboHub training flow.",
              parameters: {
                type: "OBJECT",
                properties: {
                  prompt: { type: "STRING", description: "Task sentence, for example put the red block in the bowl" },
                  budget: { type: "INTEGER", description: "One of 2, 4, 6, 8, 10" },
                },
                required: ["prompt"],
              },
            },
            {
              name: "set_budget",
              description: "Change the dollar budget for the next run without starting it.",
              parameters: {
                type: "OBJECT",
                properties: { budget: { type: "INTEGER" } },
                required: ["budget"],
              },
            },
          ],
        }],
      },
    }));
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
    mic = stream;
    capture = new AudioContext();
    const source = capture.createMediaStreamSource(stream);
    const node = capture.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (ev) => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      const pcm = pcm16k(ev.inputBuffer.getChannelData(0), capture.sampleRate);
      const bytes = new Uint8Array(pcm.buffer);
      ws.send(JSON.stringify({
        realtimeInput: { audio: { data: b64(bytes), mimeType: "audio/pcm;rate=16000" } },
      }));
    };
    source.connect(node);
    node.connect(capture.destination);
    btn.classList.add("on");
    btn.setAttribute("aria-pressed", "true");
    say("Voice on. Connecting to Gemini.");
  }

  function stop() {
    mic?.getTracks().forEach((t) => t.stop());
    mic = null;
    capture?.close();
    capture = null;
    if (ws && ws.readyState === WebSocket.OPEN) ws.close(1000);
    ws = null;
    btn.classList.remove("on");
    btn.setAttribute("aria-pressed", "false");
    say("Voice off.");
  }

  btn.addEventListener("click", async () => {
    if (btn.classList.contains("on")) { stop(); return; }
    try { await start(); }
    catch (e) { stop(); say(e.message || "Microphone or Gemini Live failed.", true); }
  });
})();
