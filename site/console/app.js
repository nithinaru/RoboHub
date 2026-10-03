import * as THREE from "https://unpkg.com/three@0.170.0/build/three.module.js";
import {
  createTask,
  logGateAudit,
  robohubConfig,
  searchTrajectories,
  setTaskStatus,
  subscribeGateProgress,
} from "./lib/supabaseClient.js";

const GATES = [
  ["fps", "Frame rate"],
  ["length", "Clip length"],
  ["scene", "Scene audit"],
  ["one_object", "One red object"],
  ["no_duplicate", "No duplicate object"],
  ["hand_visible", "Hand visible"],
  ["block_visible", "Block tracked"],
  ["events", "Grasp, lift, release"],
  ["coupled", "Block follows hand"],
  ["lift", "Lift height"],
  ["carry", "Carry distance"],
  ["video_in_bowl", "Ends in bowl"],
  ["ik", "IK joint limits"],
  ["vel", "Joint velocity"],
  ["acc", "Joint acceleration"],
  ["jerk", "Joint jerk"],
  ["close_before_lift", "Close before lift"],
  ["open_over_bowl", "Open over bowl"],
  ["sim_success", "MuJoCo success"],
  ["self_collision", "No self-collision"],
  ["episode_len", "Episode length"],
];

const matrix = document.querySelector("#gates");
const verdict = document.querySelector("#verdict");
const statusEl = document.querySelector("#status");
const hits = document.querySelector("#hits");

const cells = new Map();
for (const [id, name] of GATES) {
  const el = document.createElement("article");
  el.className = "gate";
  el.innerHTML = `<strong>${name}</strong><span>waiting</span>`;
  matrix.append(el);
  cells.set(id, el);
}

function paintGate(row) {
  const el = cells.get(row.id);
  if (!el) return;
  el.classList.remove("pass", "fail");
  if (row.passed === true) el.classList.add("pass");
  if (row.passed === false) el.classList.add("fail");
  el.querySelector("span").textContent = row.passed === false ? row.why || "fail" : String(row.value ?? "pass");
}

function resetGates() {
  verdict.textContent = "running";
  verdict.className = "pill";
  for (const el of cells.values()) {
    el.className = "gate";
    el.querySelector("span").textContent = "waiting";
  }
}

function applyAudit(audit) {
  const gates = audit.gates || audit.gate_audit?.gates || [];
  for (const row of gates) paintGate(row);
  const failed = gates.filter((row) => row.passed === false);
  verdict.textContent = failed.length ? "rejected" : "accepted";
  verdict.className = "pill " + (failed.length ? "fail" : "pass");
}

const cfg = robohubConfig({
  SUPABASE_URL: new URLSearchParams(location.search).get("supabase") || localStorage.getItem("ROBOHUB_SUPABASE_URL") || "",
  SUPABASE_ANON_KEY: localStorage.getItem("ROBOHUB_SUPABASE_ANON_KEY") || "",
});

const samples = [
  { id: "demo-a", label: "pick into bowl", distance: 0.04, xyz: path(0.2, 0.05) },
  { id: "demo-b", label: "short lift", distance: 0.21, xyz: path(0.08, 0.02) },
  { id: "demo-c", label: "wide carry", distance: 0.33, xyz: path(0.28, -0.08) },
];

function path(reach, side) {
  const pts = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    pts.push([0.12 + reach * t, side * Math.sin(t * Math.PI), 0.02 + 0.08 * Math.sin(t * Math.PI)]);
  }
  return pts;
}

document.querySelector("#search").addEventListener("click", async () => {
  const query = embed(samples[0].xyz);
  let rows = samples.map((s) => ({ id: s.id, dataset_path: s.label, distance: s.distance, xyz: s.xyz }));
  if (cfg.url && cfg.key) {
    const remote = await searchTrajectories(cfg, query, 8);
    if (Array.isArray(remote) && remote.length) {
      rows = remote.map((row) => ({ ...row, xyz: samples[0].xyz }));
    }
  }
  hits.replaceChildren();
  for (const row of rows) {
    const li = document.createElement("li");
    li.textContent = `${row.dataset_path || row.id} · cosine ${Number(row.distance).toFixed(3)}`;
    li.addEventListener("click", () => drawPlot(row.xyz || samples[0].xyz));
    hits.append(li);
  }
  drawPlot(rows[0].xyz || samples[0].xyz);
});

function embed(xyz) {
  const flat = xyz.flat();
  const out = new Array(1536).fill(0);
  for (let i = 0; i < out.length; i++) {
    const t = (i / (out.length - 1)) * (flat.length - 1);
    const j = Math.floor(t);
    const f = t - j;
    out[i] = flat[j] * (1 - f) + flat[Math.min(j + 1, flat.length - 1)] * f;
  }
  return out;
}

function drawPlot(xyz) {
  const canvas = document.querySelector("#plot");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#3ecf8e";
  ctx.lineWidth = 2;
  ctx.beginPath();
  xyz.forEach((p, i) => {
    const x = 24 + (p[0] - 0.05) * (w - 48) / 0.4;
    const y = h - 24 - p[2] * (h - 48) / 0.15;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

const renderer = new THREE.WebGLRenderer({ canvas: document.querySelector("#arm-view"), antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#070707");
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 20);
camera.position.set(0.42, 0.22, 0.36);
camera.lookAt(0, 0.14, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(1, -1, 2);
scene.add(key);

function link(parent, length, radius, color) {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, length, 16),
    new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.45 }),
  );
  mesh.position.y = length / 2;
  const pivot = new THREE.Group();
  pivot.add(mesh);
  parent.add(pivot);
  const tip = new THREE.Group();
  tip.position.y = length;
  pivot.add(tip);
  return { pivot, tip };
}

const root = new THREE.Group();
scene.add(root);
const base = new THREE.Mesh(
  new THREE.CylinderGeometry(0.04, 0.05, 0.03, 24),
  new THREE.MeshStandardMaterial({ color: 0x2a2a2a }),
);
root.add(base);
const shoulder = new THREE.Group();
shoulder.position.y = 0.03;
root.add(shoulder);
const upper = link(shoulder, 0.12, 0.012, 0xd6d6d6);
const elbow = link(upper.tip, 0.11, 0.01, 0x3ecf8e);
const wrist = link(elbow.tip, 0.06, 0.008, 0xededed);
const jawL = link(wrist.tip, 0.03, 0.004, 0x9a9a9a);
const jawR = link(wrist.tip, 0.03, 0.004, 0x9a9a9a);
jawL.pivot.position.x = -0.008;
jawR.pivot.position.x = 0.008;

let frames = demoJoints();
let frame = 0;

function demoJoints() {
  const seq = [];
  for (let i = 0; i < 90; i++) {
    const t = i / 90;
    seq.push([
      Math.sin(t * Math.PI) * 0.4,
      -0.6 + Math.sin(t * Math.PI) * 0.5,
      0.9 - t * 0.4,
      0.6,
      t * 1.2,
      t > 0.35 && t < 0.75 ? -0.1 : 1.2,
    ]);
  }
  return seq;
}

function applyJoints(q) {
  shoulder.rotation.z = q[0];
  upper.pivot.rotation.z = q[1];
  elbow.pivot.rotation.z = q[2];
  wrist.pivot.rotation.z = q[3];
  wrist.pivot.rotation.y = q[4];
  const open = q[5];
  jawL.pivot.rotation.z = open * 0.4;
  jawR.pivot.rotation.z = -open * 0.4;
}

function resize() {
  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function tick() {
  resize();
  applyJoints(frames[frame % frames.length]);
  frame += 1;
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
drawPlot(samples[0].xyz);

function sampleAudit(failDuplicate) {
  return GATES.map(([id, name], index) => {
    const failed = failDuplicate && id === "no_duplicate";
    return {
      id,
      name,
      passed: !failed,
      value: failed ? 11 : index,
      limit: "spec",
      why: failed ? "a second red block appears on 11 frames" : "",
    };
  });
}

async function streamLocal(gates) {
  resetGates();
  for (const row of gates) {
    paintGate(row);
    await new Promise((resolve) => setTimeout(resolve, 90));
  }
  applyAudit({ gates });
}

document.querySelector("#launch").addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = document.querySelector("#prompt").value.trim();
  const arm = document.querySelector("#arm-type").value;
  statusEl.textContent = "Starting run…";
  resetGates();
  frames = demoJoints();
  frame = 0;
  let taskId = "local-ui";
  if (cfg.url && cfg.key) {
    const task = await createTask(cfg, { prompt, armType: arm, status: "gating" });
    taskId = task.id;
    statusEl.textContent = `Task ${taskId} · live gate stream`;
    subscribeGateProgress(cfg, taskId, (row) => {
      applyAudit(row.gate_audit || {});
      if (row.gate_audit?.joints) frames = row.gate_audit.joints;
    });
    const gates = sampleAudit(false);
    await logGateAudit(cfg, { taskId, gateAudit: gates, passed: true });
    await setTaskStatus(cfg, taskId, "accepted");
    applyAudit({ gates });
    return;
  }
  statusEl.textContent = `Offline preview for “${prompt}” on ${arm}. Connect Supabase to stream a real task.`;
  await streamLocal(sampleAudit(Math.random() < 0.35));
});
