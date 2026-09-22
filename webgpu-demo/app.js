import { createApp, reactive } from "https://cdn.jsdelivr.net/npm/vue@3.5.21/dist/vue.esm-browser.prod.js";
import { normalizeLocale, SUPPORTED_LOCALES, translate } from "./i18n.js";

const worker = new Worker("worker.js", { type: "module" });

const $ = (selector) => document.querySelector(selector);
const loadButton = $("#load");
const runButton = $("#run");
const modelSelect = $("#model-select");
const uiMode = $("#ui-mode");
const languageSelect = $("#language-select");
const models = {
  "qwen3-0.6b": {
    name: "Qwen3 0.6B", short: "Qwen3 · 0.6B", size: "639 MB",
    url: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
    noticeKey: "model.noticeSmall", noticeClass: "mobile",
  },
  "minicpm5-2b": {
    name: "MiniCPM5 2B", short: "MiniCPM5 · 2B", size: "1.56 GB",
    url: "https://huggingface.co/openbmb/MiniCPM5-2B-GGUF",
    noticeKey: "model.noticeMedium", noticeClass: "",
  },
  "qwen3.5-4b": {
    name: "Qwen3.5 4B", short: "Qwen3.5 · 4B", size: "3.01 GB",
    url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    noticeKey: "model.noticeLarge", noticeClass: "desktop-heavy",
  },
};
const MIN_OPTIONS = 2;
const MAX_OPTIONS = 20;
const optionList = $("#option-list");
const addOptionButton = $("#add-option");
const removeOptionButton = $("#remove-option");
const files = new Map();
let ready = false;
let activePreset = "account";
let loadButtonState = "load";
let runButtonState = "run";
let supportMessage = { key: "status.checking", params: {} };
let resultsPhase = "waiting";
let lastDirectData = null;
let lastCompleteData = null;
const LANGUAGE_STORAGE_KEY = "fastjev-language";
const isMobileDevice = navigator.userAgentData?.mobile === true
  || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
  || window.matchMedia("(max-width: 600px)").matches;

function preferredLocale() {
  try {
    const saved = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (SUPPORTED_LOCALES.includes(saved)) return saved;
  } catch (_) {
    // Some embedded browsers disable local storage.
  }
  return normalizeLocale(navigator.languages?.[0] || navigator.language);
}

let locale = preferredLocale();
const t = (key, params = {}) => translate(locale, key, params);

const supportState = reactive({ text: t("status.checking"), kind: "", icon: "memory" });
createApp({ setup: () => supportState }).mount("#support");

function applyUiMode(plain) {
  document.body.classList.toggle("plain-ui", plain);
  uiMode.checked = plain;
  try {
    localStorage.setItem("semif-ui-mode", plain ? "plain" : "original");
  } catch (_) {
    // The preference is optional; inference does not depend on browser storage.
  }
}

let savedUiMode = false;
try {
  savedUiMode = localStorage.getItem("semif-ui-mode") === "plain";
} catch (_) {
  // Some embedded browsers disable local storage.
}
applyUiMode(savedUiMode);
uiMode.addEventListener("change", () => applyUiMode(uiMode.checked));

function seconds(ms) {
  return `${(ms / 1000).toFixed(3)} s`;
}

function setSupport(key, kind = "", params = {}) {
  supportMessage = { key, params };
  supportState.text = t(key, params);
  supportState.kind = kind;
  supportState.icon = kind === "error" ? "error" : kind === "ok" ? "check_circle" : "memory";
}

function setButton(button, icon, label, spinning = false) {
  const iconElement = document.createElement("span");
  iconElement.className = `material-symbols-rounded${spinning ? " spin" : ""}`;
  iconElement.setAttribute("aria-hidden", "true");
  iconElement.textContent = icon;
  button.replaceChildren(iconElement, document.createTextNode(` ${label}`));
}

function renderButtons() {
  const selected = models[modelSelect.value];
  const loadStates = {
    load: ["download", t("model.load", { name: selected.name }), false],
    loading: ["progress_activity", t("model.loading"), true],
    ready: ["check", t("model.ready"), false],
    retry: ["refresh", t("model.retry"), false],
  };
  const runStates = {
    run: ["play_arrow", t("decision.run"), false],
    running: ["progress_activity", t("decision.running"), true],
    again: ["replay", t("decision.runAgain"), false],
  };
  setButton(loadButton, ...loadStates[loadButtonState]);
  setButton(runButton, ...runStates[runButtonState]);
}

function renderSelectedModel() {
  const selected = models[modelSelect.value];
  $("#selected-model").textContent = selected.short;
  $("#model-size").textContent = t("hero.modelSize", { size: selected.size });
  $("#model-link").href = selected.url;
  if (loadButtonState === "load") $("#download-detail").textContent = t("progress.firstLoad", { size: selected.size });
  const notice = $("#model-notice");
  notice.textContent = t(selected.noticeKey);
  notice.className = `model-notice ${selected.noticeClass}`.trim();
  document.querySelectorAll("[data-quality-model]").forEach((row) => {
    row.classList.toggle("selected", row.dataset.qualityModel === modelSelect.value);
  });
  renderButtons();
}

function renderProgress(event) {
  if (!event.file) return;
  if (event.status === "progress" && Number.isFinite(event.loaded) && Number.isFinite(event.total)) {
    files.set(event.file, { loaded: event.loaded, total: event.total });
  } else if (event.status === "done" && files.has(event.file)) {
    const item = files.get(event.file);
    files.set(event.file, { loaded: item.total, total: item.total });
  }
  const totals = [...files.values()].reduce((sum, item) => ({ loaded: sum.loaded + item.loaded, total: sum.total + item.total }), { loaded: 0, total: 0 });
  if (totals.total > 0) {
    const percent = Math.min(100, (totals.loaded / totals.total) * 100);
    $("#download-meter").style.width = `${percent}%`;
    $("#download-value").textContent = `${percent.toFixed(0)}%`;
    $("#download-detail").textContent = t("progress.files");
  } else if (event.status === "initiate") {
    $("#download-value").textContent = t("progress.cacheCheck");
    $("#download-detail").textContent = event.file;
  }
}

function renderDirect(data) {
  lastDirectData = data;
  const output = $("#direct-output");
  output.classList.remove("empty");
  output.replaceChildren(...data.options.map((item) => {
    const row = document.createElement("div");
    row.className = "choice";
    const label = document.createElement("span");
    label.className = "choice-label";
    const letter = document.createElement("b");
    letter.textContent = item.label;
    const description = document.createElement("small");
    description.textContent = item.description;
    label.append(letter, description);
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(1, item.probability * 100)}%`;
    bar.append(fill);
    const score = document.createElement("em");
    score.textContent = item.probability.toFixed(3);
    row.append(label, bar, score);
    return row;
  }));
  $("#direct-total").textContent = seconds(data.totalMs);
  $("#direct-input").textContent = t("metrics.tokens", { count: data.inputTokens });
  $("#direct-readouts").textContent = t(data.readouts === 1 ? "metrics.readoutOne" : "metrics.readoutMany", { count: data.readouts });
}

function renderComplete(data) {
  lastCompleteData = data;
  resultsPhase = "complete";
  $("#generated-output").textContent = data.generatedText || t("results.noText");
  $("#generation-ttft").textContent = data.ttftMs == null ? t("metrics.noToken") : seconds(data.ttftMs);
  $("#generation-total").textContent = seconds(data.generationMs);
  $("#generation-input").textContent = t("metrics.tokens", { count: data.inputTokens });
  $("#generation-tokens").textContent = t("metrics.tokens", { count: data.generatedTokens });
  $("#ratio").textContent = t("verdict.ratio", { ratio: (data.generationMs / data.directMs).toFixed(2) });
  $("#run-note").textContent = t("verdict.measured", { direct: seconds(data.directMs), generation: seconds(data.generationMs) });
}

function resetResults() {
  resultsPhase = "direct";
  lastDirectData = null;
  lastCompleteData = null;
  $("#direct-output").textContent = t("results.directRunning");
  $("#direct-output").className = "output empty";
  $("#generated-output").textContent = t("results.waitingDirect");
  $("#generated-output").className = "output empty";
  for (const id of ["#direct-total", "#direct-input", "#generation-ttft", "#generation-total", "#generation-input", "#generation-tokens"]) $(id).textContent = "—";
  $("#direct-readouts").textContent = "—";
  $("#ratio").textContent = t("verdict.measuring");
}

worker.addEventListener("message", ({ data }) => {
  switch (data.type) {
    case "progress":
      renderProgress(data.event);
      break;
    case "loading":
      setSupport(data.messageKey || "status.workerError", "", { message: data.message || "unknown error" });
      break;
    case "loaded":
      $("#load-value").textContent = seconds(data.loadMs);
      $("#download-meter").style.width = "100%";
      if (!files.size) {
        $("#download-value").textContent = t("progress.cached");
        $("#download-detail").textContent = t("progress.noTransfer");
      }
      break;
    case "ready":
      ready = true;
      $("#warmup-value").textContent = seconds(data.warmupMs);
      setSupport("status.ready", "ok", { modelName: data.modelName });
      loadButton.disabled = true;
      modelSelect.disabled = true;
      loadButtonState = "ready";
      runButtonState = "run";
      renderButtons();
      runButton.disabled = false;
      break;
    case "direct":
      renderDirect(data);
      resultsPhase = "waiting-generation";
      $("#generated-output").textContent = t("results.reading");
      break;
    case "generation-start":
      resultsPhase = "generation";
      $("#generated-output").textContent = "";
      $("#generated-output").classList.remove("empty");
      break;
    case "generation-update":
      $("#generated-output").textContent = data.text;
      if (data.ttftMs != null) $("#generation-ttft").textContent = seconds(data.ttftMs);
      $("#generation-tokens").textContent = t("metrics.tokens", { count: data.tokens });
      break;
    case "complete": {
      renderComplete(data);
      setSupport("status.complete", "ok");
      runButton.disabled = false;
      runButtonState = "again";
      renderButtons();
      break;
    }
    case "error":
      setSupport(data.messageKey || "status.workerError", "error", data.params || { message: data.message || "unknown error" });
      runButton.disabled = !ready;
      loadButton.disabled = ready;
      modelSelect.disabled = ready;
      loadButtonState = ready ? "ready" : "retry";
      runButtonState = ready ? "again" : "run";
      renderButtons();
      break;
  }
});

worker.addEventListener("error", (event) => {
  setSupport("status.workerFailed", "error", { message: event.message });
  loadButton.disabled = false;
  loadButtonState = "retry";
  renderButtons();
});

function optionRows() {
  return [...optionList.querySelectorAll(".option-row")];
}

function syncOptionControls() {
  const rows = optionRows();
  rows.forEach((row, index) => { row.querySelector("b").textContent = String.fromCharCode(65 + index); });
  $("#option-count").textContent = `${rows.length} / ${MAX_OPTIONS}`;
  removeOptionButton.disabled = rows.length <= MIN_OPTIONS;
  addOptionButton.disabled = rows.length >= MAX_OPTIONS;
}

function appendOption(value = "") {
  if (optionRows().length >= MAX_OPTIONS) return;
  const row = document.createElement("label");
  row.className = "option-row";
  const label = document.createElement("b");
  const input = document.createElement("input");
  input.className = "option";
  input.value = value;
  input.placeholder = t("form.placeholder");
  row.append(label, input);
  optionList.append(row);
  syncOptionControls();
  return input;
}

function setOptions(values) {
  while (optionRows().length > values.length) optionRows().at(-1).remove();
  while (optionRows().length < values.length) appendOption();
  optionRows().forEach((row, index) => { row.querySelector(".option").value = values[index]; });
  syncOptionControls();
}

function localizedPreset(name) {
  return {
    state: t(`preset.${name}.state`),
    question: t(`preset.${name}.question`),
    options: [1, 2, 3].map((index) => t(`preset.${name}.option${index}`)),
  };
}

function setPreset(name, focus = true) {
  if (!["account", "email"].includes(name)) return;
  activePreset = name;
  const preset = localizedPreset(name);
  $("#state").value = preset.state;
  $("#question").value = preset.question;
  setOptions(preset.options);
  if (focus) $("#state").focus();
}

function renderLocalizedResults() {
  if (lastDirectData) renderDirect(lastDirectData);
  if (lastCompleteData) {
    renderComplete(lastCompleteData);
    return;
  }
  if (resultsPhase === "waiting") {
    $("#direct-output").textContent = t("results.waiting");
    $("#generated-output").textContent = t("results.waiting");
    $("#direct-readouts").textContent = t("metrics.readoutOne", { count: 1 });
    $("#ratio").textContent = t("verdict.prompt");
  } else if (resultsPhase === "direct") {
    $("#direct-output").textContent = t("results.directRunning");
    $("#generated-output").textContent = t("results.waitingDirect");
    $("#ratio").textContent = t("verdict.measuring");
  } else if (resultsPhase === "waiting-generation") {
    $("#generated-output").textContent = t("results.reading");
  }
  $("#run-note").textContent = t("verdict.note");
}

function applyLanguage() {
  document.documentElement.lang = locale;
  document.title = t("meta.title");
  $("meta[name='description']").content = t("meta.description");
  languageSelect.value = locale;
  languageSelect.setAttribute("aria-label", t("language.label"));
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
  });
  document.querySelectorAll(".option").forEach((input) => { input.placeholder = t("form.placeholder"); });
  supportState.text = t(supportMessage.key, supportMessage.params);
  $("#device-note").textContent = t(isMobileDevice ? "device.mobile" : "device.desktop");
  if (activePreset) setPreset(activePreset, false);
  renderSelectedModel();
  renderLocalizedResults();
}

addOptionButton.addEventListener("click", () => {
  activePreset = null;
  appendOption()?.focus();
});
removeOptionButton.addEventListener("click", () => {
  activePreset = null;
  const rows = optionRows();
  if (rows.length > MIN_OPTIONS) rows.at(-1).remove();
  syncOptionControls();
});
syncOptionControls();

async function checkWebGPU() {
  if (!navigator.gpu) {
    setSupport("status.webgpuUnavailable", "error");
    loadButton.disabled = true;
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    setSupport("status.noAdapter", "error");
    loadButton.disabled = true;
    return;
  }
  setSupport("status.webgpuReady", "ok");
}

loadButton.addEventListener("click", () => {
  loadButton.disabled = true;
  modelSelect.disabled = true;
  loadButtonState = "loading";
  renderButtons();
  worker.postMessage({
    type: "load",
    modelId: modelSelect.value,
    useLocal:
      ["127.0.0.1", "localhost"].includes(location.hostname) &&
      new URLSearchParams(location.search).has("local"),
  });
});

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => setPreset(button.dataset.preset));
});

$(".inputs").addEventListener("input", () => { activePreset = null; });

languageSelect.addEventListener("change", () => {
  locale = normalizeLocale(languageSelect.value);
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, locale);
  } catch (_) {
    // Language selection still works for this tab when storage is unavailable.
  }
  applyLanguage();
});

modelSelect.addEventListener("change", renderSelectedModel);
modelSelect.value = "minicpm5-2b";
setPreset("account", false);
applyLanguage();

runButton.addEventListener("click", () => {
  const state = $("#state").value.trim();
  const question = $("#question").value.trim();
  const options = [...document.querySelectorAll(".option")].map((input) => input.value.trim());
  if (!state || !question || options.some((option) => !option)) {
    setSupport("form.required", "error");
    return;
  }
  resetResults();
  runButton.disabled = true;
  runButtonState = "running";
  renderButtons();
  setSupport("status.comparing");
  worker.postMessage({ type: "compare", data: { state, question, options } });
});

checkWebGPU().catch((error) => setSupport("status.webgpuCheckFailed", "error", { message: error.message }));
