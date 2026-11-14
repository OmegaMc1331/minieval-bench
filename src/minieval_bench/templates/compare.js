"use strict";

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    appendChild(node, child);
  }
  return node;
}

function appendChild(node, child) {
  if (child === null || child === undefined) return;
  if (Array.isArray(child)) {
    for (const item of child) appendChild(node, item);
    return;
  }
  node.append(child.nodeType ? child : document.createTextNode(child));
}

function fmtInt(value) {
  return value === null || value === undefined ? "n/a" : value.toLocaleString("en-US");
}

function fmtRate(value) {
  return value === null || value === undefined ? "n/a" : (value * 100).toFixed(1) + "%";
}

function fmtNumber(value) {
  return value === null || value === undefined ? "n/a" : Number(value.toFixed(3)).toLocaleString("en-US");
}

function fmtIndicator(value, format) {
  if (value === null || value === undefined) return "n/a";
  if (format === "rate") return fmtRate(value);
  if (format === "ms") return value.toFixed(1);
  if (format === "cost") return value.toFixed(6);
  return fmtInt(value);
}

function fmtDelta(value, format) {
  if (value === null || value === undefined) return "n/a";
  const sign = value > 0 ? "+" : "";
  if (format === "rate") return `${sign}${(value * 100).toFixed(1)} pp`;
  if (format === "ms") return `${sign}${value.toFixed(1)}`;
  if (format === "cost") return `${sign}${value.toFixed(6)}`;
  return `${sign}${fmtInt(value)}`;
}

function fmtRelative(value) {
  if (value === null || value === undefined) return "n/a";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)}%`;
}

const payload = JSON.parse(document.getElementById("report-data").textContent);
const runs = payload.runs;
const CHART_COLORS = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"];

function shortSha(sha) {
  return typeof sha === "string" && sha.length > 12 ? sha.slice(0, 12) + "…" : sha;
}

function displayValue(value) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "boolean") return String(value);
  return String(value);
}

const RUN_FIELDS = [
  ["label", "Run", false],
  ["benchmark", "Benchmark", false],
  ["created_at_utc", "Executed (UTC)", false],
  ["minieval_version", "MiniEval version", true],
  ["seed", "Seed", true],
  ["repeats", "Repeats", true],
  ["concurrency", "Concurrency", true],
  ["shuffle", "Shuffle", true],
  ["dataset_sha256", "Dataset sha256", true],
  ["dataset_entries", "Entries", true],
  ["path", "Location", false],
];

function renderRuns(section) {
  section.append(
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null, RUN_FIELDS.map(([key, label]) => h("th", { text: label }))),
        ),
        h("tbody", null,
          runs.map((run) =>
            h("tr", null,
              RUN_FIELDS.map(([key, , warnOnDiff]) => {
                const raw = key === "dataset_sha256" ? shortSha(run[key]) : displayValue(run[key]);
                const differs = warnOnDiff && !runs.every((other) => String(other[key]) === String(run[key]));
                return h("td", {
                  class: (key === "repeats" || key === "concurrency" || key === "dataset_entries" ? "num " : "") + (differs ? "diff-cell" : ""),
                  title: key === "dataset_sha256" ? run[key] : null,
                  text: raw,
                });
              }),
            ),
          ),
        ),
      ),
    ),
    h("p", { class: "section-note", text: "Highlighted cells differ between the compared runs." }),
  );
}

function renderWarnings(section) {
  if (!payload.warnings || payload.warnings.length === 0) return;
  section.append(
    h("div", { class: "warning-box", role: "alert" },
      h("strong", { text: "Read the comparison with care" }),
      h("ul", null, payload.warnings.map((warning) => h("li", { text: warning }))),
    ),
  );
}

const CHART_INDICATORS = [
  ["answer_accuracy", "Answer accuracy"],
  ["overall_pass_rate", "Overall pass rate"],
  ["success_rate", "Request success rate"],
  ["median_latency_ms", "Median latency"],
];

function renderComparison(section) {
  if (payload.commonModels.length === 0) {
    section.append(h("p", { class: "empty-note", text: "No model is common to every run." }));
    return;
  }
  let selected = CHART_INDICATORS[0][0];
  const chartArea = h("div");

  const drawChart = () => {
    chartArea.replaceChildren(indicatorChart(selected));
  };

  const radios = h(
    "fieldset",
    null,
    h("legend", { text: "Indicator" }),
    h("div", { class: "radio-row" },
      CHART_INDICATORS.map(([key, label]) =>
        h("label", null,
          h("input", {
            type: "radio",
            name: "compare-metric",
            value: key,
            checked: key === selected ? "" : null,
            onchange: (event) => {
              selected = event.target.value;
              drawChart();
            },
          }),
          label,
        ),
      ),
    ),
    h("div", { class: "legend" },
      runs.map((run, index) =>
        h("span", { class: "legend-item" },
          h("span", { class: "legend-swatch", style: `background:var(--${CHART_COLORS[index % CHART_COLORS.length]})` }),
          run.label,
        ),
      ),
    ),
    h("p", { class: "section-note", text: "One bar per run for each model; values below the bars." }),
    chartArea,
  );

  section.append(radios, ...payload.commonModels.map(modelTables));
  drawChart();
}

function indicatorChart(metricKey) {
  const blocks = payload.commonModels.map((model) => {
    const indicator = model.indicators.find((item) => item.key === metricKey);
    const numeric = indicator.values.filter((value) => value !== null && value !== undefined);
    const max = numeric.length ? Math.max(...numeric) : 0;
    return h("div", { class: "multi-chart" },
      h("h4", { text: model.model }),
      h("div", { class: "chart-rows" },
        runs.map((run, index) => {
          const value = indicator.values[index];
          return h("div", { class: "chart-row" },
            h("span", { class: "row-label", text: run.label }),
            h("div", { class: "track" },
              value === null || value === undefined || max <= 0
                ? null
                : h("div", {
                    class: "fill",
                    style: `width:${Math.max(0.5, (value / max) * 100)}%;background:var(--${CHART_COLORS[index % CHART_COLORS.length]})`,
                  }),
            ),
            h("span", { class: "row-value", text: fmtIndicator(value, indicator.format) }),
          );
        }),
      ),
    );
  });
  return h("div", null, blocks);
}

function modelTables(model) {
  return h("div", null,
    h("h3", { text: model.model }),
    model.configConsistent
      ? null
      : h("p", {
          class: "bar-note",
          text: `Configuration differs across runs (${model.configFields.join(", ")}).`,
        }),
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            h("th", { text: "Indicator" }),
            runs.map((run) => h("th", { text: run.label })),
            h("th", { class: "num", text: "Δ (first → last)" }),
            h("th", { class: "num", text: "Δ relative" }),
          ),
        ),
        h("tbody", null,
          model.indicators.map((indicator) =>
            h("tr", null,
              h("td", { text: indicator.label }),
              indicator.values.map((value) =>
                h("td", { class: "num", text: fmtIndicator(value, indicator.format) }),
              ),
              h("td", { class: "num " + deltaClass(indicator.deltaAbs), text: fmtDelta(indicator.deltaAbs, indicator.format) }),
              h("td", { class: "num " + deltaClass(indicator.deltaRel), text: fmtRelative(indicator.deltaRel) }),
            ),
          ),
        ),
      ),
    ),
    h("p", {
      class: "section-note",
      text: "Relative change is shown only when the first value exists and is not zero; n/a marks values that are unavailable or not defined.",
    }),
  );
}

function deltaClass(value) {
  if (value === null || value === undefined) return "delta-na";
  if (value > 0) return "delta-up";
  if (value < 0) return "delta-down";
  return "delta-na";
}

function renderExclusives(section) {
  const entries = Object.entries(payload.modelsOnlyIn || {});
  if (entries.length === 0) {
    section.append(h("p", { class: "empty-note", text: "Every model is present in all runs." }));
    return;
  }
  section.append(
    h("ul", null,
      entries.map(([label, names]) =>
        h("li", null,
          h("span", { class: "mono", text: label }),
          `: not comparable everywhere — ${names.join(", ")}`,
        ),
      ),
    ),
  );
}

const SECTIONS = [
  ["runs", "Compared runs", renderRuns],
  ["warnings", "Comparability", renderWarnings],
  ["comparison", "Model comparison", renderComparison],
  ["exclusive", "Model coverage", renderExclusives],
];

function buildPage() {
  const app = document.getElementById("app");
  const nav = document.getElementById("section-nav");
  for (const [id, title, render] of SECTIONS) {
    nav.append(h("a", { href: "#" + id, text: title }));
    const section = h("section", { id }, h("h2", { text: title }));
    app.append(section);
    render(section);
  }
}

function setupThemeToggle() {
  const root = document.documentElement;
  const button = document.getElementById("theme-toggle");
  const preferDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = preferDark ? "dark" : "light";
  const syncLabel = () => {
    button.textContent = root.dataset.theme === "dark" ? "Light theme" : "Dark theme";
  };
  button.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    syncLabel();
  });
  syncLabel();
}

buildPage();
setupThemeToggle();
