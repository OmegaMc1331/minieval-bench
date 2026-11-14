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

function fmtMs(value) {
  return value === null || value === undefined ? "n/a" : value.toFixed(1);
}

function fmtCost(value) {
  return value === null || value === undefined ? "n/a" : value.toFixed(6);
}

const payload = JSON.parse(document.getElementById("report-data").textContent);
const stats = payload.stats || [];
const manifest = payload.manifest || {};

const trials = (payload.trials || []).map((row, index) => ({
  model: payload.models[row[0]],
  entryId: payload.entryIds[row[1]],
  repeat: row[2],
  ok: row[3],
  passed: row[4],
  latencyMs: row[5],
  queueMs: row[6],
  inputTokens: row[7],
  outputTokens: row[8],
  costUsd: row[9],
  errorType: row[10] === null ? null : payload.errorTypes[row[10]],
  response: payload.responses[index],
  errorMessage: payload.errorMessages[index],
}));

const datasetById = payload.dataset
  ? Object.fromEntries(payload.dataset.map((entry) => [entry.id, entry]))
  : null;

const CHART_COLORS = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"];

function colorForIndex(index) {
  return CHART_COLORS[index % CHART_COLORS.length];
}

function shortSha(sha) {
  return typeof sha === "string" && sha.length > 12 ? sha.slice(0, 12) + "…" : sha;
}

function renderOverview(section) {
  const successful = trials.filter((trial) => trial.ok).length;
  const planned = manifest.dataset_entries * manifest.repeats * (manifest.models || []).length;
  section.append(
    h("dl", { class: "facts" },
      h("dt", { text: "Benchmark" }), h("dd", { text: manifest.benchmark }),
      h("dt", { text: "Executed (UTC)" }), h("dd", { text: manifest.created_at_utc }),
      h("dt", { text: "MiniEval version" }), h("dd", { text: manifest.minieval_version }),
      h("dt", { text: "Models" },), h("dd", { text: String((manifest.models || []).length) }),
      h("dt", { text: "Trials recorded" }), h("dd", { text: fmtInt(trials.length) }),
      h("dt", { text: "Requests planned" }), h("dd", { text: planned ? fmtInt(planned) : "n/a" }),
      h("dt", { text: "Requests succeeded" }), h("dd", { text: fmtInt(successful) }),
      h("dt", { text: "Duration" }),
      h("dd", {
        text: manifest.duration_seconds === null || manifest.duration_seconds === undefined
          ? "n/a"
          : manifest.duration_seconds.toFixed(1) + " s",
      }),
      h("dt", { text: "Dataset" }), h("dd", { text: manifest.dataset_path }),
      h("dt", { text: "Dataset sha256" }),
      h("dd", { title: manifest.dataset_sha256, text: shortSha(manifest.dataset_sha256) }),
    ),
  );
}

const PERFORMANCE_METRICS = [
  ["answer_accuracy", "Answer accuracy"],
  ["overall_pass_rate", "Overall pass rate"],
  ["success_rate", "Request success rate"],
];

function renderPerformance(section) {
  let selected = PERFORMANCE_METRICS[0][0];

  const chartArea = h("div");
  const table = renderPerformanceTable();

  function draw() {
    chartArea.replaceChildren(rateChart(selected));
  }

  const radios = h(
    "fieldset",
    null,
    h("legend", { text: "Indicator" }),
    h(
      "div",
      { class: "radio-row" },
      PERFORMANCE_METRICS.map(([key, label]) =>
        h("label", null,
          h("input", {
            type: "radio",
            name: "performance-metric",
            value: key,
            checked: key === selected ? "" : null,
            onchange: (event) => {
              selected = event.target.value;
              draw();
            },
          }),
          label,
        ),
      ),
    ),
  );

  section.append(radios, chartArea, table);
  draw();
}

function rateChart(metricKey) {
  const rows = [...stats]
    .sort((a, b) => (b[metricKey] ?? -1) - (a[metricKey] ?? -1))
    .map((item) =>
      h("div", { class: "chart-row" },
        h("span", { class: "row-label", text: item.model }),
        h("div", { class: "track", role: "img", "aria-label": ariaForRate(item, metricKey) },
          item[metricKey] === null || item[metricKey] === undefined
            ? null
            : h("div", {
                class: "fill",
                style: `width:${Math.max(0.5, item[metricKey] * 100)}%;background:var(--c1)`,
              }),
        ),
        h("span", { class: "row-value", text: fmtRate(item[metricKey]) }),
      ),
    );
  return h("div", { class: "chart-rows" }, rows);
}

function ariaForRate(item, metricKey) {
  const label = PERFORMANCE_METRICS.find(([key]) => key === metricKey)[1];
  return `${item.model}: ${label} ${fmtRate(item[metricKey])}`;
}

function renderPerformanceTable() {
  return h(
    "details",
    { class: "data-details" },
    h("summary", { text: "View exact values" }),
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Model", "Requests", "OK", "Failed", "Correct", "Incorrect", "Success",
              "Accuracy", "Coverage", "Overall"].map((label, index) =>
              h("th", { class: index > 0 ? "num" : null, text: label }),
            ),
          ),
        ),
        h("tbody", null,
          stats.map((item) =>
            h("tr", null,
              h("td", { text: item.model }),
              h("td", { class: "num", text: fmtInt(item.requests) }),
              h("td", { class: "num", text: fmtInt(item.succeeded) }),
              h("td", { class: "num", text: fmtInt(item.failed) }),
              h("td", { class: "num", text: fmtInt(item.passed) }),
              h("td", { class: "num", text: item.incorrect === null ? "n/a" : fmtInt(item.incorrect) }),
              h("td", { class: "num", text: fmtRate(item.success_rate) }),
              h("td", { class: "num", text: fmtRate(item.answer_accuracy) }),
              h("td", { class: "num", text: fmtRate(item.evaluation_coverage) }),
              h("td", { class: "num", text: fmtRate(item.overall_pass_rate) }),
            ),
          ),
        ),
      ),
    ),
  );
}

const LATENCY_BINS = 20;

function renderLatency(section) {
  const selected = new Map(stats.map((item) => [item.model, true]));

  const charts = h("div");
  const checkboxArea = h("fieldset");

  function draw() {
    const active = stats.filter((item) => selected.get(item.model));
    charts.replaceChildren(
      ...active.map((item) => latencyBlock(item)),
      active.length === 0
        ? h("p", { class: "empty-note", text: "Select at least one model." })
        : null,
    );
  }

  checkboxArea.append(
    h("legend", { text: "Models" }),
    h(
      "div",
      { class: "radio-row" },
      stats.map((item) =>
        h("label", null,
          h("input", {
            type: "checkbox",
            checked: "",
            onchange: (event) => {
              selected.set(item.model, event.target.checked);
              draw();
            },
          }),
          item.model,
        ),
      ),
    ),
  );

  section.append(
    checkboxArea,
    h("p", {
      class: "section-note",
      text: `Histograms use ${LATENCY_BINS} equal-width bins over the successful requests of each model. Latency measures the provider call only; queue delay is reported separately.`,
    }),
    charts,
  );
  draw();
}

function latencyBlock(item) {
  const values = trials
    .filter((trial) => trial.model === item.model && trial.ok)
    .map((trial) => trial.latencyMs)
    .filter((value) => value !== null);
  const block = h("div", { class: "multi-chart" },
    h("h4", { text: item.model }),
    h("p", { class: "sub", text: `${values.length} successful requests` }),
  );
  if (values.length === 0) {
    block.append(h("p", { class: "empty-note", text: "No successful requests: no latency data." }));
    return block;
  }
  const histogram = buildHistogram(values, LATENCY_BINS);
  const maxCount = Math.max(...histogram.bins);
  const bars = histogram.bins.map((count, index) => {
    const start = histogram.min + (index * histogram.width);
    const end = start + histogram.width;
    return h("div", {
      class: "bin",
      tabindex: "0",
      style: `height:${Math.max(1, (count / maxCount) * 100)}%`,
      title: `[${start.toFixed(1)}, ${end.toFixed(1)}) ms: ${count} requests`,
      "aria-label": `Bin ${index + 1}: ${count} requests between ${start.toFixed(1)} and ${end.toFixed(1)} milliseconds`,
    });
  });
  block.append(
    h("div", { class: "histogram", role: "img", "aria-label": ariaForHistogram(item.model, histogram) }, bars),
    h("div", { class: "histogram-axis" },
      h("span", { text: histogram.min.toFixed(1) + " ms" }),
      h("span", { text: "median " + fmtMs(item.median_latency_ms) + " ms" }),
      h("span", { text: "p95 " + fmtMs(item.p95_latency_ms) + " ms" }),
      h("span", { text: histogram.max.toFixed(1) + " ms" }),
    ),
    h("details", { class: "data-details" },
      h("summary", { text: "View bin counts" }),
      h("div", { class: "table-scroll" },
        h("table", null,
          h("thead", null,
            h("tr", null,
              h("th", { text: "Range (ms)" }),
              h("th", { class: "num", text: "Requests" }),
            ),
          ),
          h("tbody", null,
            histogram.bins.map((count, index) => {
              const start = histogram.min + index * histogram.width;
              const end = start + histogram.width;
              return h("tr", null,
                h("td", { class: "mono", text: `[${start.toFixed(1)}, ${end.toFixed(1)})` }),
                h("td", { class: "num", text: String(count) }),
              );
            }),
          ),
        ),
      ),
    ),
  );
  return block;
}

function ariaForHistogram(model, histogram) {
  const total = histogram.bins.reduce((sum, count) => sum + count, 0);
  return `Latency distribution for ${model}: ${total} requests between ` +
    `${histogram.min.toFixed(1)} and ${histogram.max.toFixed(1)} milliseconds in ${LATENCY_BINS} bins.`;
}

function buildHistogram(values, binCount) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) / binCount || 1;
  const bins = Array.from({ length: binCount }, () => 0);
  for (const value of values) {
    let index = Math.floor((value - min) / width);
    if (index >= binCount) index = binCount - 1;
    bins[index] += 1;
  }
  return { bins, min, max, width };
}

function renderTokensCost(section) {
  section.append(
    h("h3", { text: "Token consumption" }),
    tokenChart(),
    h("p", {
      class: "section-note",
      text: "Bars stack input and output tokens over the successful requests that reported usage. Partial coverage is marked explicitly.",
    }),
    h("h3", { text: "Estimated cost" }),
    costChart(),
    h("p", {
      class: "section-note",
      text: "Hatched bars mark partially known costs; n/a means no pricing data was available. Models are not ranked by cost when data is missing.",
    }),
  );
}

function coverageNote(reported, succeeded) {
  if (reported === null || reported === undefined) return "coverage unknown";
  if (reported === succeeded) return null;
  return `${reported}/${succeeded} requests`;
}

function tokenChart() {
  const withData = stats.filter(
    (item) => item.input_tokens !== null || item.output_tokens !== null,
  );
  if (withData.length === 0) {
    return h("p", { class: "empty-note", text: "No token usage was reported in this run." });
  }
  const totals = withData.map(
    (item) => (item.input_tokens || 0) + (item.output_tokens || 0),
  );
  const maxTotal = Math.max(...totals);
  return h("div", { class: "chart-rows" },
    h("div", { class: "legend" },
      h("span", { class: "legend-item" },
        h("span", { class: "legend-swatch", style: "background:var(--c1)" }), "Input"),
      h("span", { class: "legend-item" },
        h("span", { class: "legend-swatch", style: "background:var(--c2)" }), "Output"),
    ),
    withData.map((item) => {
      const input = item.input_tokens || 0;
      const output = item.output_tokens || 0;
      const note = coverageNote(item.input_tokens_reported, item.succeeded);
      return h("div", { class: "chart-row" },
        h("span", { class: "row-label", text: item.model }),
        h("div", { class: "track", role: "img", "aria-label": ariaForTokens(item) },
          h("div", { class: "segment", style: `width:${(input / maxTotal) * 100}%;background:var(--c1)`, title: `Input: ${fmtInt(item.input_tokens)}` }),
          h("div", { class: "segment", style: `width:${(output / maxTotal) * 100}%;background:var(--c2)`, title: `Output: ${fmtInt(item.output_tokens)}` }),
        ),
        h("span", { class: "row-value" },
          `${fmtInt(item.input_tokens)} in / ${fmtInt(item.output_tokens)} out`,
          note ? h("span", { class: "bar-note", text: ` (${note})` }) : null,
        ),
      );
    }),
  );
}

function ariaForTokens(item) {
  return `${item.model}: ${fmtInt(item.input_tokens)} input tokens, ${fmtInt(item.output_tokens)} output tokens` +
    (coverageNote(item.input_tokens_reported, item.succeeded)
      ? `, coverage ${coverageNote(item.input_tokens_reported, item.succeeded)}`
      : ", full coverage");
}

function costChart() {
  const ranked = [...stats].sort((a, b) => costRank(b) - costRank(a));
  return h("div", { class: "chart-rows" },
    h("div", { class: "legend" },
      h("span", { class: "legend-item" },
        h("span", { class: "legend-swatch", style: "background:var(--c1)" }), "Known cost"),
      h("span", { class: "legend-item" },
        h("span", { class: "legend-swatch", style: "background:repeating-linear-gradient(45deg,var(--c1) 0 4px,transparent 4px 8px)" }),
        "Partially known"),
    ),
    ranked.map((item) => {
      const full = item.cost_reported === item.succeeded;
      const note = coverageNote(item.cost_reported, item.succeeded);
      const maxCost = Math.max(...stats.map((other) => other.cost_usd || 0));
      const known = item.cost_usd !== null && item.cost_usd !== undefined;
      return h("div", { class: "chart-row" },
        h("span", { class: "row-label", text: item.model }),
        h("div", { class: "track", role: "img", "aria-label": ariaForCost(item) },
          known && maxCost > 0
            ? h("div", {
                class: "fill" + (full ? "" : " partial"),
                style: `width:${Math.max(0.5, ((item.cost_usd || 0) / maxCost) * 100)}%;background:var(--c1)`,
              })
            : null,
        ),
        h("span", { class: "row-value" },
          fmtCost(item.cost_usd) + (item.cost_usd === 0 && full ? " (measured zero)" : ""),
          known && !full && note ? h("span", { class: "bar-note", text: ` (${note}, partial)` }) : null,
        ),
      );
    }),
  );
}

function costRank(item) {
  if (item.cost_usd === null || item.cost_usd === undefined) return -1;
  const full = item.cost_reported === item.succeeded;
  return (full ? 1 : 0) * 1e12 + item.cost_usd;
}

function ariaForCost(item) {
  if (item.cost_usd === null || item.cost_usd === undefined) return `${item.model}: cost unknown`;
  const note = coverageNote(item.cost_reported, item.succeeded);
  return `${item.model}: estimated cost ${fmtCost(item.cost_usd)} USD` + (note ? `, ${note}` : ", full coverage");
}

const OUTCOME_FILTERS = [
  ["all", "All outcomes"],
  ["correct", "Correct answers"],
  ["incorrect", "Incorrect answers"],
  ["failed", "Failed requests"],
  ["not-evaluable", "Not evaluable"],
];

const TRIAL_COLUMNS = [
  ["model", "Model", true],
  ["entryId", "Entry", true],
  ["repeat", "Rep", true],
  ["status", "Status", false],
  ["evaluation", "Eval", false],
  ["latencyMs", "Latency (ms)", true],
  ["queueMs", "Queue (ms)", true],
  ["inputTokens", "In tok", true],
  ["outputTokens", "Out tok", true],
  ["costUsd", "Cost (USD)", true],
  ["errorType", "Error", false],
];

const explorer = {
  search: "",
  model: "all",
  outcome: "all",
  errorType: "all",
  sortKey: null,
  sortAsc: true,
  page: 1,
  pageSize: 25,
};

function trialOutcome(trial) {
  if (!trial.ok) return "failed";
  if (trial.passed === true) return "correct";
  if (trial.passed === false) return "incorrect";
  return "not-evaluable";
}

function trialMatches(trial) {
  if (explorer.model !== "all" && trial.model !== explorer.model) return false;
  if (explorer.outcome !== "all" && trialOutcome(trial) !== explorer.outcome) return false;
  if (explorer.errorType !== "all" && trial.errorType !== explorer.errorType) return false;
  if (explorer.search) {
    const needle = explorer.search.toLowerCase();
    const haystack = `${trial.entryId} ${trial.response || ""} ${trial.errorMessage || ""}`.toLowerCase();
    if (!haystack.includes(needle)) return false;
  }
  return true;
}

function trialSortValue(trial) {
  const key = explorer.sortKey;
  if (!key) return null;
  if (key === "model") return trial.model;
  if (key === "entryId") return trial.entryId;
  return trial[key];
}

function compareTrials(a, b) {
  const aValue = trialSortValue(a);
  const bValue = trialSortValue(b);
  if (aValue === null && bValue === null) return 0;
  if (aValue === null) return 1;
  if (bValue === null) return -1;
  const order = typeof aValue === "string"
    ? aValue.localeCompare(bValue)
    : aValue - bValue;
  return explorer.sortAsc ? order : -order;
}

function renderTrials(section) {
  if (trials.length === 0) {
    section.append(h("p", { class: "empty-note", text: "No trial data is present in this export." }));
    return;
  }

  const toolbar = h("div", { class: "toolbar" });
  const tableWrap = h("div", { class: "table-scroll" });
  const pagination = h("div", { class: "pagination" });
  const detailArea = h("div");

  const searchInput = h("input", {
    type: "text",
    placeholder: "Search entry, response or error…",
    "aria-label": "Search trials",
    oninput: (event) => {
      explorer.search = event.target.value;
      explorer.page = 1;
      draw();
    },
  });
  const modelSelect = selectControl("Filter by model", [["all", "All models"],
    ...payload.models.map((name) => [name, name])], (value) => {
    explorer.model = value;
    explorer.page = 1;
    draw();
  });
  const outcomeSelect = selectControl("Filter by outcome", OUTCOME_FILTERS, (value) => {
    explorer.outcome = value;
    explorer.page = 1;
    draw();
  });
  const errorSelect = selectControl("Filter by error type", [["all", "All error types"],
    ...payload.errorTypes.map((name) => [name, name])], (value) => {
    explorer.errorType = value;
    explorer.page = 1;
    draw();
  });
  const pageSizeSelect = selectControl("Rows per page", [["25", "25 rows"], ["50", "50 rows"], ["100", "100 rows"]], (value) => {
    explorer.pageSize = Number(value);
    explorer.page = 1;
    draw();
  });

  toolbar.append(
    searchInput,
    modelSelect,
    outcomeSelect,
    errorSelect,
    pageSizeSelect,
  );
  section.append(toolbar, detailArea, tableWrap, pagination);

  function draw() {
    const filtered = trials.filter(trialMatches).sort(compareTrials);
    const pageCount = Math.max(1, Math.ceil(filtered.length / explorer.pageSize));
    explorer.page = Math.min(explorer.page, pageCount);
    const start = (explorer.page - 1) * explorer.pageSize;
    const pageRows = filtered.slice(start, start + explorer.pageSize);

    renderTrialTable(pageRows);
    pagination.replaceChildren(
      h("button", {
        type: "button",
        text: "Previous",
        disabled: explorer.page <= 1 ? "" : null,
        onclick: () => {
          explorer.page -= 1;
          draw();
        },
      }),
      h("span", { text: `Page ${explorer.page} of ${pageCount}` }),
      h("button", {
        type: "button",
        text: "Next",
        disabled: explorer.page >= pageCount ? "" : null,
        onclick: () => {
          explorer.page += 1;
          draw();
        },
      }),
      h("span", { class: "result-count", text: `${fmtInt(filtered.length)} of ${fmtInt(trials.length)} trials` }),
    );
  }

  function renderTrialTable(pageRows) {
    tableWrap.replaceChildren(
      h("table", { "aria-label": "Trial results" },
        h("thead", null, h("tr", null, TRIAL_COLUMNS.map(([key, label, sortable]) =>
          h("th", { class: key === "model" || key === "entryId" ? null : "num" },
            sortable
              ? h("button", {
                  type: "button",
                  onclick: () => {
                    if (explorer.sortKey === key) {
                      explorer.sortAsc = !explorer.sortAsc;
                    } else {
                      explorer.sortKey = key;
                      explorer.sortAsc = true;
                    }
                    draw();
                  },
                  text: sortLabel(key, label),
                })
              : label,
          ),
        ))),
        h("tbody", null, pageRows.map((trial) => renderTrialRow(trial))),
      ),
    );
  }

  function sortLabel(key, label) {
    const arrow = explorer.sortKey === key ? (explorer.sortAsc ? " ↑" : " ↓") : "";
    return label + arrow;
  }

  function renderTrialRow(trial) {
    const row = h("tr", {
      tabindex: "0",
      role: "button",
      "aria-label": `Trial ${trial.entryId}, model ${trial.model}, repeat ${trial.repeat}, ${trialOutcome(trial)}`,
      onclick: () => showTrialDetail(trial),
      onkeydown: (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          showTrialDetail(trial);
        }
      },
    },
      h("td", { class: "mono", text: trial.model }),
      h("td", { class: "mono", text: trial.entryId }),
      h("td", { class: "num", text: String(trial.repeat) }),
      h("td", null, h("span", { class: trial.ok ? "status-ok" : "status-failed", text: trial.ok ? "ok" : "failed" })),
      h("td", null, evaluationCell(trial)),
      h("td", { class: "num", text: fmtMs(trial.latencyMs) }),
      h("td", { class: "num", text: fmtMs(trial.queueMs) }),
      h("td", { class: "num", text: fmtInt(trial.inputTokens) }),
      h("td", { class: "num", text: fmtInt(trial.outputTokens) }),
      h("td", { class: "num", text: fmtCost(trial.costUsd) }),
      h("td", { class: "mono", text: trial.errorType || "" }),
    );
    return row;
  }

  function evaluationCell(trial) {
    if (!trial.ok) return h("span", { class: "eval-na", text: "—" });
    if (trial.passed === true) return h("span", { class: "eval-pass", text: "pass" });
    if (trial.passed === false) return h("span", { class: "eval-fail", text: "fail" });
    return h("span", { class: "eval-na", text: "n/a" });
  }

  function showTrialDetail(trial) {
    const entry = datasetById ? datasetById[trial.entryId] : null;
    const detail = h("div", { class: "trial-detail" },
      h("header", null,
        h("h4", { text: `${trial.model} · ${trial.entryId} · repeat ${trial.repeat}` }),
        h("button", {
          type: "button",
          text: "Close",
          onclick: closeDetail,
        }),
      ),
      factRow("Status", trial.ok ? "ok" : "failed"),
      factRow("Evaluation", trial.ok ? String(trial.passed) : "not evaluable"),
      factRow("Latency", fmtMs(trial.latencyMs) + " ms"),
      factRow("Queue delay", fmtMs(trial.queueMs) + " ms"),
      factRow("Tokens", `in ${fmtInt(trial.inputTokens)} / out ${fmtInt(trial.outputTokens)}`),
      factRow("Estimated cost", fmtCost(trial.costUsd) + " USD"),
      trial.errorType ? factRow("Error type", trial.errorType) : null,
      h("p", { class: "field-label", text: "Response" }),
      h("pre", { text: trial.response === null ? "n/a" : trial.response }),
      trial.errorMessage ? h("p", { class: "field-label", text: "Error message" }) : null,
      trial.errorMessage ? h("pre", { text: trial.errorMessage }) : null,
      h("p", { class: "field-label", text: "Prompt" }),
      h("pre", { text: entry ? entry.prompt : "Not recorded in this export." }),
      h("p", { class: "field-label", text: "Reference" }),
      h("pre", { text: entry ? `${entry.reference} (${entry.metric})` : "Not recorded in this export." }),
      entry && entry.system ? h("p", { class: "field-label", text: "System prompt" }) : null,
      entry && entry.system ? h("pre", { text: entry.system }) : null,
    );
    detailArea.replaceChildren(detail);
    detail.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeDetail();
    });
    detail.querySelector("button").focus();
  }

  function closeDetail() {
    detailArea.replaceChildren();
  }

  draw();
}

function factRow(label, value) {
  return h("div", null, h("span", { class: "field-label", text: label + ": " }), h("span", { class: "mono", text: value }));
}

function selectControl(label, options, onChange) {
  const select = h("select", { "aria-label": label },
    options.map(([value, text]) =>
      h("option", { value, text }),
    ),
  );
  select.addEventListener("change", () => onChange(select.value));
  return select;
}

function renderErrors(section) {
  const modelsWithErrors = stats.filter((item) => Object.keys(item.errors || {}).length > 0);
  if (modelsWithErrors.length === 0) {
    section.append(h("p", { class: "empty-note", text: "No provider errors were recorded in this run." }));
    return;
  }

  const allTypes = [...new Set(modelsWithErrors.flatMap((item) => Object.keys(item.errors)))].sort();
  const typeColor = new Map(allTypes.map((name, index) => [name, colorForIndex(index)]));
  const maxErrors = Math.max(
    ...modelsWithErrors.map((item) => Object.values(item.errors).reduce((sum, count) => sum + count, 0)),
  );

  section.append(
    h("div", { class: "chart-rows" },
      h("div", { class: "legend" },
        allTypes.map((name) =>
          h("span", { class: "legend-item" },
            h("span", { class: "legend-swatch", style: `background:var(--${typeColor.get(name)})` }),
            name,
          ),
        ),
      ),
      modelsWithErrors.map((item) => {
        const total = Object.values(item.errors).reduce((sum, count) => sum + count, 0);
        return h("div", { class: "chart-row" },
          h("span", { class: "row-label", text: item.model }),
          h("div", { class: "track", role: "img", "aria-label": ariaForErrors(item) },
            allTypes.map((errorType) =>
              item.errors[errorType]
                ? h("div", {
                    class: "segment",
                    style: `width:${(item.errors[errorType] / maxErrors) * 100}%;background:var(--${typeColor.get(errorType)})`,
                    title: `${errorType}: ${item.errors[errorType]}`,
                  })
                : null,
            ),
          ),
          h("span", { class: "row-value", text: `${total} error${total === 1 ? "" : "s"}` }),
        );
      }),
    ),
    h("h3", { text: "Failed requests" }),
    failedRequestsTable(),
  );
}

function ariaForErrors(item) {
  return item.model + ": " + Object.entries(item.errors)
    .map(([errorType, count]) => `${errorType} ${count}`).join(", ");
}

function failedRequestsTable() {
  const failed = trials.filter((trial) => !trial.ok);
  return h("div", { class: "table-scroll" },
    h("table", null,
      h("thead", null,
        h("tr", null,
          h("th", { text: "Model" }),
          h("th", { text: "Entry" }),
          h("th", { class: "num", text: "Repeat" }),
          h("th", { text: "Error type" }),
          h("th", { text: "Message" }),
        ),
      ),
      h("tbody", null,
        failed.map((trial) =>
          h("tr", null,
            h("td", { class: "mono", text: trial.model }),
            h("td", { class: "mono", text: trial.entryId }),
            h("td", { class: "num", text: String(trial.repeat) }),
            h("td", { class: "mono", text: trial.errorType || "" }),
            h("td", { title: trial.errorMessage || "", text: truncate(trial.errorMessage || "", 90) }),
          ),
        ),
      ),
    ),
  );
}

function truncate(text, limit) {
  return text.length > limit ? text.slice(0, limit) + "…" : text;
}

function renderMetadata(section) {
  section.append(
    h("dl", { class: "facts" },
      h("dt", { text: "Benchmark" }), h("dd", { text: manifest.benchmark }),
      h("dt", { text: "MiniEval version" }), h("dd", { text: manifest.minieval_version }),
      h("dt", { text: "Global seed" }), h("dd", { text: String(manifest.seed) }),
      h("dt", { text: "Per-repeat seeds" }),
      h("dd", {
        text: manifest.trial_seeds ? manifest.trial_seeds.join(", ") : "n/a",
      }),
      h("dt", { text: "Repeats" }), h("dd", { text: String(manifest.repeats) }),
      h("dt", { text: "Concurrency" }), h("dd", { text: String(manifest.concurrency) }),
      h("dt", { text: "Shuffle" }),
      h("dd", { text: manifest.shuffle === null || manifest.shuffle === undefined ? "n/a" : String(manifest.shuffle) }),
      h("dt", { text: "Dataset" }), h("dd", { text: manifest.dataset_path }),
      h("dt", { text: "Dataset sha256" }), h("dd", { text: manifest.dataset_sha256 }),
    ),
    h("h3", { text: "Models" }),
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Name", "Provider", "Model string", "Temperature", "Max tokens", "Timeout (s)", "Cost / 1M in / out"].map((label, index) =>
              h("th", { class: index > 2 ? "num" : null, text: label }),
            ),
          ),
        ),
        h("tbody", null,
          (manifest.models || []).map((model) =>
            h("tr", null,
              h("td", { class: "mono", text: model.name }),
              h("td", { text: model.provider }),
              h("td", { class: "mono", text: model.model || "" }),
              h("td", { class: "num", text: model.temperature === null || model.temperature === undefined ? "n/a" : String(model.temperature) }),
              h("td", { class: "num", text: model.max_tokens === null || model.max_tokens === undefined ? "n/a" : String(model.max_tokens) }),
              h("td", { class: "num", text: model.timeout === null || model.timeout === undefined ? "n/a" : String(model.timeout) }),
              h("td", { class: "num", text: model.input_cost_per_1m !== null && model.input_cost_per_1m !== undefined
                ? `${model.input_cost_per_1m} / ${model.output_cost_per_1m}` : "n/a" }),
            ),
          ),
        ),
      ),
    ),
    h("p", {
      class: "section-note",
      text: payload.dataset
        ? "This export contains a dataset snapshot: prompts and references are shown in the trial explorer."
        : "This export does not contain a dataset snapshot: prompts and references are not shown in the trial explorer.",
    }),
  );
}

const SECTIONS = [
  ["overview", "Overview", renderOverview],
  ["performance", "Model performance", renderPerformance],
  ["latency", "Latency", renderLatency],
  ["tokens-cost", "Tokens & cost", renderTokensCost],
  ["trials", "Trial explorer", renderTrials],
  ["errors", "Errors", renderErrors],
  ["metadata", "Run metadata", renderMetadata],
];

function buildPage() {
  const app = document.getElementById("app");
  const nav = document.getElementById("section-nav");
  for (const [id, title, render] of SECTIONS) {
    nav.append(h("a", { href: "#" + id, text: title }));
    const section = h("section", { id },
      h("h2", { text: title }),
    );
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
