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

function fmtScore(value) {
  return value === null || value === undefined ? "n/a" : Number(value.toFixed(3)).toLocaleString("en-US");
}

function fmtMs(value) {
  return value === null || value === undefined ? "n/a" : value.toFixed(1);
}

function fmtCost(value) {
  return value === null || value === undefined ? "n/a" : value.toFixed(6);
}

function fmtSeconds(value) {
  return value === null || value === undefined ? "n/a" : value.toFixed(1);
}

function shortSha(sha) {
  return typeof sha === "string" && sha.length > 12 ? sha.slice(0, 12) + "…" : sha;
}

const payload = JSON.parse(document.getElementById("report-data").textContent);

function statusClass(status) {
  if (status === "completed") return "delta-up";
  if (status === "partially_completed") return "delta-down";
  return "delta-na";
}

function renderOverview(section) {
  const overview = payload.overview;
  const suite = overview.suite || {};
  const statuses = Object.entries(overview.statuses || {})
    .map(([status, count]) => `${status} x${count}`)
    .join(", ");
  section.append(
    h("div", { class: "table-scroll" },
      h("table", null,
        h("tbody", null,
          [
            ["Suite", `${suite.id} ${suite.version || ""}`.trim()],
            ["Executed (UTC)", overview.created_at_utc || "n/a"],
            ["MiniEval version", payload.manifest.minieval_version],
            ["License", suite.license || "n/a"],
            ["Models", (overview.models || []).join(", ") || "n/a"],
            ["Tasks", `${overview.task_count} (${statuses})`],
            ["Questions", fmtInt(overview.entry_count)],
            ["Requests executed", `${fmtInt(overview.executed_calls)} / ${fmtInt(overview.planned_calls)} planned`],
            ["Duration (s)", fmtSeconds(overview.duration_seconds)],
          ].map(([field, value]) =>
            h("tr", null, h("td", { class: "dim" }, field), h("td", null, value))
          ),
        ),
      ),
    ),
  );
  const notes = [];
  if (overview.interrupted) {
    notes.push("The run was interrupted: results are partial and the suite is not fully validated.");
  }
  if (overview.protocol_modified) {
    const overrides = JSON.stringify(
      (payload.manifest.generation || {}).user_overrides || {}
    );
    notes.push(`Protocol modified from the official suite by user overrides: ${overrides}.`);
  }
  if (overview.sampled) {
    notes.push("Sampling was applied: scores describe a deterministic subset, not the full suite.");
  }
  if (overview.resume_used) {
    notes.push("Some task results were reused from a previous run with a matching protocol.");
  }
  for (const note of notes) {
    section.append(h("p", { class: "bar-note", role: "note", text: note }));
  }
}

function renderTaskBreakdown(section) {
  section.append(
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Task", "Status", "Weight", "Metrics", "Questions", "Calls", "Duration (s)", "Detail"].map(
              (label, index) => h("th", { class: [2, 4, 5, 6].includes(index) ? "num" : null, text: label })
            ),
          ),
        ),
        h("tbody", null,
          payload.tasks.map((task) =>
            h("tr", null,
              h("td", { class: "mono", text: task.id }),
              h("td", { class: statusClass(task.status), text: task.status }),
              h("td", { class: "num", text: String(task.weight) }),
              h("td", {
                text: Object.entries(task.metrics || {})
                  .map(([name, count]) => `${name} x${count}`)
                  .join(", ") || "n/a",
              }),
              h("td", { class: "num", text: fmtInt(task.entryCount) }),
              h("td", { class: "num", text: `${fmtInt(task.executedCalls)} / ${fmtInt(task.plannedCalls)}` }),
              h("td", { class: "num", text: fmtSeconds(task.durationSeconds) }),
              h("td", {
                text: [task.reused ? "reused" : null, task.error].filter(Boolean).join(" · ") || "",
              }),
            ),
          ),
        ),
      ),
    ),
    h("p", {
      class: "section-note",
      text: "Unavailable or not-executed tasks contribute no observations and never count" +
        " as zeros; partially completed tasks keep the trials that ran.",
    }),
  );
}

function rateByFormat(value, format) {
  return format === "score" ? fmtScore(value) : fmtRate(value);
}

function renderModelComparison(section) {
  for (const task of payload.tasks) {
    if (!task.stats || task.stats.length === 0) continue;
    section.append(
      h("h3", { text: task.id }),
      h("div", { class: "table-scroll" },
        h("table", null,
          h("thead", null,
            h("tr", null,
              ["Model", "Success", "Accuracy (deterministic)", "JSON conformance", "Code pass", "Judge score", "Judge judged/eligible"].map(
                (label, index) => h("th", { class: index > 0 ? "num" : null, text: label })
              ),
            ),
          ),
          h("tbody", null,
            task.stats.map((stats) =>
              h("tr", null,
                h("td", { class: "mono", text: stats.model }),
                h("td", { class: "num", text: fmtRate(stats.success_rate) }),
                h("td", { class: "num", text: fmtRate(stats.answer_accuracy) }),
                h("td", { class: "num", text: stats.json_schema_trials ? fmtRate(stats.json_schema_conformance) : "n/a" }),
                h("td", { class: "num", text: stats.code_eval_trials ? fmtRate(stats.code_eval_pass_rate) : "n/a" }),
                h("td", { class: "num", text: stats.judge_trials ? fmtScore(stats.judge_score_mean) : "n/a" }),
                h("td", {
                  class: "num",
                  text: `${stats.judge_evaluations ?? 0} / ${stats.judge_trials ?? 0}`,
                }),
              ),
            ),
          ),
        ),
      ),
    );
  }
  section.append(
    h("h3", { text: "Family aggregates" }),
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Family", "Model", "Macro", "Micro", "Trials", "Tasks covered"].map((label, index) =>
              h("th", { class: index > 1 ? "num" : null, text: label })
            ),
          ),
        ),
        h("tbody", null,
          (payload.familyAggregates || []).flatMap((aggregate) =>
            aggregate.perModel.map((item) =>
              h("tr", null,
                h("td", { text: aggregate.label }),
                h("td", { class: "mono", text: item.model }),
                h("td", { class: "num", text: rateByFormat(item.macro, item.format) }),
                h("td", { class: "num", text: rateByFormat(item.micro, item.format) }),
                h("td", { class: "num", text: fmtInt(item.trials) }),
                h("td", { class: "num", text: `${item.tasksCovered} / ${aggregate.tasks}` }),
              ),
            ),
          ),
        ),
      ),
    ),
    h("p", {
      class: "section-note",
      text: "Metric families are never merged into a global score. Macro averages the" +
        " per-task rates with the suite weights; micro pools trials across tasks." +
        " Judge means depend on the rubric scale of the suite.",
    }),
  );
}

function renderLatency(section) {
  if ((payload.latency || []).length === 0) {
    section.append(h("p", { class: "empty-note", text: "No latency was recorded." }));
    return;
  }
  section.append(
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Task", "Model", "Median gen (ms)", "Median eval (ms)", "Median total (ms)"].map((label, index) =>
              h("th", { class: index > 1 ? "num" : null, text: label })
            ),
          ),
        ),
        h("tbody", null,
          payload.latency.map((row) =>
            h("tr", null,
              h("td", { class: "mono", text: row.task }),
              h("td", { class: "mono", text: row.model }),
              h("td", { class: "num", text: fmtMs(row.median_generation_ms) }),
              h("td", { class: "num", text: fmtMs(row.median_evaluation_ms) }),
              h("td", { class: "num", text: fmtMs(row.median_total_ms) }),
            ),
          ),
        ),
      ),
    ),
    h("p", {
      class: "section-note",
      text: "Generation covers the model call (provider retries included), evaluation" +
        " the evaluator run, total both. Values are per-task medians.",
    }),
  );
}

function renderConsumption(section) {
  section.append(
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Model", "Input tokens", "Output tokens", "Generation cost (USD)", "Evaluation cost (USD)"].map(
              (label, index) => h("th", { class: index > 0 ? "num" : null, text: label })
            ),
          ),
        ),
        h("tbody", null,
          (payload.consumption || []).map((row) =>
            h("tr", null,
              h("td", { class: "mono", text: row.model }),
              h("td", { class: "num", text: fmtInt(row.input_tokens) }),
              h("td", { class: "num", text: fmtInt(row.output_tokens) }),
              h("td", {
                class: "num",
                title: row.cost_partial ? "partial: some trials reported no cost" : null,
                text: fmtCost(row.cost_usd) + (row.cost_partial && row.cost_usd !== null ? " (partial)" : ""),
              }),
              h("td", { class: "num", text: fmtCost(row.evaluation_cost_usd) }),
            ),
          ),
        ),
      ),
    ),
    h("p", {
      class: "section-note",
      text: "Token and cost totals sum the values that were reported; n/a or (partial)" +
        " marks missing coverage rather than a real zero.",
    }),
  );
}

function renderReproducibility(section) {
  const lock = payload.manifest || {};
  const suite = (payload.overview || {}).suite || {};
  const sampling = payload.sampling || { method: "none", selections: {} };
  const samplingRows = Object.entries(sampling.selections || {}).map(([taskId, values]) =>
    h("tr", null,
      h("td", { class: "mono", text: taskId }),
      h("td", { class: "num", text: `${values.selected} / ${values.total}` }),
      h("td", { class: "num", text: values.seed === null || values.seed === undefined ? "n/a" : String(values.seed) }),
      h("td", {
        class: "mono",
        title: (values.selected_ids || []).join(", "),
        text: (values.selected_ids || []).length + " ids",
      }),
    ),
  );
  section.append(
    h("div", { class: "table-scroll" },
      h("table", null,
        h("tbody", null,
          [
            ["Suite definition fingerprint", shortSha(suite.definition_fingerprint) || "n/a"],
            ["Protocol lockfile", "benchmark.lock.json (checked by `suite verify`)"],
            ["Sampling", sampling.method === "none" ? "none (full suite)" : sampling.method],
            ["MiniEval version", lock.minieval_version || "n/a"],
          ].map(([field, value]) =>
            h("tr", null, h("td", { class: "dim" }, field), h("td", null, value))
          ),
        ),
      ),
    ),
  );
  if (samplingRows.length > 0) {
    section.append(
      h("div", { class: "table-scroll" },
        h("table", null,
          h("thead", null,
            h("tr", null,
              ["Task", "Selected / total", "Seed", "Selection"].map((label, index) =>
                h("th", { class: index > 0 ? "num" : null, text: label })
              ),
            ),
          ),
          h("tbody", null, samplingRows),
        ),
      ),
    );
  }
  section.append(
    h("p", {
      class: "section-note",
      text: "The lockfile pins the protocol: suite definition, dataset contents, sampling" +
        " selection, schemas, tests, rubric, models and parameters. Reproducibility is" +
        " protocol reproducibility: a hosted model can still change server-side.",
    }),
  );
}

const SECTIONS = [
  ["overview", "Overview", renderOverview],
  ["tasks", "Task breakdown", renderTaskBreakdown],
  ["comparison", "Model comparison", renderModelComparison],
  ["latency", "Latency", renderLatency],
  ["consumption", "Tokens & costs", renderConsumption],
  ["reproducibility", "Reproducibility", renderReproducibility],
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
