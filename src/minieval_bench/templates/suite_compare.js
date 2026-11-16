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

function fmtDeltaRate(value) {
  if (value === null || value === undefined) return "n/a";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)} pp`;
}

function fmtDeltaMs(value) {
  if (value === null || value === undefined) return "n/a";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}`;
}

function deltaClass(value) {
  if (value === null || value === undefined) return "delta-na";
  if (value > 0) return "delta-up";
  if (value < 0) return "delta-down";
  return "delta-na";
}

const payload = JSON.parse(document.getElementById("report-data").textContent);
const runs = payload.runs;

function renderRuns(section) {
  section.append(
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Run", "Suite", "Version", "Executed (UTC)", "Location"].map((label) =>
              h("th", { text: label })
            ),
          ),
        ),
        h("tbody", null,
          runs.map((run) =>
            h("tr", null,
              h("td", { class: "mono", text: run.label }),
              h("td", { text: run.suite || "n/a" }),
              h("td", { text: run.version || "n/a" }),
              h("td", { text: run.created_at_utc || "n/a" }),
              h("td", { class: "mono", text: run.path }),
            ),
          ),
        ),
      ),
    ),
  );
}

function renderProtocol(section) {
  if (payload.protocolEqual) {
    section.append(
      h("p", {
        class: "bar-note",
        text: "The two runs share the same protocol lockfile: identical suite" +
          " definition, datasets, sampling, models, parameters and evaluators.",
      }),
    );
    return;
  }
  section.append(
    h("div", { class: "warning-box", role: "alert" },
      h("strong", { text: "The protocols differ: no score differences are computed" }),
      h("ul", null, payload.differences.map((difference) => h("li", { text: difference }))),
      h("p", {
        text: "Individual results stay available in each run directory; models are not" +
          " ranked across different protocols.",
      }),
    ),
  );
}

function renderTasks(section) {
  if (!payload.tasks) {
    section.append(h("p", { class: "empty-note", text: "No comparison tables: the protocols differ." }));
    return;
  }
  for (const task of payload.tasks) {
    const familyColumns = task.families;
    section.append(
      h("h3", { text: task.id }),
      h("div", { class: "table-scroll" },
        h("table", null,
          h("thead", null,
            h("tr", null,
              h("th", { text: "Model" }),
              familyColumns.flatMap((family) => [
                h("th", { text: family.label }),
                h("th", { class: "num", text: "Δ" }),
              ]),
            ),
          ),
          h("tbody", null,
            task.rows.map((row) =>
              h("tr", null,
                h("td", { class: "mono", text: row.model }),
                familyColumns.flatMap((family) => {
                  const values = row.values[family.family] || [];
                  const first = values[0];
                  const last = values[values.length - 1];
                  const delta =
                    first === null || first === undefined || last === null || last === undefined
                      ? null
                      : last - first;
                  return [
                    h("td", {
                      class: "num",
                      text: values
                        .map((value) =>
                          family.family === "judge" ? fmtScore(value) : fmtRate(value)
                        )
                        .join(" / "),
                    }),
                    h("td", {
                      class: "num " + deltaClass(delta),
                      text: family.family === "judge" ? fmtScore(delta) : fmtDeltaRate(delta),
                    }),
                  ];
                }),
              ),
            ),
          ),
        ),
      ),
      h("div", { class: "table-scroll" },
        h("table", null,
          h("thead", null,
            h("tr", null,
              ["Model", ...runs.map((run) => `${run.label}: median total (ms)`), "Δ (ms)"].map(
                (label, index) => h("th", { class: index > 0 ? "num" : null, text: label })
              ),
            ),
          ),
          h("tbody", null,
            task.latency.map((row) => {
              const first = row.values[0];
              const last = row.values[row.values.length - 1];
              const delta =
                first === null || first === undefined || last === null || last === undefined
                  ? null
                  : last - first;
              return h("tr", null,
                h("td", { class: "mono", text: row.model }),
                row.values.map((value) => h("td", { class: "num", text: fmtMs(value) })),
                h("td", { class: "num " + deltaClass(delta), text: fmtDeltaMs(delta) }),
              );
            }),
          ),
        ),
      ),
    );
  }
}

function renderPaired(section) {
  if (!payload.paired || payload.paired.length === 0) {
    section.append(
      h("p", {
        class: "empty-note",
        text: "No paired statistics: they require exactly two runs with shared models" +
          " and question ids per task.",
      }),
    );
    return;
  }
  const last = runs[runs.length - 1].label;
  const first = runs[0].label;
  section.append(
    h("p", {
      class: "section-note",
      text: `Per-question differences (${last} minus ${first}), repeats averaged first,` +
        " computed only on questions present in both runs. The interval is a bootstrap" +
        " over question means and is not a significance test.",
    }),
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            ["Task", "Model", "Metric", "Questions", "Mean diff", "95% CI low", "95% CI high"].map(
              (label, index) => h("th", { class: index > 2 ? "num" : null, text: label })
            ),
          ),
        ),
        h("tbody", null,
          payload.paired.map((entry) => {
            const fmt = (value) =>
              value === null || value === undefined ? "n/a" : value.toFixed(3);
            return h("tr", null,
              h("td", { class: "mono", text: entry.task }),
              h("td", { class: "mono", text: entry.model }),
              h("td", { text: entry.metric || "pass rate" }),
              h("td", { class: "num", text: fmtInt(entry.units) }),
              h("td", { class: "num", text: fmt(entry.differences_summary.mean) }),
              h("td", { class: "num", text: fmt(entry.ci95.low) }),
              h("td", { class: "num", text: fmt(entry.ci95.high) }),
            );
          }),
        ),
      ),
    ),
  );
}

const SECTIONS = [
  ["runs", "Compared runs", renderRuns],
  ["protocol", "Protocol", renderProtocol],
  ["tasks", "Task comparison", renderTasks],
  ["paired", "Paired differences", renderPaired],
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
