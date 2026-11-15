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

const payload = JSON.parse(document.getElementById("report-data").textContent);

const OUTCOME_LABELS = {
  "win": "win",
  "tie": "tie",
  "mixed_verdict": "mixed verdict (a win and a tie: not a confirmed win)",
  "disagreement": "disagreement (positions swapped returned opposite verdicts)",
  "incomplete": "incomplete (fewer than two judgments)",
  "unavailable": "unavailable",
  "judge_error": "judge error",
};

function outcomeLabel(outcome) {
  if (outcome.startsWith("win:")) {
    return "win: " + outcome.slice(4);
  }
  return OUTCOME_LABELS[outcome] || outcome;
}

const SUMMARY_COUNT_KEYS = [
  "comparisons",
  "wins", "ties", "mixed_verdicts", "disagreements", "judge_errors", "incomplete",
  "unavailable", "judge_calls",
];

function summaryCountEntries(summary) {
  const byPrefix = new Map();
  for (const [key, value] of Object.entries(summary)) {
    if (key.startsWith("wins:")) byPrefix.set(key, value);
  }
  const rows = [];
  for (const key of SUMMARY_COUNT_KEYS) {
    if (key === "wins") {
      for (const [prefKey, value] of byPrefix) rows.push([prefKey, value]);
    } else if (summary[key] !== undefined) {
      rows.push([key, summary[key]]);
    }
  }
  for (const [key, value] of Object.entries(summary)) {
    if (!rows.some(([rowKey]) => rowKey === key) && typeof value === "number") {
      rows.push([key, value]);
    }
  }
  return rows;
}

function renderSummary(section) {
  const summary = payload.summary;
  const usage = [
    ["input_tokens", summary.input_tokens],
    ["output_tokens", summary.output_tokens],
    ["cost_usd", summary.cost_usd],
    ["usage_known_judgments", summary.usage_known_judgments],
  ];
  section.append(
    h("p", {
      class: "section-note",
      text: `Blind comparison of ${payload.modelA} and ${payload.modelB}. Model names were hidden ` +
        "from the judge; each pair was judged twice with positions swapped, and verdicts were " +
        "mapped back to the original responses. Only concordant verdicts count as wins; a win " +
        "combined with a tie is reported as a mixed verdict.",
    }),
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            h("th", { text: "Measure" }),
            h("th", { class: "num", text: "Count" }),
          ),
        ),
        h("tbody", null,
          summaryCountEntries(summary).map(([measure, count]) =>
            h("tr", null,
              h("td", { class: "mono", text: measure }),
              h("td", { class: "num", text: String(count) }),
            ),
          ),
        ),
      ),
    ),
    h("p", {
      class: "section-note",
      text: "Counts of wins, ties, mixed verdicts, disagreements and unavailable judgments. " +
        "Token and cost totals cover only the judgments that reported them " +
        `(${summary.usage_known_judgments} of ${summary.judge_calls} judgment(s)); a partial ` +
        "total is not the full cost. No global ranking or Elo score is computed: these tallies " +
        "do not establish statistical superiority.",
    }),
    h("div", { class: "table-scroll" },
      h("table", null,
        h("thead", null,
          h("tr", null,
            h("th", { text: "Usage" }),
            h("th", { class: "num", text: "Value" }),
          ),
        ),
        h("tbody", null,
          usage.map(([measure, value]) =>
            h("tr", null,
              h("td", { class: "mono", text: measure }),
              h("td", {
                class: "num",
                text: value === null || value === undefined
                  ? "n/a"
                  : measure === "cost_usd" ? value.toFixed(6) : String(value),
              }),
            ),
          ),
        ),
      ),
    ),
  );
}

function renderPairs(section) {
  if (payload.records.length === 0) {
    section.append(h("p", { class: "empty-note", text: "No comparable pairs were found." }));
    return;
  }
  section.append(
    h("p", {
      class: "section-note",
      text: "Responses are shown side by side after judging; model identities were hidden during evaluation.",
    }),
    payload.records.map((record) =>
      h("div", { class: "multi-chart" },
        h("h4", {
          text: `${record.entry_id} · repeat ${record.repeat} · ${outcomeLabel(record.outcome)}`,
        }),
        record.error_message
          ? h("p", { class: "bar-note", text: record.error_message })
          : null,
        h("div", { class: "pair-grid" },
          h("div", null,
            h("p", { class: "sub", text: `Response A — ${record.model_a}` }),
            h("pre", { text: record.response_a }),
          ),
          h("div", null,
            h("p", { class: "sub", text: `Response B — ${record.model_b}` }),
            h("pre", { text: record.response_b }),
          ),
        ),
        record.judgments.length > 0
          ? h("ul", { class: "judgment-list" },
              record.judgments.map((judgment) =>
                h("li", null,
                  h("span", { class: "mono", text: judgment.order + " order" }),
                  ` → ${judgment.preferred}`
                  + (judgment.reason ? ` — ${judgment.reason}` : ""),
                ),
              ),
            )
          : null,
      ),
    ),
  );
}

const SECTIONS = [
  ["summary", "Pairwise summary", renderSummary],
  ["pairs", "Compared responses", renderPairs],
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
