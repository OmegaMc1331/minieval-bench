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
  "disagreement": "disagreement (positions swapped returned opposite verdicts)",
  "unavailable": "unavailable",
  "judge_error": "judge error",
};

function outcomeLabel(outcome) {
  if (outcome.startsWith("win:")) {
    return "win: " + outcome.slice(4);
  }
  return OUTCOME_LABELS[outcome] || outcome;
}

function renderSummary(section) {
  section.append(
    h("p", {
      class: "section-note",
      text: `Blind comparison of ${payload.modelA} and ${payload.modelB}. Model names were hidden ` +
        "from the judge; each pair was judged twice with positions swapped, and verdicts were " +
        "mapped back to the original responses. Contradictory verdicts are kept as disagreements.",
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
          Object.entries(payload.summary).map(([measure, count]) =>
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
      text: "Counts of wins, ties, disagreements and unavailable judgments. No global ranking or "
        + "Elo score is computed: these tallies do not establish statistical superiority.",
    }),
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
