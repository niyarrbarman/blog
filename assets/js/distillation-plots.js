(function () {
  "use strict";

  var PLOT_IDS = [
    "window-length-chart",
    "source-eos-chart",
    "data-mix-chart",
    "selection-probe-chart",
    "retention-chart",
  ];
  var CONFIG = { displayModeBar: false, responsive: true };

  function colors() {
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    return dark
      ? {
          text: "#ebe5db",
          muted: "#b9ad9e",
          grid: "#4a4037",
          math: "#d4b86a",
          general: "#8ea4c8",
          accent: "#d9825b",
          green: "#79b99b",
          rose: "#d48aaa",
        }
      : {
          text: "#3d3632",
          muted: "#665b51",
          grid: "#ded4c6",
          math: "#8b6914",
          general: "#607d9f",
          accent: "#bd5d3f",
          green: "#3b9979",
          rose: "#a85b80",
        };
  }

  function layout(options) {
    var palette = colors();
    return Object.assign(
      {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Inter, sans-serif", size: 12, color: palette.text },
        margin: { l: 54, r: 20, t: 42, b: 106 },
        hoverlabel: { bgcolor: palette.text, bordercolor: palette.text, font: { color: "#ffffff" } },
        legend: { orientation: "h", y: -0.35, yanchor: "top", x: 0, font: { size: 11, color: palette.text } },
      },
      options || {}
    );
  }

  function axis(title) {
    var palette = colors();
    return {
      title: { text: title, font: { color: palette.text, size: 12 } },
      gridcolor: palette.grid,
      zerolinecolor: palette.grid,
      linecolor: palette.grid,
      tickfont: { color: palette.muted },
    };
  }

  function chartTitle(text) {
    var palette = colors();
    return { text: text, x: 0, xanchor: "left", y: 0.98, font: { size: 14, color: palette.text } };
  }

  function plot(id, data, figureLayout) {
    var element = document.getElementById(id);
    if (element && typeof Plotly !== "undefined") {
      Plotly.react(element, data, figureLayout, CONFIG);
    }
  }

  function renderWindowLength() {
    var palette = colors();
    plot(
      "window-length-chart",
      [
        {
          type: "bar",
          name: "tokens in over-length documents",
          x: ["4k", "16k", "32k"],
          y: [74.1, 40.0, 18.7],
          marker: { color: palette.general },
          text: ["74.1%", "40.0%", "18.7%"],
          textposition: "outside",
          cliponaxis: false,
          hovertemplate: "%{x} window: %{y:.1f}%<extra>over-length documents</extra>",
        },
        {
          type: "bar",
          name: "packed windows with no EOS",
          x: ["4k", "16k", "32k"],
          y: [51.8, 18.7, 7.3],
          marker: { color: palette.accent },
          text: ["51.8%", "18.7%", "7.3%"],
          textposition: "outside",
          cliponaxis: false,
          hovertemplate: "%{x} window: %{y:.1f}%<extra>no EOS</extra>",
        },
      ],
      layout({
        title: chartTitle("longer windows restored the end-of-sequence signal"),
        barmode: "group",
        bargap: 0.25,
        yaxis: Object.assign(axis("share of the blend (%)"), { range: [0, 88], ticksuffix: "%" }),
        xaxis: axis("training sequence length"),
      })
    );
  }

  function renderSourceEos() {
    var palette = colors();
    plot(
      "source-eos-chart",
      [
        {
          type: "bar",
          x: ["math_proofs_v2<br>(dropped)", "arXiv<br>(capped)", "cleaned blend"],
          y: [52.0, 10.0, 7.3],
          marker: { color: [palette.accent, palette.math, palette.general] },
          text: ["52%", "10%", "7.3%"],
          textposition: "outside",
          cliponaxis: false,
          hovertemplate: "%{x}: %{y:.1f}%<extra>window contains no EOS</extra>",
        },
      ],
      layout({
        title: chartTitle("32k was enough for the blend, not every source"),
        showlegend: false,
        margin: { l: 54, r: 20, t: 52, b: 62 },
        xaxis: Object.assign(axis(""), { tickfont: { color: palette.text, size: 11 } }),
        yaxis: Object.assign(axis("P(window contains no EOS)"), { range: [0, 62], ticksuffix: "%" }),
        shapes: [
          {
            type: "line",
            x0: 0,
            x1: 1,
            y0: 20,
            y1: 20,
            xref: "paper",
            yref: "y",
            line: { color: palette.muted, dash: "dot", width: 1 },
          },
        ],
        annotations: [
          { x: 0.98, y: 20, xref: "paper", yref: "y", text: "20% guide", showarrow: false, xanchor: "right", yanchor: "bottom", font: { color: palette.muted, size: 11 } },
        ],
      })
    );
  }

  function renderDataMix() {
    var palette = colors();
    plot(
      "data-mix-chart",
      [
        {
          type: "bar",
          orientation: "h",
          y: ["mathematics", "general text", "code", "science", "instruction", "STEM"],
          x: [42.94, 26.73, 19.4, 5.2, 3.07, 2.66],
          marker: { color: [palette.math, palette.general, palette.green, "#8f89a8", palette.rose, "#bf9a4b"] },
          text: ["42.94%", "26.73%", "19.40%", "5.20%", "3.07%", "2.66%"],
          textposition: "outside",
          cliponaxis: false,
          hovertemplate: "%{y}: %{x:.2f}%<extra>share of the cleaned 112B-token pool</extra>",
        },
      ],
      layout({
        title: chartTitle("the cleaned 112b-token pool"),
        showlegend: false,
        margin: { l: 100, r: 45, t: 42, b: 46 },
        xaxis: Object.assign(axis("share of cleaned pool"), { range: [0, 50], ticksuffix: "%" }),
        yaxis: Object.assign(axis(""), { autorange: "reversed" }),
      })
    );
  }

  function selectionTrace(name, color, math, fineweb, winner) {
    var line = { color: color, width: winner ? 4 : 2 };
    return [
      {
        type: "scatter",
        mode: "lines+markers",
        name: name,
        legendgroup: name,
        x: [0, 100],
        y: math,
        line: line,
        marker: { size: winner ? 9 : 7 },
        hovertemplate: name + ", %{x} iterations: %{y:.2f}<extra>math perplexity</extra>",
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: name,
        legendgroup: name,
        showlegend: false,
        x: [0, 100],
        y: fineweb,
        xaxis: "x2",
        yaxis: "y2",
        line: line,
        marker: { size: winner ? 9 : 7 },
        hovertemplate: name + ", %{x} iterations: %{y:.2f}<extra>FineWeb-Edu perplexity</extra>",
      },
    ];
  }

  function renderSelectionProbe() {
    var palette = colors();
    var traces = [];
    [
      ["candidate 1", "#c75b52", [3866.8862, 9.6617], [13764.9744, 228.8315], false],
      ["candidate 2", "#39a37d", [3379.2594, 23.4367], [30019.5969, 378.5886], false],
      ["candidate 3", "#6d62ad", [1657.3176, 6.7255], [10133.9196, 138.7728], false],
      ["candidate 4 - winner", palette.math, [2659.6681, 3.7869], [5182.3272, 38.9717], true],
      ["candidate 5", "#c475a0", [1276.3471, 9.4717], [4826.1427, 223.6296], false],
    ].forEach(function (candidate) {
      traces = traces.concat(selectionTrace(candidate[0], candidate[1], candidate[2], candidate[3], candidate[4]));
    });

    plot(
      "selection-probe-chart",
      traces,
      layout({
        title: chartTitle("the short distillation probe changed the ranking"),
        height: 560,
        margin: { l: 58, r: 20, t: 76, b: 134 },
        legend: { orientation: "h", y: -0.35, yanchor: "top", x: 0, font: { size: 10, color: palette.text } },
        xaxis: Object.assign(axis("distillation iterations"), { domain: [0, 1], range: [-5, 105] }),
        yaxis: Object.assign(axis("math perplexity"), { type: "log", domain: [0.55, 0.86] }),
        xaxis2: Object.assign(axis("distillation iterations"), { domain: [0, 1], range: [-5, 105], anchor: "y2" }),
        yaxis2: Object.assign(axis("FineWeb-Edu perplexity"), { type: "log", domain: [0, 0.30] }),
        annotations: [
          { text: "math held-out set", x: 0, y: 0.93, xref: "paper", yref: "paper", showarrow: false, xanchor: "left", font: { color: palette.text, size: 12 } },
          { text: "FineWeb-Edu held-out set", x: 0, y: 0.40, xref: "paper", yref: "paper", showarrow: false, xanchor: "left", font: { color: palette.text, size: 12 } },
        ],
      })
    );
  }

  function renderRetention() {
    var palette = colors();
    plot(
      "retention-chart",
      [
        {
          type: "scatter",
          mode: "lines+markers+text",
          name: "mathematics, compressed",
          x: [23.22, 13.98, 8.36],
          y: [100, 101.2, 99.2],
          text: ["100.0%", "101.2%", "99.2%"],
          textposition: "top center",
          line: { color: palette.math, width: 3 },
          marker: { size: 10 },
          hovertemplate: "%{x:.2f}B parameters: %{y:.1f}%<extra>math retained</extra>",
        },
        {
          type: "scatter",
          mode: "lines+markers+text",
          name: "general knowledge, compressed",
          x: [23.22, 13.98, 8.36],
          y: [100, 96.0, 92.8],
          text: ["100.0%", "96.0%", "92.8%"],
          textposition: "bottom center",
          line: { color: palette.general, width: 3, dash: "dash" },
          marker: { size: 10, symbol: "square" },
          hovertemplate: "%{x:.2f}B parameters: %{y:.1f}%<extra>general retained</extra>",
        },
        {
          type: "scatter",
          mode: "markers+text",
          name: "Luciole-8B-Base, from scratch",
          x: [8, 8],
          y: [63.2, 96.0],
          text: ["math 63.2%", "general 96.0%"],
          textposition: "middle left",
          marker: { size: 13, symbol: "x", color: palette.accent, line: { width: 2 } },
          hovertemplate: "%{y:.1f}%<extra>Luciole-8B-Base retained</extra>",
        },
      ],
      layout({
        title: chartTitle("capability retained through compression"),
        margin: { l: 54, r: 20, t: 72, b: 120 },
        yaxis: Object.assign(axis("retained relative to the teacher (%)"), { range: [58, 108], ticksuffix: "%" }),
        xaxis: Object.assign(axis("parameters (B)"), { autorange: "reversed", range: [24, 7] }),
        shapes: [
          { type: "line", x0: 7, x1: 24, y0: 100, y1: 100, line: { color: colors().muted, width: 1, dash: "dot" } },
        ],
      })
    );
  }

  function renderAll() {
    if (typeof Plotly === "undefined") return;
    renderWindowLength();
    renderSourceEos();
    renderDataMix();
    renderSelectionProbe();
    renderRetention();
  }

  function resizePlots() {
    if (typeof Plotly === "undefined") return;
    PLOT_IDS.forEach(function (id) {
      var element = document.getElementById(id);
      if (element && element.data) Plotly.Plots.resize(element);
    });
  }

  renderAll();
  document.addEventListener("themechange", renderAll);
  new MutationObserver(renderAll).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  window.addEventListener("resize", resizePlots);
})();
