(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // Tab navigation
  // ---------------------------------------------------------------------------
  document.querySelectorAll(".tab-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll(".tab-btn").forEach(function (b) {
        b.classList.remove("active");
      });
      document.querySelectorAll(".tab-panel").forEach(function (p) {
        p.classList.add("hidden");
      });
      btn.classList.add("active");
      var tab = btn.getAttribute("data-tab");
      var panel = document.getElementById("tab-" + tab);
      if (panel) {
        panel.classList.remove("hidden");
      }
    });
  });

  // ---------------------------------------------------------------------------
  // API helper
  // ---------------------------------------------------------------------------
  async function api(path, body) {
    var res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      var errText = await res.text();
      var parsed;
      try {
        parsed = JSON.parse(errText);
      } catch (_) {
        parsed = null;
      }
      var msg = (parsed && (parsed.detail || parsed.message)) || errText || "Request failed (" + res.status + ")";
      throw new Error(msg);
    }
    return res.json();
  }

  // ---------------------------------------------------------------------------
  // UI helpers
  // ---------------------------------------------------------------------------
  function showError(id, msg) {
    var el = document.getElementById(id + "-error");
    if (el) {
      el.textContent = msg;
      el.classList.remove("hidden");
    }
  }

  function clearError(id) {
    var el = document.getElementById(id + "-error");
    if (el) {
      el.textContent = "";
      el.classList.add("hidden");
    }
  }

  function setSpinner(id, visible) {
    var el = document.getElementById(id + "-spinner");
    if (el) {
      if (visible) {
        el.classList.remove("hidden");
      } else {
        el.classList.add("hidden");
      }
    }
  }

  function showResults(id) {
    var el = document.getElementById(id + "-results");
    if (el) {
      el.classList.remove("hidden");
    }
  }

  function hideResults(id) {
    var el = document.getElementById(id + "-results");
    if (el) {
      el.classList.add("hidden");
    }
  }

  // ---------------------------------------------------------------------------
  // renderPKCards(containerId, items)
  // items: [{label, value, unit}]
  // ---------------------------------------------------------------------------
  function renderPKCards(containerId, items) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";
    items.forEach(function (item) {
      var card = document.createElement("div");
      card.className = "pk-card";
      var labelEl = document.createElement("span");
      labelEl.className = "pk-label";
      labelEl.textContent = item.label;
      var valueEl = document.createElement("span");
      valueEl.className = "pk-value";
      valueEl.textContent = typeof item.value === "number" ? item.value.toPrecision(4) : item.value;
      var unitEl = document.createElement("span");
      unitEl.className = "pk-unit";
      unitEl.textContent = item.unit || "";
      card.appendChild(labelEl);
      card.appendChild(valueEl);
      card.appendChild(unitEl);
      container.appendChild(card);
    });
  }

  // ---------------------------------------------------------------------------
  // drawLineChart(svgId, datasets)
  // datasets: [{label, color, times, values}]
  // W=600 H=280 with margins
  // ---------------------------------------------------------------------------
  function drawLineChart(svgId, datasets) {
    var svg = document.getElementById(svgId);
    if (!svg) return;
    svg.innerHTML = "";

    var W = 600, H = 280;
    var ML = 55, MR = 20, MT = 20, MB = 45;
    var pw = W - ML - MR;
    var ph = H - MT - MB;

    var allTimes = [];
    var allValues = [];
    datasets.forEach(function (d) {
      allTimes = allTimes.concat(d.times || []);
      allValues = allValues.concat(d.values || []);
    });

    if (allTimes.length === 0) return;

    var tMin = 0;
    var tMax = Math.max.apply(null, allTimes);
    var vMin = 0;
    var vMax = Math.max.apply(null, allValues) * 1.1 || 1;

    function tx(t) { return ML + ((t - tMin) / (tMax - tMin || 1)) * pw; }
    function ty(v) { return MT + ph - ((v - vMin) / (vMax - vMin || 1)) * ph; }

    var ns = "http://www.w3.org/2000/svg";

    // Axes
    function line(x1, y1, x2, y2, color, width) {
      var el = document.createElementNS(ns, "line");
      el.setAttribute("x1", x1); el.setAttribute("y1", y1);
      el.setAttribute("x2", x2); el.setAttribute("y2", y2);
      el.setAttribute("stroke", color || "var(--axis-color)");
      el.setAttribute("stroke-width", width || 1);
      svg.appendChild(el);
    }

    line(ML, MT, ML, MT + ph);
    line(ML, MT + ph, ML + pw, MT + ph);

    // Y ticks
    var yTicks = 5;
    for (var i = 0; i <= yTicks; i++) {
      var v = vMin + (vMax - vMin) * i / yTicks;
      var yp = ty(v);
      line(ML - 4, yp, ML, yp, "var(--axis-color)");
      var tLabel = document.createElementNS(ns, "text");
      tLabel.setAttribute("x", ML - 7);
      tLabel.setAttribute("y", yp + 4);
      tLabel.setAttribute("text-anchor", "end");
      tLabel.setAttribute("font-size", "10");
      tLabel.setAttribute("fill", "var(--label-color)");
      tLabel.textContent = v.toPrecision(3);
      svg.appendChild(tLabel);
      if (i > 0) {
        var gridLine = document.createElementNS(ns, "line");
        gridLine.setAttribute("x1", ML); gridLine.setAttribute("y1", yp);
        gridLine.setAttribute("x2", ML + pw); gridLine.setAttribute("y2", yp);
        gridLine.setAttribute("stroke", "var(--axis-color)");
        gridLine.setAttribute("stroke-width", "0.5");
        gridLine.setAttribute("stroke-dasharray", "3,3");
        svg.appendChild(gridLine);
      }
    }

    // X ticks
    var xTicks = 6;
    for (var j = 0; j <= xTicks; j++) {
      var t = tMin + (tMax - tMin) * j / xTicks;
      var xp = tx(t);
      line(xp, MT + ph, xp, MT + ph + 4, "var(--axis-color)");
      var xLabel = document.createElementNS(ns, "text");
      xLabel.setAttribute("x", xp);
      xLabel.setAttribute("y", MT + ph + 16);
      xLabel.setAttribute("text-anchor", "middle");
      xLabel.setAttribute("font-size", "10");
      xLabel.setAttribute("fill", "var(--label-color)");
      xLabel.textContent = t.toFixed(1);
      svg.appendChild(xLabel);
    }

    // Axis labels
    var xAxisLabel = document.createElementNS(ns, "text");
    xAxisLabel.setAttribute("x", ML + pw / 2);
    xAxisLabel.setAttribute("y", H - 5);
    xAxisLabel.setAttribute("text-anchor", "middle");
    xAxisLabel.setAttribute("font-size", "11");
    xAxisLabel.setAttribute("fill", "var(--label-color)");
    xAxisLabel.textContent = "Time (h)";
    svg.appendChild(xAxisLabel);

    var yAxisLabel = document.createElementNS(ns, "text");
    yAxisLabel.setAttribute("x", -(MT + ph / 2));
    yAxisLabel.setAttribute("y", 12);
    yAxisLabel.setAttribute("text-anchor", "middle");
    yAxisLabel.setAttribute("font-size", "11");
    yAxisLabel.setAttribute("fill", "var(--label-color)");
    yAxisLabel.setAttribute("transform", "rotate(-90)");
    yAxisLabel.textContent = "Concentration (mg/L)";
    svg.appendChild(yAxisLabel);

    // Lines per dataset
    var colors = ["var(--accent-1)", "var(--accent-2)", "var(--accent-3)", "var(--accent-warn)"];
    datasets.forEach(function (ds, idx) {
      if (!ds.times || ds.times.length === 0) return;
      var pts = ds.times.map(function (t, k) {
        return tx(t) + "," + ty(ds.values[k]);
      }).join(" ");
      var polyline = document.createElementNS(ns, "polyline");
      polyline.setAttribute("points", pts);
      polyline.setAttribute("fill", "none");
      polyline.setAttribute("stroke", ds.color || colors[idx % colors.length]);
      polyline.setAttribute("stroke-width", "2");
      polyline.setAttribute("stroke-linejoin", "round");
      svg.appendChild(polyline);
    });

    // Legend
    if (datasets.length > 1) {
      datasets.forEach(function (ds, idx) {
        var lx = ML + idx * 110;
        var ly = H - 8;
        var rectEl = document.createElementNS(ns, "rect");
        rectEl.setAttribute("x", lx); rectEl.setAttribute("y", ly - 8);
        rectEl.setAttribute("width", 14); rectEl.setAttribute("height", 3);
        rectEl.setAttribute("fill", ds.color || colors[idx % colors.length]);
        svg.appendChild(rectEl);
        var lLabel = document.createElementNS(ns, "text");
        lLabel.setAttribute("x", lx + 18);
        lLabel.setAttribute("y", ly - 5);
        lLabel.setAttribute("font-size", "10");
        lLabel.setAttribute("fill", "var(--label-color)");
        lLabel.textContent = ds.label || "";
        svg.appendChild(lLabel);
      });
    }
  }

  // ---------------------------------------------------------------------------
  // drawBarChart(svgId, labels, values, color)
  // ---------------------------------------------------------------------------
  function drawBarChart(svgId, labels, values, color) {
    var svg = document.getElementById(svgId);
    if (!svg) return;
    svg.innerHTML = "";

    var W = 600, H = 280;
    var ML = 55, MR = 20, MT = 20, MB = 55;
    var pw = W - ML - MR;
    var ph = H - MT - MB;
    var ns = "http://www.w3.org/2000/svg";

    if (!labels || labels.length === 0) return;

    var vMax = Math.max.apply(null, values) * 1.15 || 1;
    var barW = pw / labels.length * 0.6;
    var gap = pw / labels.length;
    var fillColor = color || "var(--accent-1)";

    function tyBar(v) { return MT + ph - (v / vMax) * ph; }

    // Axes
    function line(x1, y1, x2, y2) {
      var el = document.createElementNS(ns, "line");
      el.setAttribute("x1", x1); el.setAttribute("y1", y1);
      el.setAttribute("x2", x2); el.setAttribute("y2", y2);
      el.setAttribute("stroke", "var(--axis-color)");
      el.setAttribute("stroke-width", 1);
      svg.appendChild(el);
    }
    line(ML, MT, ML, MT + ph);
    line(ML, MT + ph, ML + pw, MT + ph);

    // Y ticks
    for (var i = 0; i <= 5; i++) {
      var v = vMax * i / 5;
      var yp = tyBar(v);
      line(ML - 4, yp, ML, yp);
      var tLabel = document.createElementNS(ns, "text");
      tLabel.setAttribute("x", ML - 7);
      tLabel.setAttribute("y", yp + 4);
      tLabel.setAttribute("text-anchor", "end");
      tLabel.setAttribute("font-size", "10");
      tLabel.setAttribute("fill", "var(--label-color)");
      tLabel.textContent = v.toPrecision(3);
      svg.appendChild(tLabel);
    }

    // Bars
    labels.forEach(function (lbl, idx) {
      var x = ML + idx * gap + (gap - barW) / 2;
      var barH = (values[idx] / vMax) * ph;
      var y = MT + ph - barH;
      var rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", x); rect.setAttribute("y", y);
      rect.setAttribute("width", barW); rect.setAttribute("height", barH);
      rect.setAttribute("fill", fillColor);
      rect.setAttribute("rx", 2);
      svg.appendChild(rect);
      var xLabel = document.createElementNS(ns, "text");
      xLabel.setAttribute("x", x + barW / 2);
      xLabel.setAttribute("y", MT + ph + 15);
      xLabel.setAttribute("text-anchor", "middle");
      xLabel.setAttribute("font-size", "10");
      xLabel.setAttribute("fill", "var(--label-color)");
      xLabel.textContent = lbl;
      svg.appendChild(xLabel);
    });
  }

  // ---------------------------------------------------------------------------
  // Compound Explorer
  // ---------------------------------------------------------------------------
  document.getElementById("ex-run").addEventListener("click", async function () {
    var id = "ex";
    clearError(id);
    hideResults(id);
    setSpinner(id, true);

    var smiles = document.getElementById("ex-smiles").value.trim();
    var dose = parseFloat(document.getElementById("ex-dose").value);
    var route = document.getElementById("ex-route").value;

    try {
      var data = await api("/predict/full", { smiles: smiles, dose_mg: dose, route: route });
      setSpinner(id, false);

      var pk = data.pk_summary || data.pk || {};
      var adme = data.adme || {};
      var cards = [];

      if (pk.cmax !== undefined) cards.push({ label: "Cmax", value: pk.cmax, unit: "mg/L" });
      if (pk.tmax !== undefined) cards.push({ label: "Tmax", value: pk.tmax, unit: "h" });
      if (pk.auc_0_inf !== undefined) cards.push({ label: "AUC∞", value: pk.auc_0_inf, unit: "mg·h/L" });
      if (pk.t_half !== undefined) cards.push({ label: "t½", value: pk.t_half, unit: "h" });
      if (pk.cl_L_per_h !== undefined) cards.push({ label: "CL", value: pk.cl_L_per_h, unit: "L/h" });
      if (pk.vdss_L !== undefined) cards.push({ label: "Vdss", value: pk.vdss_L, unit: "L" });
      if (pk.f_oral !== undefined) cards.push({ label: "F", value: pk.f_oral, unit: "" });

      renderPKCards("ex-cards", cards);

      var times = data.times || data.time_h || [];
      var concentrations = data.concentrations || data.plasma_conc_mg_L || [];
      drawLineChart("ex-chart", [{ label: smiles.slice(0, 20), times: times, values: concentrations }]);

      // ADME grid
      var admeContainer = document.getElementById("ex-adme");
      if (admeContainer) {
        admeContainer.innerHTML = "";
        Object.keys(adme).forEach(function (key) {
          var item = document.createElement("div");
          item.className = "adme-item";
          var keyEl = document.createElement("span");
          keyEl.className = "adme-key";
          keyEl.textContent = key;
          var valEl = document.createElement("span");
          valEl.className = "adme-val";
          var v = adme[key];
          valEl.textContent = typeof v === "number" ? v.toPrecision(3) : v;
          item.appendChild(keyEl);
          item.appendChild(valEl);
          admeContainer.appendChild(item);
        });
      }

      showResults(id);
    } catch (err) {
      setSpinner(id, false);
      showError(id, err.message);
    }
  });

  // ---------------------------------------------------------------------------
  // PK Simulator
  // ---------------------------------------------------------------------------
  document.getElementById("sim-run").addEventListener("click", async function () {
    var id = "sim";
    clearError(id);
    hideResults(id);
    setSpinner(id, true);

    var drug = {
      name: document.getElementById("sim-name").value.trim(),
      mw: parseFloat(document.getElementById("sim-mw").value),
      logP: parseFloat(document.getElementById("sim-logp").value),
      pka: [14.0],
      drug_type: document.getElementById("sim-drug-type").value,
      fup: parseFloat(document.getElementById("sim-fup").value),
      rbp: 1.0,
      clint_hepatic_L_per_h: parseFloat(document.getElementById("sim-clint").value),
      clr_L_per_h: 0.0,
      peff: parseFloat(document.getElementById("sim-peff").value),
      vdss_L_per_kg: parseFloat(document.getElementById("sim-vdss").value),
      route_of_elimination: "hepatic",
    };
    var payload = {
      drug: drug,
      dose_mg: parseFloat(document.getElementById("sim-dose").value),
      route: document.getElementById("sim-route").value,
      duration_h: parseFloat(document.getElementById("sim-dur").value),
      body_weight: parseFloat(document.getElementById("sim-bw").value),
    };

    try {
      var data = await api("/simulate", payload);
      setSpinner(id, false);

      var pk = data.pk_summary || {};
      var cards = [];
      if (pk.cmax !== undefined) cards.push({ label: "Cmax", value: pk.cmax, unit: "mg/L" });
      if (pk.tmax !== undefined) cards.push({ label: "Tmax", value: pk.tmax, unit: "h" });
      if (pk.auc_0_inf !== undefined) cards.push({ label: "AUC∞", value: pk.auc_0_inf, unit: "mg·h/L" });
      if (pk.t_half !== undefined) cards.push({ label: "t½", value: pk.t_half, unit: "h" });
      if (pk.cl_L_per_h !== undefined) cards.push({ label: "CL", value: pk.cl_L_per_h, unit: "L/h" });
      if (pk.vdss_L !== undefined) cards.push({ label: "Vdss", value: pk.vdss_L, unit: "L" });
      renderPKCards("sim-cards", cards);

      var times = data.times || data.time_h || [];
      var conc = data.concentrations || data.plasma_conc_mg_L || [];
      drawLineChart("sim-chart", [{ label: drug.name, times: times, values: conc }]);

      showResults(id);
    } catch (err) {
      setSpinner(id, false);
      showError(id, err.message);
    }
  });

  // ---------------------------------------------------------------------------
  // Dose Optimizer
  // ---------------------------------------------------------------------------
  document.getElementById("opt-run").addEventListener("click", async function () {
    var id = "opt";
    clearError(id);
    hideResults(id);
    setSpinner(id, true);

    var drug = {
      name: document.getElementById("opt-name").value.trim(),
      mw: parseFloat(document.getElementById("opt-mw").value),
      logP: parseFloat(document.getElementById("opt-logp").value),
      pka: [14.0],
      drug_type: document.getElementById("opt-drug-type").value,
      fup: parseFloat(document.getElementById("opt-fup").value),
      rbp: 1.0,
      clint_hepatic_L_per_h: parseFloat(document.getElementById("opt-clint").value),
      clr_L_per_h: 0.0,
      peff: parseFloat(document.getElementById("opt-peff").value),
      vdss_L_per_kg: parseFloat(document.getElementById("opt-vdss").value),
      route_of_elimination: "hepatic",
    };
    var payload = {
      drug: drug,
      target_cmin: parseFloat(document.getElementById("opt-target-min").value),
      target_cmax: parseFloat(document.getElementById("opt-target-max").value),
      route: document.getElementById("opt-route").value,
      body_weight: parseFloat(document.getElementById("opt-bw").value),
      n_episodes: 500,
    };

    try {
      var data = await api("/dose/rl_optimize", payload);
      setSpinner(id, false);

      var cards = [];
      if (data.optimal_dose_mg !== undefined) cards.push({ label: "Optimal Dose", value: data.optimal_dose_mg, unit: "mg" });
      if (data.interval_h !== undefined) cards.push({ label: "Interval", value: data.interval_h, unit: "h" });
      if (data.predicted_cmax !== undefined) cards.push({ label: "Pred. Cmax", value: data.predicted_cmax, unit: "mg/L" });
      if (data.predicted_cmin !== undefined) cards.push({ label: "Pred. Cmin", value: data.predicted_cmin, unit: "mg/L" });
      renderPKCards("opt-cards", cards);

      var doses = data.dose_history || [];
      var rewards = data.reward_history || [];
      if (doses.length > 0) {
        drawBarChart("opt-chart", doses.map(function (d, i) { return "ep" + (i + 1); }), doses, "var(--accent-2)");
      } else if (rewards.length > 0) {
        drawBarChart("opt-chart", rewards.map(function (_, i) { return "" + (i + 1); }), rewards, "var(--accent-3)");
      }

      showResults(id);
    } catch (err) {
      setSpinner(id, false);
      showError(id, err.message);
    }
  });

  // ---------------------------------------------------------------------------
  // DDI Checker
  // ---------------------------------------------------------------------------
  document.getElementById("ddi-run").addEventListener("click", async function () {
    var id = "ddi";
    clearError(id);
    hideResults(id);
    setSpinner(id, true);

    function buildDrug(prefix) {
      return {
        name: document.getElementById(prefix + "-name").value.trim(),
        mw: parseFloat(document.getElementById(prefix + "-mw").value),
        logP: parseFloat(document.getElementById(prefix + "-logp").value),
        pka: [14.0],
        drug_type: document.getElementById(prefix + "-drug-type").value,
        fup: parseFloat(document.getElementById(prefix + "-fup").value),
        rbp: 1.0,
        clint_hepatic_L_per_h: parseFloat(document.getElementById(prefix + "-clint").value),
        clr_L_per_h: 0.0,
        peff: parseFloat(document.getElementById(prefix + "-peff").value),
        vdss_L_per_kg: parseFloat(document.getElementById(prefix + "-vdss").value),
        route_of_elimination: "hepatic",
      };
    }

    var victim = buildDrug("ddi-v");
    var perpetrator = buildDrug("ddi-p");
    var ki = parseFloat(document.getElementById("ddi-p-ki").value);

    var payload = {
      victim: victim,
      perpetrator: perpetrator,
      ki_uM: ki,
      victim_dose_mg: parseFloat(document.getElementById("ddi-v-dose").value),
      perpetrator_dose_mg: parseFloat(document.getElementById("ddi-p-dose").value),
      route: document.getElementById("ddi-route").value,
      duration_h: parseFloat(document.getElementById("ddi-dur").value),
    };

    try {
      var data = await api("/ddi/simulate", payload);
      setSpinner(id, false);

      var cards = [];
      if (data.auc_ratio !== undefined) cards.push({ label: "AUC Ratio", value: data.auc_ratio, unit: "" });
      if (data.cmax_ratio !== undefined) cards.push({ label: "Cmax Ratio", value: data.cmax_ratio, unit: "" });
      if (data.risk_category !== undefined) cards.push({ label: "Risk", value: data.risk_category, unit: "" });
      if (data.mechanism !== undefined) cards.push({ label: "Mechanism", value: data.mechanism, unit: "" });
      renderPKCards("ddi-cards", cards);

      var times = data.times || data.time_h || [];
      var victimConc = data.victim_concentrations || data.victim_conc || [];
      var perpetratorConc = data.perpetrator_concentrations || data.perpetrator_conc || [];
      var datasets = [];
      if (victimConc.length > 0) datasets.push({ label: victim.name, times: times, values: victimConc, color: "var(--accent-1)" });
      if (perpetratorConc.length > 0) datasets.push({ label: perpetrator.name, times: times, values: perpetratorConc, color: "var(--accent-warn)" });
      if (datasets.length > 0) drawLineChart("ddi-chart", datasets);

      var riskContainer = document.getElementById("ddi-risk");
      if (riskContainer) {
        riskContainer.innerHTML = "";
        var flags = data.risk_flags || [];
        if (flags.length > 0) {
          flags.forEach(function (flag) {
            var tag = document.createElement("span");
            tag.className = "risk-tag";
            tag.textContent = flag;
            riskContainer.appendChild(tag);
          });
        } else if (data.risk_category) {
          var ok = document.createElement("span");
          ok.className = "risk-ok";
          ok.textContent = "Risk: " + data.risk_category;
          riskContainer.appendChild(ok);
        }
      }

      showResults(id);
    } catch (err) {
      setSpinner(id, false);
      showError(id, err.message);
    }
  });
})();
