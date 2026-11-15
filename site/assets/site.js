/* Explainer site behaviour: menu, published-results chart, structure-loss
   weights, test-set selection simulation. No dependencies. */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /* ---------- theme: light by default, remembered per visitor ---------- */
  const toggle = $(".theme-toggle");
  const syncToggle = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    if (toggle) toggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  };
  if (toggle) toggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage may be blocked; the theme still applies */ }
    syncToggle();
    document.dispatchEvent(new CustomEvent("themechange"));
  });
  syncToggle();

  /* ---------- on GitHub Pages: open notes and code in GitHub's viewer ----------
     Locally, links like ../research/decisions.md open the raw file. On github.io the same files
     exist in the published snapshot (gh-pages branch); GitHub renders them far better. */
  if (location.hostname.endsWith(".github.io")) {
    const owner = location.hostname.split(".")[0];
    const repo = location.pathname.split("/")[1];
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a[href]");
      if (!a || a.target === "_blank" || /^[a-z]+:/i.test(a.getAttribute("href"))) return;
      const url = new URL(a.getAttribute("href"), location.href);
      if (!/\.(md|py|ya?ml|txt)$/i.test(url.pathname)) return;
      const path = url.pathname.split("/").slice(2).join("/");
      e.preventDefault();
      location.href = `https://github.com/${owner}/${repo}/blob/gh-pages/${path}`;
    });
  }

  /* ---------- menu: close the popover after choosing a link ---------- */
  const menu = $("#site-menu");
  if (menu) menu.addEventListener("click", (e) => {
    if (e.target.closest("a") && menu.matches(":popover-open")) menu.hidePopover();
  });

  /* ---------- published results chart ---------- */
  const chartEl = $("#published-chart");
  if (chartEl) {
    const METHODS = [
      { name: "U-Net", source: "PraNet paper", color: "--s1", shape: "circle" },
      { name: "PraNet", source: "PraNet paper", color: "--s2", shape: "square" },
      { name: "Polyp-PVT", source: "Polyp-PVT paper", color: "--s3", shape: "diamond" },
    ];
    const ROWS = [
      { key: "Kvasir", label: "Kvasir", group: "Seen hospitals" },
      { key: "CVC-ClinicDB", label: "CVC-ClinicDB", group: "Seen hospitals" },
      { key: "CVC-ColonDB", label: "CVC-ColonDB", group: "Unseen hospitals" },
      { key: "ETIS-LaribPolypDB", label: "ETIS", group: "Unseen hospitals" },
      { key: "CVC-300", label: "CVC-300", group: "Unseen hospitals" },
    ];
    let data = null;
    const tip = document.createElement("div");
    tip.className = "tooltip";
    tip.setAttribute("role", "status");
    chartEl.appendChild(tip);

    const parseCSV = (text) => {
      const [head, ...lines] = text.trim().split(/\r?\n/);
      const cols = head.split(",");
      return lines.map((line) => {
        const cells = line.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map((c) => c.replace(/,$/, "").replace(/^"|"$/g, ""));
        return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
      });
    };

    const shapePath = (shape, x, y, r) => {
      if (shape === "square") return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" rx="1.5"`;
      if (shape === "diamond") { const d = r * 1.3; return `<path d="M${x} ${y - d}L${x + d} ${y}L${x} ${y + d}L${x - d} ${y}Z"`; }
      return `<circle cx="${x}" cy="${y}" r="${r}"`;
    };

    const render = () => {
      if (!data) return;
      const W = Math.max(320, chartEl.clientWidth);
      const narrow = W < 560;
      const labelW = narrow ? 92 : 132, gapW = narrow ? 48 : 96, rowH = narrow ? 40 : 46, groupH = 30;
      const x0 = labelW, x1 = W - gapW, lo = 0.3, hi = 1.0;
      const X = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
      let y = 8, out = [];
      const ticks = narrow ? [0.4, 0.6, 0.8, 1.0] : [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
      const rowsY = [];
      let lastGroup = null;
      ROWS.forEach((r) => {
        if (r.group !== lastGroup) { y += lastGroup ? 14 : 0; out.push(`<text x="0" y="${y + 18}" style="font-size:12px;fill:var(--ink-3);font-weight:500">${r.group}</text>`); y += groupH; lastGroup = r.group; }
        rowsY.push([r, y + rowH / 2]); y += rowH;
      });
      const H = y + 34;
      const grid = ticks.map((t) => `<line x1="${X(t)}" x2="${X(t)}" y1="${8}" y2="${H - 30}" style="stroke:var(--grid);stroke-width:1"/><text x="${X(t)}" y="${H - 10}" text-anchor="middle" style="font-size:12px;fill:var(--muted);font-variant-numeric:tabular-nums">${t.toFixed(1)}</text>`).join("");
      out.unshift(grid);
      out.push(`<text x="${x1 + 12}" y="${26}" style="font-size:12px;fill:var(--ink-3);font-weight:500">${narrow ? "Gap" : "Gap to U-Net"}</text>`);
      rowsY.forEach(([r, cy]) => {
        const vals = METHODS.map((m) => ({ m, v: data[`${m.name}|${m.source}|${r.key}`] })).filter((d) => d.v != null);
        const vs = vals.map((d) => d.v), mn = Math.min(...vs), mx = Math.max(...vs);
        out.push(`<text x="${x0 - 12}" y="${cy + 4}" text-anchor="end" style="font-size:${narrow ? 13 : 14}px;fill:var(--ink)">${r.label}</text>`);
        out.push(`<line x1="${X(mn)}" x2="${X(mx)}" y1="${cy}" y2="${cy}" style="stroke:var(--axis);stroke-width:2;stroke-linecap:round"/>`);
        vals.forEach(({ m, v }) => {
          const label = `${m.name} on ${r.label}: ${v.toFixed(3)} (${m.source})`;
          out.push(`${shapePath(m.shape, X(v), cy, 6)} class="mark" tabindex="0" role="img" aria-label="${label}" data-tip="<strong>${m.name}</strong> · ${r.label}<br>${v.toFixed(3)} mean Dice · ${m.source}" style="fill:var(${m.color});stroke:var(--card);stroke-width:2"/>`);
        });
        const best = vals.reduce((a, b) => (b.v > a.v ? b : a));
        const unet = vals.find((d) => d.m.name === "U-Net").v;
        out.push(`<text x="${x1 + 12}" y="${cy + 4}" style="font-size:13px;fill:var(--ink-2);font-variant-numeric:tabular-nums">+${(best.v - unet).toFixed(2)}${best.m.name === "Polyp-PVT" ? "" : "*"}</text>`);
      });
      out.push(`<line x1="${x0}" x2="${x1}" y1="${H - 30}" y2="${H - 30}" style="stroke:var(--axis);stroke-width:1"/>`);
      const svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true" focusable="false">${out.join("")}</svg>`;
      const old = chartEl.querySelector("svg");
      if (old) old.remove();
      chartEl.insertAdjacentHTML("afterbegin", svg);
      chartEl.querySelector("svg").removeAttribute("aria-hidden");
    };

    const showTip = (mark) => {
      const box = chartEl.getBoundingClientRect(), m = mark.getBoundingClientRect();
      tip.innerHTML = mark.dataset.tip;
      const x = Math.min(Math.max(m.left + m.width / 2 - box.left, 90), box.width - 90);
      tip.style.left = `${x}px`;
      tip.style.top = `${m.top - box.top}px`;
      tip.classList.add("show");
    };
    const hideTip = () => tip.classList.remove("show");
    chartEl.addEventListener("pointerover", (e) => { const m = e.target.closest(".mark"); if (m) showTip(m); });
    chartEl.addEventListener("pointerout", (e) => { if (e.target.closest(".mark")) hideTip(); });
    chartEl.addEventListener("focusin", (e) => { const m = e.target.closest(".mark"); if (m) showTip(m); });
    chartEl.addEventListener("focusout", hideTip);

    const buildTable = (rows) => {
      const wrap = $("#published-table");
      if (!wrap) return;
      const body = rows.map((r) => `<tr><th scope="row">${r.method}</th><td data-label="Test set">${r.dataset}</td><td data-label="Hospital">${r.centre}</td><td class="num" data-label="mDice">${r.mdice}</td><td class="num" data-label="mIoU">${r.miou || "–"}</td><td data-label="Source">${r.reported_in}, ${r.table}</td></tr>`).join("");
      wrap.innerHTML = `<table><caption>Published results used in the chart</caption><thead><tr><th scope="col">Method</th><th scope="col">Test set</th><th scope="col">Hospital</th><th scope="col" class="num">mDice</th><th scope="col" class="num">mIoU</th><th scope="col">Source</th></tr></thead><tbody>${body}</tbody></table>`;
    };

    fetch("../research/published_results.csv", { cache: "no-cache" })
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then((text) => {
        const rows = parseCSV(text).filter((r) => r.protocol === "pranet");
        const wanted = new Set(METHODS.map((m) => `${m.name}|${m.source}`));
        const used = rows.filter((r) => wanted.has(`${r.method}|${r.reported_in}`));
        data = {};
        used.forEach((r) => { data[`${r.method}|${r.reported_in}|${r.dataset}`] = parseFloat(r.mdice); });
        render();
        buildTable(used);
        new ResizeObserver(() => render()).observe(chartEl);
      })
      .catch(() => {
        chartEl.insertAdjacentHTML("afterbegin", `<p class="note">The chart reads <code>research/published_results.csv</code>. Open the site through a local web server from the repository root (see README) to see it.</p>`);
      });
  }

  /* ---------- structure loss weights ---------- */
  const maskC = $("#mask-canvas"), weightC = $("#weight-canvas");
  if (maskC && weightC) {
    const N = maskC.width; // 176 px grid = a 352 px image at half resolution
    const mask = new Float32Array(N * N);
    const blobs = [[0.5, 0.52, 0.27], [0.36, 0.42, 0.17], [0.64, 0.62, 0.16], [0.6, 0.36, 0.12]];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const x = i / N, y = j / N;
      mask[j * N + i] = blobs.some(([cx, cy, r]) => (x - cx) ** 2 + (y - cy) ** 2 < r * r) ? 1 : 0;
    }
    // weight 1 = panel background; more weight = stronger blue (dataviz sequential blue steps)
    const rampLight = ["#f5f5f7", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"];
    const rampDark = ["#2c2c2e", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"];
    const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

    const boxBlur = (src, k) => {
      // integral image, zero padding (like padding=15 in avg_pool2d with count_include_pad=True)
      const S = new Float64Array((N + 1) * (N + 1));
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++)
        S[(j + 1) * (N + 1) + i + 1] = src[j * N + i] + S[j * (N + 1) + i + 1] + S[(j + 1) * (N + 1) + i] - S[j * (N + 1) + i];
      const h = (k - 1) / 2, out = new Float32Array(N * N), area = k * k;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const x0 = Math.max(0, i - h), x1 = Math.min(N, i + h + 1), y0 = Math.max(0, j - h), y1 = Math.min(N, j + h + 1);
        const s = S[y1 * (N + 1) + x1] - S[y0 * (N + 1) + x1] - S[y1 * (N + 1) + x0] + S[y0 * (N + 1) + x0];
        out[j * N + i] = s / area;
      }
      return out;
    };

    const draw = () => {
      const dark = document.documentElement.dataset.theme === "dark";
      const kFull = +$("#k").value, lam = +$("#lam").value;
      $("#k-out").textContent = kFull; $("#lam-out").textContent = lam;
      const k = Math.max(3, Math.round(kFull / 2) | 1);
      const blur = boxBlur(mask, k);
      const bg = hex(dark ? "#2c2c2e" : "#f5f5f7"), fg = hex(css("--s1") || "#2a78d6");
      const ramp = (dark ? rampDark : rampLight).map(hex);
      const mi = maskC.getContext("2d").createImageData(N, N), wi = weightC.getContext("2d").createImageData(N, N);
      for (let p = 0; p < N * N; p++) {
        const c = mask[p] ? fg : bg;
        mi.data.set([...c, 255], p * 4);
        const w = 1 + lam * Math.abs(blur[p] - mask[p]); // PraNet: 1 + 5*|avgpool(mask) - mask|
        const idx = Math.min(ramp.length - 1, Math.max(0, Math.floor(((w - 1) / 5) * (ramp.length - 1) + 0.0001)));
        wi.data.set([...ramp[idx], 255], p * 4);
      }
      maskC.getContext("2d").putImageData(mi, 0, 0);
      weightC.getContext("2d").putImageData(wi, 0, 0);
    };
    ["#k", "#lam"].forEach((id) => $(id).addEventListener("input", draw));
    document.addEventListener("themechange", draw);
    draw();
  }

  /* ---------- test-set selection simulation ---------- */
  const epochsEl = $("#epochs");
  if (epochsEl) {
    const TRUE = 0.8, SD_IMAGE = 0.2, TRIALS = 4000;
    let n = 100;
    const mulberry32 = (a) => () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const gauss = (rnd) => { let u = 0, v = 0; while (u === 0) u = rnd(); v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const update = () => {
      const E = +epochsEl.value;
      $("#epochs-out").textContent = E;
      const rnd = mulberry32(42 + E * 7 + n);
      const sd = SD_IMAGE / Math.sqrt(n);
      let sum = 0;
      for (let t = 0; t < TRIALS; t++) {
        let best = -Infinity;
        for (let e = 0; e < E; e++) best = Math.max(best, TRUE + sd * gauss(rnd));
        sum += best;
      }
      const rep = sum / TRIALS;
      $("#r-true").textContent = TRUE.toFixed(3);
      $("#r-rep").textContent = rep.toFixed(3);
      $("#r-gap").textContent = `${rep - TRUE >= 0 ? "+" : "−"}${Math.abs(rep - TRUE).toFixed(3)}`;
    };
    epochsEl.addEventListener("input", update);
    document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".segmented button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      n = +b.dataset.n;
      update();
    }));
    update();
  }
})();

/* ===================== stage 2 ===================== */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  if (!document.querySelector("#ov-canvas, #th-chart, #overfit-chart")) return;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const fmt = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));

  /* ---------- a small line chart with a crosshair tooltip ---------- */
  function lineChart(el, { series, xLabel, yMin = 0, yMax = 1, xFmt = (x) => x, refLines = [], height = 260 }) {
    const tip = document.createElement("div");
    tip.className = "tooltip";
    const draw = () => {
      const W = Math.max(300, el.clientWidth), H = height;
      const m = { l: 40, r: 16, t: 12, b: 38 };
      const xs = series[0].points.map((p) => p.x);
      const x0 = Math.min(...xs), x1 = Math.max(...xs);
      const X = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
      const Y = (y) => H - m.b - ((y - yMin) / (yMax - yMin)) * (H - m.t - m.b);
      const ticksY = [0, 0.25, 0.5, 0.75, 1].map((f) => yMin + f * (yMax - yMin));
      const nX = W < 520 ? 4 : 6;
      const ticksX = Array.from({ length: nX + 1 }, (_, i) => x0 + (i * (x1 - x0)) / nX);
      let s = "";
      ticksY.forEach((t) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" style="stroke:var(--grid)"/><text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end" style="font-size:11px;fill:var(--muted)">${t.toFixed(2)}</text>`; });
      ticksX.forEach((t) => { s += `<text x="${X(t)}" y="${H - m.b + 18}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${xFmt(t)}</text>`; });
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" style="font-size:12px;fill:var(--ink-2)">${xLabel}</text>`;
      refLines.forEach((r) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" style="stroke:var(--ink-3);stroke-dasharray:4 4"/><text x="${W - m.r}" y="${Y(r.y) - 6}" text-anchor="end" style="font-size:11px;fill:var(--ink-2)">${r.label}</text>`; });
      series.forEach((sr) => {
        const d = sr.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`).join("");
        s += `<path d="${d}" style="fill:none;stroke:var(${sr.color});stroke-width:2;stroke-linejoin:round"/>`;
        (sr.marks || []).forEach((mk) => { s += `<circle cx="${X(mk.x)}" cy="${Y(mk.y)}" r="5" style="fill:var(${sr.color});stroke:var(--card);stroke-width:2"/>`; });
      });
      s += `<line class="xhair" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--ink-3);opacity:0"/>`;
      series.forEach((sr, i) => { s += `<circle class="xdot" data-i="${i}" r="4" style="fill:var(${sr.color});stroke:var(--card);stroke-width:2;opacity:0"/>`; });
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      el.appendChild(tip);
      const svg = el.querySelector("svg");
      const hide = () => { tip.classList.remove("show"); svg.querySelectorAll(".xhair,.xdot").forEach((n) => (n.style.opacity = 0)); };
      svg.addEventListener("pointermove", (e) => {
        const r = svg.getBoundingClientRect();
        const px = ((e.clientX - r.left) / r.width) * W;
        const xv = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
        const idx = xs.reduce((b, x, i) => (Math.abs(x - xv) < Math.abs(xs[b] - xv) ? i : b), 0);
        const xx = X(xs[idx]);
        const line = svg.querySelector(".xhair");
        line.setAttribute("x1", xx); line.setAttribute("x2", xx); line.style.opacity = 1;
        svg.querySelectorAll(".xdot").forEach((c) => {
          const p = series[+c.dataset.i].points[idx];
          c.setAttribute("cx", xx); c.setAttribute("cy", Y(p.y)); c.style.opacity = 1;
        });
        tip.innerHTML = `<strong>${xLabel} ${xFmt(xs[idx])}</strong><br>` + series.map((sr) => `${sr.name}: ${sr.points[idx].y.toFixed(3)}`).join("<br>");
        tip.style.left = `${Math.min(Math.max((xx / W) * r.width, 80), r.width - 80)}px`;
        tip.style.top = `${(Y(Math.max(...series.map((sr) => sr.points[idx].y))) / H) * r.height}px`;
        tip.classList.add("show");
      });
      svg.addEventListener("pointerleave", hide);
    };
    draw();
    new ResizeObserver(draw).observe(el);
    return { redraw: draw, set(newSeries, newRefs) { series = newSeries; refLines = newRefs || refLines; draw(); } };
  }

  /* ---------- numbers from saved results ---------- */
  const setJSON = (key, text) => document.querySelectorAll(`[data-json="${key}"]`).forEach((n) => (n.textContent = text));
  fetch("../reports/stage2_checks.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => {
    const t = d.timing, o = d.overfit;
    setJSON("timing.minutes", Math.round(t.minutes_per_50_epochs));
    setJSON("timing.minutes2", Math.round(t.minutes_per_50_epochs));
    setJSON("timing.speedup", `${t.speedup.toFixed(1)}×`);
    setJSON("timing.peak", `${(t.after.peak_mb / 1024).toFixed(2)} GB`);
    setJSON("timing.same", `${t.before.first_epoch_train_loss.toFixed(5)} both times`);
    setJSON("overfit.dice", fmt(o.best_val.dice));
    setJSON("overfit.biou", fmt(o.best_val.boundary_iou));
    setJSON("overfit.params", `${(o.parameters / 1e6).toFixed(1)}M`);
    const oc = $("#overfit-chart");
    if (oc) lineChart(oc, {
      xLabel: "epoch", xFmt: (x) => Math.round(x),
      series: [
        { name: "Dice", color: "--s1", points: o.curve.map((p) => ({ x: p.epoch, y: p.dice })) },
        { name: "Boundary IoU", color: "--s2", points: o.curve.map((p) => ({ x: p.epoch, y: p.boundary_iou })) },
      ],
      refLines: [{ y: 0.9, label: "0.9 bar" }],
    });
    const tc = $("#timing-chart");
    if (tc) {
      const rows = [["Before", t.before], ["After", t.after]];
      const maxS = t.before.train_seconds_per_epoch + t.before.val_seconds_per_epoch;
      const draw = () => {
        const W = Math.max(300, tc.clientWidth), lab = 64, H = 120, bh = 30;
        const X = (s) => (s / maxS) * (W - lab - 90);
        let s = "";
        rows.forEach(([name, v], i) => {
          const y = 14 + i * 50, tr = v.train_seconds_per_epoch, va = v.val_seconds_per_epoch;
          s += `<text x="0" y="${y + bh / 2 + 4}" style="font-size:13px;fill:var(--ink)">${name}</text>`;
          s += `<rect x="${lab}" y="${y}" width="${X(tr)}" height="${bh}" rx="4" style="fill:var(--s1)"><title>${name}: training ${tr.toFixed(1)} s</title></rect>`;
          s += `<rect x="${lab + X(tr) + 2}" y="${y}" width="${Math.max(X(va) - 2, 2)}" height="${bh}" rx="4" style="fill:var(--s2)"><title>${name}: validation ${va.toFixed(1)} s</title></rect>`;
          s += `<text x="${lab + X(tr + va) + 8}" y="${y + bh / 2 + 4}" style="font-size:12px;fill:var(--ink-2)">${tr.toFixed(0)} s + ${va.toFixed(0)} s</text>`;
        });
        tc.innerHTML = `<ul class="legend"><li><svg aria-hidden="true" style="fill:var(--s1)"><use href="assets/icons.svg?v=6#m-square"/></svg>Training</li><li><svg aria-hidden="true" style="fill:var(--s2)"><use href="assets/icons.svg?v=6#m-circle"/></svg>Validation</li></ul><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      };
      draw();
      new ResizeObserver(draw).observe(tc);
    }
  }).catch(() => {});

  fetch("../reports/data_inventory.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => {
    const el = $("#similarity-table");
    if (!el) return;
    const rows = Object.entries(d.nearest_median).map(([k, v]) => `<tr><th scope="row">${k}</th><td class="num" data-label="Median similarity">${v.toFixed(3)}</td></tr>`).join("");
    el.innerHTML = `<table><caption>Median similarity to the nearest training image (table view of the figure)</caption><thead><tr><th scope="col">Set</th><th scope="col" class="num">Median similarity</th></tr></thead><tbody>${rows}</tbody></table>`;
  }).catch(() => {});

  /* ---------- overlap calculator ---------- */
  const ov = $("#ov-canvas");
  if (ov) {
    const N = 120, D = Math.max(1, Math.round(0.02 * Math.hypot(N, N)));
    const ellipse = (cx, cy, rx, ry) => { const m = new Uint8Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) m[y * N + x] = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1 ? 1 : 0; return m; };
    const truth = ellipse(58, 60, 34, 28);
    const boundary = (m) => { // mask pixels within chessboard distance D of background (outside counts as background)
      const b = new Uint8Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (!m[y * N + x]) continue;
        let near = false;
        for (let dy = -D; dy <= D && !near; dy++) for (let dx = -D; dx <= D; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || xx < 0 || yy >= N || xx >= N || !m[yy * N + xx]) { near = true; break; }
        }
        b[y * N + x] = near ? 1 : 0;
      }
      return b;
    };
    const tb = boundary(truth);
    const update = () => {
      const shift = +$("#ov-shift").value, scale = +$("#ov-scale").value / 100;
      $("#ov-shift-out").textContent = shift; $("#ov-scale-out").textContent = `${Math.round(scale * 100)}%`;
      const pred = ellipse(58 + shift, 60 + shift * 0.4, 34 * scale, 28 * scale), pb = boundary(pred);
      let tp = 0, fp = 0, fn = 0, bi = 0, bu = 0;
      for (let i = 0; i < N * N; i++) { const p = pred[i], t = truth[i]; tp += p & t; fp += p & !t; fn += !p & t; bi += pb[i] & tb[i]; bu += pb[i] | tb[i]; }
      $("#ov-dice").textContent = fmt((2 * tp) / (2 * tp + fp + fn || 1));
      $("#ov-iou").textContent = fmt(tp / (tp + fp + fn || 1));
      $("#ov-biou").textContent = fmt(bu ? bi / bu : 1);
      const dark = document.documentElement.dataset.theme === "dark";
      const bg = hex(dark ? "#2c2c2e" : "#f5f5f7"), s1 = hex(css("--s1")), s2 = hex(css("--s2")), s3 = hex(css("--s3"));
      const a = ov.getContext("2d").createImageData(N, N), b = $("#ov-bcanvas").getContext("2d").createImageData(N, N);
      for (let i = 0; i < N * N; i++) {
        const c = pred[i] && truth[i] ? s1 : truth[i] ? s3 : pred[i] ? s2 : bg;
        a.data.set([...c, 255], i * 4);
        const cb = pb[i] && tb[i] ? s1 : tb[i] ? s3 : pb[i] ? s2 : bg;
        b.data.set([...cb, 255], i * 4);
      }
      ov.getContext("2d").putImageData(a, 0, 0);
      $("#ov-bcanvas").getContext("2d").putImageData(b, 0, 0);
    };
    ["#ov-shift", "#ov-scale"].forEach((id) => $(id).addEventListener("input", update));
    document.addEventListener("themechange", update);
    update();
  }

  /* ---------- threshold sweep (same steps as seg.metrics.pranet_curves) ---------- */
  const th = $("#th-chart");
  if (th) {
    const N = 96, truth = new Uint8Array(N * N), base = new Float32Array(N * N);
    let a = 7 >>> 0;
    const rnd = () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const d = Math.hypot((x - 46) / 26, (y - 50) / 21) - 1; // signed distance-ish
      truth[y * N + x] = d < 0 ? 1 : 0;
      base[y * N + x] = 1 / (1 + Math.exp(d * 9 + (rnd() - 0.5) * 2.2));
    }
    let chart = null;
    const update = () => {
      const conf = +$("#th-conf").value / 100, bias = +$("#th-bias").value / 100;
      $("#th-conf-out").textContent = `${Math.round(conf * 100)}%`; $("#th-bias-out").textContent = `${bias >= 0 ? "+" : "−"}${Math.abs(bias).toFixed(2)}`;
      const prob = base.map((p) => Math.min(1, Math.max(0, 0.5 + conf * (p - 0.5) + bias)));
      let tp = 0, fp = 0, fn = 0;
      for (let i = 0; i < prob.length; i++) { const p = prob[i] >= 0.5, t = truth[i]; tp += p && t; fp += p && !t; fn += !p && t; }
      $("#th-raw").textContent = fmt((2 * tp) / (2 * tp + fp + fn || 1));
      let lo = Infinity, hi = -Infinity; prob.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      const lv = new Uint8Array(prob.length);
      if (hi > lo) prob.forEach((v, i) => (lv[i] = Math.min(255, Math.floor(((v - lo) / (hi - lo)) * 255 + 0.5))));
      const fg = new Float64Array(256), all = new Float64Array(256); let nfg = 0;
      lv.forEach((q, i) => { all[q]++; if (truth[i]) { fg[q]++; nfg++; } });
      const pts = []; let ctp = 0, cp = 0, sum = 0, mx = 0;
      for (let k = 0; k < 256; k++) { const q = 255 - k; ctp += fg[q]; cp += all[q]; const dsc = ctp > 0 ? (2 * ctp) / (nfg + cp) : 0; pts.push({ x: q / 255, y: dsc }); sum += dsc; mx = Math.max(mx, dsc); }
      pts.reverse();
      $("#th-mdice").textContent = fmt(sum / 256); $("#th-max").textContent = fmt(mx);
      const series = [{ name: "Dice", color: "--s1", points: pts }];
      const refs = [{ y: sum / 256, label: `mDice ${fmt(sum / 256)}` }];
      if (!chart) chart = lineChart(th, { series, xLabel: "threshold on the stretched map", xFmt: (x) => x.toFixed(2), refLines: refs, height: 240 });
      else chart.set(series, refs);
    };
    ["#th-conf", "#th-bias"].forEach((id) => $(id).addEventListener("input", update));
    update();
  }
})();

/* ===================== stage 3 ===================== */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  if (!$("#flow, #s3-tolerance")) return;
  const SHORT = { "ETIS-LaribPolypDB": "ETIS" };
  const short = (n) => SHORT[n] || n;
  const sign = (v, d = 4) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;

  /* tooltip shared by the marks of one chart; recreated after every redraw */
  const attachTips = (el) => {
    const tip = document.createElement("div");
    tip.className = "tooltip";
    tip.setAttribute("role", "status");
    el.appendChild(tip);
    const show = (mark) => {
      const box = el.getBoundingClientRect(), m = mark.getBoundingClientRect();
      tip.innerHTML = mark.dataset.tip;
      tip.style.left = `${Math.min(Math.max(m.left + m.width / 2 - box.left, 90), box.width - 90)}px`;
      tip.style.top = `${m.top - box.top}px`;
      tip.classList.add("show");
    };
    const hide = () => tip.classList.remove("show");
    el.onpointerover = (e) => { const m = e.target.closest(".mark"); if (m) show(m); };
    el.onpointerout = (e) => { if (e.target.closest(".mark")) hide(); };
    el.onfocusin = (e) => { const m = e.target.closest(".mark"); if (m) show(m); };
    el.onfocusout = hide;
  };
  const drawInto = (el, W, H, body) => {
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${body}</svg>`;
    attachTips(el);
  };

  /* ---------- which steps run in our code ---------- */
  const flow = $("#flow");
  if (flow) {
    const NOTES = {
      pub: "With the authors' maps, steps 1-5 are already done: the files are their output. Only our evaluation runs, so any difference is our evaluation's fault.",
      run: "With the released weights, every step runs in our code, on a Mac instead of an NVIDIA GPU. Only the trained model comes from the authors.",
    };
    const set = (check) => {
      flow.dataset.check = check;
      flow.querySelectorAll("li").forEach((li) => li.classList.toggle("ours", li.dataset.in.split(" ").includes(check)));
      $("#flow-note").textContent = NOTES[check];
      document.querySelectorAll("[data-check]").forEach((b) => b.tagName === "BUTTON" && b.setAttribute("aria-pressed", String(b.dataset.check === check)));
    };
    document.querySelectorAll("button[data-check]").forEach((b) => b.addEventListener("click", () => set(b.dataset.check)));
    set("pub");
  }

  fetch("data/stage3.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => {
    const tol = d.tolerance;
    const put = (key, text) => document.querySelectorAll(`[data-s3="${key}"]`).forEach((n) => (n.textContent = text));
    const within = d.sets.filter((s) => Math.abs(s.ours - s.paper) <= tol).length;
    const lossless = d.sets.filter((s) => s.dataset !== "Kvasir");
    put("hero.pub", `${d.published_within} of ${d.published_rows}`);
    put("hero.run", `${within} of ${d.sets.length}`);
    put("hero.maps", lossless.reduce((a, s) => a + s.maps_within_one_level, 0).toLocaleString("en"));
    put("hero.ms", `${Math.round(d.sets[0].forward_ms.mps)} ms`);

    /* ---------- distance from the paper, both checks ---------- */
    const tc = $("#s3-tolerance");
    if (tc) {
      const draw = () => {
        const W = Math.max(300, tc.clientWidth), narrow = W < 560;
        const lab = narrow ? 106 : 128, right = 16, rowH = narrow ? 44 : 50, top = 10;
        const lim = 0.008, X = (v) => lab + ((v + lim) / (2 * lim)) * (W - lab - right);
        const H = top + d.sets.length * rowH + 34;
        let s = `<rect x="${X(-tol)}" y="${top}" width="${X(tol) - X(-tol)}" height="${d.sets.length * rowH}" rx="6" style="fill:var(--accent-soft)"/>`;
        const ticks = narrow ? [-0.005, 0, 0.005] : [-0.0075, -0.005, -0.0025, 0, 0.0025, 0.005, 0.0075];
        ticks.forEach((t) => {
          s += `<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${H - 30}" style="stroke:${t === 0 ? "var(--ink-3)" : "var(--grid)"};stroke-width:1"/>`;
          s += `<text x="${X(t)}" y="${H - 12}" text-anchor="middle" style="font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums">${t === 0 ? "paper" : sign(t, narrow ? 3 : 4)}</text>`;
        });
        d.sets.forEach((st, i) => {
          const cy = top + i * rowH + rowH / 2, name = short(st.dataset);
          s += `<text x="${lab - 12}" y="${cy + 4}" text-anchor="end" style="font-size:${narrow ? 13 : 14}px;fill:var(--ink)">${name}</text>`;
          const a = st.authors_maps - st.paper, o = st.ours - st.paper;
          const ya = cy - 8, yo = cy + 8; // two lanes, so identical values stay visible
          s += `<line x1="${X(a)}" x2="${X(o)}" y1="${ya}" y2="${yo}" style="stroke:var(--axis);stroke-width:1.5;stroke-linecap:round"/>`;
          s += `<rect class="mark" tabindex="0" x="${X(a) - 6}" y="${ya - 6}" width="12" height="12" rx="1.5" style="fill:var(--s1);stroke:var(--card);stroke-width:2" aria-label="${name}, authors' maps: ${st.authors_maps.toFixed(4)}" data-tip="<strong>${name}</strong> · authors' maps<br>${st.authors_maps.toFixed(4)} (paper ${st.paper.toFixed(3)}, ${sign(a)})"/>`;
          s += `<circle class="mark" tabindex="0" cx="${X(o)}" cy="${yo}" r="6" style="fill:var(--s2);stroke:var(--card);stroke-width:2" aria-label="${name}, released weights: ${st.ours.toFixed(4)}" data-tip="<strong>${name}</strong> · released weights<br>${st.ours.toFixed(4)} (paper ${st.paper.toFixed(3)}, ${sign(o)})"/>`;
        });
        drawInto(tc, W, H, s);
      };
      draw();
      new ResizeObserver(draw).observe(tc);
      const body = d.sets.map((s) => `<tr><th scope="row">${short(s.dataset)}</th><td class="num" data-label="Images">${s.n_images}</td><td class="num" data-label="Paper">${s.paper.toFixed(3)}</td><td class="num" data-label="Authors' maps">${s.authors_maps.toFixed(4)}</td><td class="num" data-label="Released weights">${s.ours.toFixed(4)}</td><td class="num" data-label="Dice at 0.5">${s.dice.toFixed(4)}</td><td class="num" data-label="Boundary IoU">${s.boundary_iou.toFixed(4)}</td></tr>`).join("");
      $("#s3-table").innerHTML = `<table><caption>PraNet on the five test sets (table view of the chart). mDice unless stated; the last two columns are for the released weights.</caption><thead><tr><th scope="col">Test set</th><th scope="col" class="num">Images</th><th scope="col" class="num">Paper</th><th scope="col" class="num">Authors' maps</th><th scope="col" class="num">Released weights</th><th scope="col" class="num">Dice at 0.5</th><th scope="col" class="num">Boundary IoU</th></tr></thead><tbody>${body}</tbody></table>`;
    }

    /* ---------- Kvasir, per image, with the decoder switch ---------- */
    const kc = $("#s3-kvasir");
    if (kc) {
      const imgs = d.kvasir.images;
      let dec = "pillow";
      const p = d.kvasir.paired;
      $("#kv-ci").textContent = `${sign(p.mean)}, 95% interval ${sign(p.ci[0])} to ${sign(p.ci[1])}`;
      const draw = () => {
        const W = Math.max(300, kc.clientWidth), H = W < 560 ? 280 : 320;
        const m = { l: 52, r: 16, t: 14, b: 40 }, lo = -0.2, hi = 0.15;
        const X = (v) => m.l + v * (W - m.l - m.r);
        const Y = (v) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);
        let s = "";
        [-0.2, -0.1, 0, 0.1].forEach((t) => {
          s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" style="stroke:${t === 0 ? "var(--ink-3)" : "var(--grid)"}"/>`;
          s += `<text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end" style="font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums">${t === 0 ? "0" : sign(t, 1)}</text>`;
        });
        [0, 0.25, 0.5, 0.75, 1].forEach((t) => { s += `<text x="${X(t)}" y="${H - m.b + 18}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${t.toFixed(2)}</text>`; });
        s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" style="font-size:12px;fill:var(--ink-2)">mDice of the authors' map</text>`;
        imgs.forEach((r) => {
          const g = r[dec] - r.authors, y = Y(Math.max(lo, Math.min(hi, g)));
          s += `<circle class="mark kv-dot" tabindex="0" cx="${X(r.authors)}" cy="${y}" r="5" style="fill:var(--s1);fill-opacity:.75;stroke:var(--card);stroke-width:1.5" aria-label="${r.image}: ${sign(g, 3)}" data-tip="<strong>${r.image.slice(0, 12)}…</strong><br>authors ${r.authors.toFixed(3)} · ours ${r[dec].toFixed(3)}<br>difference ${sign(g, 3)}"/>`;
        });
        drawInto(kc, W, H, s);
      };
      const summary = () => {
        const g = imgs.map((r) => r[dec] - r.authors);
        $("#kv-mean").textContent = sign(g.reduce((a, b) => a + b, 0) / g.length);
        $("#kv-big").textContent = `${g.filter((v) => Math.abs(v) > 0.01).length} of ${g.length}`;
        $("#kv-worst").textContent = sign(g.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0), 3);
      };
      draw(); summary();
      new ResizeObserver(draw).observe(kc);
      document.querySelectorAll("button[data-dec]").forEach((b) => b.addEventListener("click", () => {
        dec = b.dataset.dec;
        document.querySelectorAll("button[data-dec]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        draw(); summary();
      }));
    }

    /* ---------- forward pass per device ---------- */
    const sc = $("#s3-speed");
    if (sc) {
      const draw = () => {
        const W = Math.max(300, sc.clientWidth), narrow = W < 560;
        const lab = narrow ? 106 : 128, bh = narrow ? 14 : 16, rowH = bh * 2 + 2 + 18, top = 6;
        const max = 70, X = (v) => lab + (v / max) * (W - lab - 64);
        const H = top + d.sets.length * rowH + 26;
        let s = "";
        [0, 20, 40, 60].forEach((t) => {
          s += `<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${H - 24}" style="stroke:var(--grid)"/><text x="${X(t)}" y="${H - 6}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${t} ms</text>`;
        });
        d.sets.forEach((st, i) => {
          const y = top + i * rowH, name = short(st.dataset);
          s += `<text x="${lab - 12}" y="${y + bh + 5}" text-anchor="end" style="font-size:${narrow ? 13 : 14}px;fill:var(--ink)">${name}</text>`;
          [["mps", "Apple GPU", "--s1", 0], ["cpu", "CPU", "--s2", bh + 2]].forEach(([k, label, col, dy]) => {
            const v = st.forward_ms[k];
            s += `<rect class="mark" tabindex="0" x="${lab}" y="${y + dy}" width="${X(v) - lab}" height="${bh}" rx="4" style="fill:var(${col})" aria-label="${name}, ${label}: ${v.toFixed(1)} ms" data-tip="<strong>${name}</strong> · ${label}<br>${v.toFixed(1)} ms per image"/>`;
            s += `<text x="${X(v) + 8}" y="${y + dy + bh - 3}" style="font-size:12px;fill:var(--ink-2);font-variant-numeric:tabular-nums">${Math.round(v)}</text>`;
          });
        });
        drawInto(sc, W, H, s);
      };
      draw();
      new ResizeObserver(draw).observe(sc);
    }
  }).catch(() => {
    document.querySelectorAll("#s3-tolerance, #s3-kvasir, #s3-speed").forEach((el) => {
      el.innerHTML = `<p class="note">The chart reads <code>site/data/stage3.json</code>. Open the site through a local web server from the repository root (see README) to see it.</p>`;
    });
  });
})();

/* ===================== stage 4 ===================== */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  if (!$("#s4-unet")) return;
  const fmtN = (n) => n.toLocaleString("en");
  const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };
  const tipFor = (el) => {
    const tip = document.createElement("div");
    tip.className = "tooltip";
    tip.setAttribute("role", "status");
    el.appendChild(tip);
    return {
      show(html, x, y) { tip.innerHTML = html; tip.style.left = `${Math.min(Math.max(x, 90), el.clientWidth - 90)}px`; tip.style.top = `${y}px`; tip.classList.add("show"); },
      hide() { tip.classList.remove("show"); },
    };
  };

  fetch("data/stage4.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => {
    const runs = d.runs, put = (k, v) => document.querySelectorAll(`[data-s4="${k}"]`).forEach((n) => (n.textContent = v));
    const curve = runs.flatMap((r) => r.curve);
    put("params", `${Math.round(d.model.params.total / 1e6)} M`);
    put("minutes", `${Math.round(median(curve.map((p) => p.seconds)) / 60)} min`);
    put("runs", `${runs.filter((r) => r.status === "done").length} of ${runs.length}`);
    put("date", new Date(d.generated).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" }));

    /* ---------- the U-Net, drawn from the measured shapes ---------- */
    const unetEl = $("#s4-unet");
    const enc = d.model.encoder, dec = d.model.decoder;
    const infoFor = (side, i, shp) => {
      const [c, s] = [shp[0], shp[1]];
      if (side === "L" && i === 0) return ["Step 1: the picture comes in", `The colonoscopy picture enters at ${s} × ${s} pixels. Small 3 × 3 filters describe every pixel with ${c} numbers: edges, colours, textures.`];
      if (side === "L" && i === enc.length - 1) return [`Step ${i + 1}: the bottom of the U`, `The smallest version: ${s} × ${s} spots, ${fmtN(c)} numbers each. Here the model knows best what is in the picture, but has lost the exact edges.`];
      if (side === "L") return [`Step ${i + 1}: shrink`, `Halved to ${s} × ${s} spots. Each spot now covers a bigger area and is described by ${c} numbers, from simple shapes towards whole polyp parts.`];
      if (i === dec.length - 1) return [`Step ${enc.length + i + 1}: back to full size`, `Doubled to ${s} × ${s} pixels again. A final 1 × 1 filter turns the ${c} numbers of each pixel into one: how sure the model is that the pixel is polyp.`];
      return [`Step ${enc.length + i + 1}: grow`, `Doubled to ${s} × ${s} spots. The coarse "what" from below is mixed with the fine detail arriving over the bridge, ${c} numbers per spot.`];
    };
    const BRIDGE = ["A bridge (skip connection)", "It copies the picture from the shrink side straight across to the grow side, so fine detail such as the polyp's edge is not lost on the way down."];
    let active = null;
    const showInfo = ([t, x]) => { $("#s4-info-title").textContent = t; $("#s4-info-text").textContent = x; };
    const drawUnet = () => {
      const W = Math.max(300, unetEl.clientWidth), narrow = W < 560;
      const hOf = (s) => 16 + (Math.log2(s) - 4) * (narrow ? 8 : 10);
      const wOf = (c) => (narrow ? 22 : 30) + (Math.log2(c) - 6) * (narrow ? 11 : 18);
      const rowH = hOf(256) + (narrow ? 18 : 22), top = 28;
      const xl = W * 0.27, xr = W * 0.73, xm = W / 2;
      const rows = enc.length;
      const H = top + rows * rowH + 18;
      const cy = (i) => top + i * rowH + hOf(256) / 2;
      let s = `<defs><marker id="s4-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" style="fill:var(--ink-3)"/></marker></defs>`;
      s += `<text x="${xl}" y="14" text-anchor="middle" style="font-size:12px;fill:var(--ink-2)">picture in</text>`;
      s += `<text x="${xr}" y="14" text-anchor="middle" style="font-size:12px;fill:var(--ink-2)">outline out</text>`;
      // bridges first, so blocks sit on top
      dec.forEach((b, j) => {
        const i = rows - 2 - j, y = cy(i);
        const bx1 = xl + wOf(enc[i].shape[0]) / 2 + 6, bx2 = xr - wOf(b.shape[0]) / 2 - 6;
        s += `<line x1="${bx1}" x2="${bx2}" y1="${y}" y2="${y}" style="stroke:var(--ink-3);stroke-width:2;stroke-dasharray:6 5"/>`;
        s += `<line class="unet-block" data-bridge="1" tabindex="0" role="button" aria-label="Bridge at step ${i + 1}" x1="${bx1}" x2="${bx2}" y1="${y}" y2="${y}" style="stroke:transparent;stroke-width:18"/>`;
      });
      // arrows between steps
      for (let i = 0; i < rows - 1; i++) {
        const xTo = i === rows - 2 ? xm : xl;
        if (i < rows - 2) s += `<line x1="${xl}" x2="${xl}" y1="${cy(i) + hOf(enc[i].shape[1]) / 2 + 2}" y2="${cy(i + 1) - hOf(enc[i + 1].shape[1]) / 2 - 3}" style="stroke:var(--ink-3);stroke-width:1.5" marker-end="url(#s4-arrow)"/>`;
        else s += `<line x1="${xl}" x2="${xTo - wOf(enc[i + 1].shape[0]) / 2 - 3}" y1="${cy(i) + hOf(enc[i].shape[1]) / 2 + 2}" y2="${cy(i + 1)}" style="stroke:var(--ink-3);stroke-width:1.5" marker-end="url(#s4-arrow)"/>`;
      }
      s += `<line x1="${xm + wOf(enc[rows - 1].shape[0]) / 2 + 3}" x2="${xr}" y1="${cy(rows - 1)}" y2="${cy(rows - 2) + hOf(dec[0].shape[1]) / 2 + 3}" style="stroke:var(--ink-3);stroke-width:1.5" marker-end="url(#s4-arrow)"/>`;
      for (let j = 0; j < dec.length - 1; j++) {
        const i = rows - 2 - j;
        s += `<line x1="${xr}" x2="${xr}" y1="${cy(i) - hOf(dec[j].shape[1]) / 2 - 2}" y2="${cy(i - 1) + hOf(dec[j + 1].shape[1]) / 2 + 3}" style="stroke:var(--ink-3);stroke-width:1.5" marker-end="url(#s4-arrow)"/>`;
      }
      const block = (side, i, shp, x, y, col) => {
        const w = wOf(shp[0]), h = hOf(shp[1]);
        const key = `${side}${i}`;
        const lab = narrow ? `${shp[1]}²` : `${shp[1]}×${shp[1]}×${shp[0]}`;
        return `<rect class="unet-block${active === key ? " active" : ""}" data-key="${key}" tabindex="0" role="button" aria-label="${infoFor(side, i, shp)[0]}" x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}" rx="5" style="fill:var(${col})"/>`
          + (side === "L" && i === rows - 1
            ? `<text x="${x}" y="${y + h / 2 + 14}" text-anchor="middle" style="font-size:11px;fill:var(--ink-2);font-variant-numeric:tabular-nums;pointer-events:none">${lab}</text>`
            : `<text x="${side === "L" ? x - w / 2 - 8 : x + w / 2 + 8}" y="${y + 4}" text-anchor="${side === "L" ? "end" : "start"}" style="font-size:11px;fill:var(--ink-2);font-variant-numeric:tabular-nums;pointer-events:none">${lab}</text>`);
      };
      enc.forEach((b, i) => { s += block("L", i, b.shape, i === rows - 1 ? xm : xl, cy(i), "--s1"); });
      dec.forEach((b, j) => { s += block("R", j, b.shape, xr, cy(rows - 2 - j), "--s2"); });
      unetEl.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
    };
    const pick = (t) => {
      if (!t) return;
      if (t.dataset.bridge) { active = null; showInfo(BRIDGE); drawUnet(); return; }
      const key = t.dataset.key; if (!key) return;
      const side = key[0], i = +key.slice(1);
      active = key;
      showInfo(infoFor(side, i, side === "L" ? enc[i].shape : dec[i].shape));
      unetEl.querySelectorAll(".unet-block").forEach((n) => n.classList.toggle("active", n.dataset.key === key));
    };
    unetEl.addEventListener("pointerover", (e) => pick(e.target.closest(".unet-block")));
    unetEl.addEventListener("click", (e) => pick(e.target.closest(".unet-block")));
    unetEl.addEventListener("focusin", (e) => pick(e.target.closest(".unet-block")));
    unetEl.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(e.target.closest(".unet-block")); } });
    drawUnet();
    new ResizeObserver(drawUnet).observe(unetEl);

    /* ---------- cost per pass ---------- */
    const costEl = $("#s4-cost");
    if (costEl && d.timing) {
      const NAMES = {
        "U-Net 64-1024": ["Paper size (our choice)", "--s1"],
        "U-Net 64-1024, channels-last": ["Paper size, other memory layout", "--s3"],
        "U-Net 64-1024, float16 autocast": ["Paper size, half precision", "--s3"],
        "U-Net 64-1024, bfloat16 autocast": ["Paper size, half precision (bf16)", "--s3"],
        "U-Net 32-512 (nnU-Net widths)": ["Half width (for stage 7)", "--s2"],
        "ResNet-34 U-Net (rung 2, anchor)": ["Stage 5 model, for scale", "--s4"],
      };
      const rows = d.timing.variants.map((v) => ({ label: NAMES[v.variant]?.[0] || v.variant, col: NAMES[v.variant]?.[1] || "--s3", v }));
      const tip = tipFor(costEl);
      const draw = () => {
        const W = Math.max(300, costEl.clientWidth), narrow = W < 560;
        const lab = narrow ? 0 : 230, bh = 22, rowH = narrow ? 52 : 36;
        const max = Math.max(...rows.map((r) => r.v.seconds_per_epoch));
        const X = (v) => lab + (v / max) * (W - lab - 60);
        const H = rows.length * rowH + 4;
        let s = "";
        rows.forEach((r, i) => {
          const y = i * rowH + (narrow ? 20 : 6), v = r.v.seconds_per_epoch;
          s += narrow ? `<text x="0" y="${y - 6}" style="font-size:13px;fill:var(--ink)">${r.label}</text>` : `<text x="${lab - 12}" y="${y + bh / 2 + 4}" text-anchor="end" style="font-size:13px;fill:var(--ink)">${r.label}</text>`;
          s += `<rect class="mark" tabindex="0" x="${lab}" y="${y}" width="${Math.max(2, X(v) - lab)}" height="${bh}" rx="4" style="fill:var(${r.col})" aria-label="${r.label}: ${Math.round(v)} seconds" data-tip="<strong>${r.label}</strong><br>${Math.round(v)} s per pass · ${(r.v.parameters / 1e6).toFixed(1)} M parameters<br>${r.v.gflops_per_image.toFixed(0)} GFLOP per picture · ${((r.v.peak_mb || 0) / 1024).toFixed(1)} GB memory"/>`;
          s += `<text x="${X(v) + 8}" y="${y + bh / 2 + 4}" style="font-size:12px;fill:var(--ink-2);font-variant-numeric:tabular-nums">${Math.round(v)} s</text>`;
        });
        costEl.querySelector("svg")?.remove();
        costEl.insertAdjacentHTML("afterbegin", `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`);
      };
      costEl.addEventListener("pointerover", (e) => { const m = e.target.closest(".mark"); if (m) { const b = m.getBoundingClientRect(), c = costEl.getBoundingClientRect(); tip.show(m.dataset.tip, b.left - c.left + b.width / 2, b.top - c.top); } });
      costEl.addEventListener("pointerout", (e) => { if (e.target.closest(".mark")) tip.hide(); });
      costEl.addEventListener("focusin", (e) => { const m = e.target.closest(".mark"); if (m) { const b = m.getBoundingClientRect(), c = costEl.getBoundingClientRect(); tip.show(m.dataset.tip, b.left - c.left + b.width / 2, b.top - c.top); } });
      costEl.addEventListener("focusout", () => tip.hide());
      draw();
      new ResizeObserver(draw).observe(costEl);
    }

    /* ---------- training curves ---------- */
    const COLORS = ["--s1", "--s2", "--s3", "--s4", "--ink-2"];
    const runLabel = (r) => r.lr ? `learning speed ${Number(r.lr)}` : "best learning speed";
    const curveEl = $("#s4-curve");
    const withData = runs.map((r, i) => ({ ...r, i })).filter((r) => r.curve.length);
    if (curveEl && withData.length) {
      const tip = tipFor(curveEl);
      const draw = () => {
        const W = Math.max(300, curveEl.clientWidth), H = W < 560 ? 240 : 280;
        const m = { l: 40, r: 16, t: 12, b: 38 }, x1 = 50;
        const X = (x) => m.l + ((x - 1) / (x1 - 1)) * (W - m.l - m.r);
        const Y = (y) => H - m.b - y * (H - m.t - m.b);
        let s = "";
        [0, 0.25, 0.5, 0.75, 1].forEach((t) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" style="stroke:var(--grid)"/><text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end" style="font-size:11px;fill:var(--muted)">${t.toFixed(2)}</text>`; });
        [1, 10, 20, 30, 40, 50].forEach((t) => { s += `<text x="${X(t)}" y="${H - m.b + 18}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${t}</text>`; });
        s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" style="font-size:12px;fill:var(--ink-2)">pass</text>`;
        withData.forEach((r) => {
          const pts = r.curve.map((p, k) => `${k ? "L" : "M"}${X(p.epoch).toFixed(1)} ${Y(p.dice).toFixed(1)}`).join("");
          s += `<path d="${pts}" style="fill:none;stroke:var(${COLORS[r.i]});stroke-width:2;stroke-linejoin:round"/>`;
          const best = r.curve.reduce((a, b) => (b.dice > a.dice ? b : a));
          s += `<circle cx="${X(best.epoch)}" cy="${Y(best.dice)}" r="5" style="fill:var(${COLORS[r.i]});stroke:var(--card);stroke-width:2"/>`;
        });
        s += `<line class="xhair" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--ink-3);opacity:0"/>`;
        const legend = withData.length > 1 ? `<ul class="legend">${withData.map((r) => `<li><svg aria-hidden="true" style="fill:var(${COLORS[r.i]})"><use href="assets/icons.svg?v=6#m-circle"/></svg>Run ${r.i + 1}, ${runLabel(r)}</li>`).join("")}</ul>` : "";
        curveEl.querySelector("svg")?.remove();
        curveEl.querySelector(".legend")?.remove();
        curveEl.insertAdjacentHTML("afterbegin", `${legend}<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`);
        const svg = curveEl.querySelector("svg");
        svg.addEventListener("pointermove", (e) => {
          const b = svg.getBoundingClientRect(), px = ((e.clientX - b.left) / b.width) * W;
          const ep = Math.round(1 + ((px - m.l) / (W - m.l - m.r)) * (x1 - 1));
          const lines = withData.map((r) => ({ r, p: r.curve.find((q) => q.epoch === ep) })).filter((o) => o.p);
          if (!lines.length) { tip.hide(); svg.querySelector(".xhair").style.opacity = 0; return; }
          const xx = X(ep), hair = svg.querySelector(".xhair");
          hair.setAttribute("x1", xx); hair.setAttribute("x2", xx); hair.style.opacity = 1;
          tip.show(`<strong>Pass ${ep}</strong><br>${lines.map((o) => `${withData.length > 1 ? `Run ${o.r.i + 1}: ` : ""}Dice ${o.p.dice.toFixed(3)} · ${Math.round(o.p.seconds / 60)} min`).join("<br>")}`, (xx / W) * b.width, (Y(Math.max(...lines.map((o) => o.p.dice))) / H) * b.height);
        });
        svg.addEventListener("pointerleave", () => { tip.hide(); svg.querySelector(".xhair").style.opacity = 0; });
      };
      draw();
      new ResizeObserver(draw).observe(curveEl);
      const cur = withData[withData.length - 1];
      $("#s4-best").textContent = Math.max(...withData.flatMap((r) => r.curve.map((p) => p.dice))).toFixed(3);
      $("#s4-done").textContent = `${cur.curve.length} of ${cur.epochs}`;
      $("#s4-pass").textContent = `${Math.round(median(cur.curve.map((p) => p.seconds)) / 60)} min`;
    }

    /* ---------- the five runs ---------- */
    const STATUS = { done: ["i-done", "Done"], running: ["i-next", "Running"], waiting: ["i-planned", "Waiting"] };
    $("#s4-runs").innerHTML = runs.map((r, i) => {
      const what = i < 3 ? `Tries learning speed ${Number(r.lr)}` : `Best learning speed${r.lr ? ` (${Number(r.lr)})` : ""}, random start ${r.seed}`;
      const why = i < 3 ? "Finds the best learning speed" : "Shows how much luck matters";
      const [icon, label] = STATUS[r.status];
      const pct = (100 * r.curve.length) / r.epochs;
      return `<li data-status="${r.status}"><span class="run-name">Run ${i + 1}. ${what}<small>${why} · ${r.curve.length} of ${r.epochs} passes</small></span><span class="run-status"><svg class="icon" aria-hidden="true"><use href="assets/icons.svg?v=6#${icon}"/></svg>${label}</span><span class="bar" aria-hidden="true"><span style="width:${pct}%"></span></span></li>`;
    }).join("");
  }).catch(() => {
    document.querySelectorAll("#s4-unet, #s4-cost, #s4-curve").forEach((el) => { el.innerHTML = `<p class="note">The chart reads <code>site/data/stage4.json</code>. Open the site through a local web server from the repository root (see README).</p>`; });
  });
})();
