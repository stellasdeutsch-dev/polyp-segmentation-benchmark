/* Journal page: everything is drawn from data/journal.json and the saved reports. No dependencies. */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const KINDS = {
    ours: { label: "Our slip, caught by a check", short: "Ours", color: "--s1" },
    published: { label: "Published code or paper", short: "Published", color: "--s2" },
    library: { label: "Library behaviour", short: "Library", color: "--s3" },
    data: { label: "Property of the data", short: "Data", color: "--s4" },
  };
  const STAGE_PAGES = { 1: "stage-1.html", 2: "stage-2.html", 3: "stage-3.html", 4: "stage-4.html" };

  /* shared tooltip per chart */
  function tooltip(el) {
    const tip = document.createElement("div");
    tip.className = "tooltip";
    el.appendChild(tip);
    const box = () => el.getBoundingClientRect();
    return {
      show(html, target) {
        const b = box(), r = target.getBoundingClientRect();
        tip.innerHTML = html;
        tip.style.left = `${Math.min(Math.max(r.left + r.width / 2 - b.left, 90), b.width - 90)}px`;
        tip.style.top = `${r.top - b.top}px`;
        tip.classList.add("show");
      },
      hide() { tip.classList.remove("show"); },
      bind(svg) {
        svg.addEventListener("pointerover", (e) => { const t = e.target.closest("[data-tip]"); if (t) this.show(t.dataset.tip, t); });
        svg.addEventListener("pointerout", (e) => { if (e.target.closest("[data-tip]")) this.hide(); });
        svg.addEventListener("focusin", (e) => { const t = e.target.closest("[data-tip]"); if (t) this.show(t.dataset.tip, t); });
        svg.addEventListener("focusout", () => this.hide());
      },
    };
  }

  function responsive(el, draw) {
    draw();
    new ResizeObserver(() => draw()).observe(el);
  }

  /* ---------------- KPIs, progress ---------------- */
  function kpis(d) {
    const set = (k, v) => { const n = $(`[data-k="${k}"]`); if (n) n.textContent = v; };
    const total = Object.values(d.problem_totals).reduce((a, b) => a + b, 0);
    set("iterations", d.iterations.length);
    set("tests", d.tests.total);
    set("problems", total);
    set("published", d.problem_totals.published);
    $("#generated").textContent = d.generated;
  }

  function progress(d) {
    const el = $("#progress");
    const started = new Set(d.stages_started);
    const current = Math.max(...d.stages_started);
    el.innerHTML = d.stages.map((s) => {
      const state = s.stage < current ? "done" : s.stage === current ? "current" : "planned";
      const label = { done: "Done", current: "In progress", planned: "Planned" }[state];
      const title = STAGE_PAGES[s.stage] && started.has(s.stage)
        ? `<a href="${STAGE_PAGES[s.stage]}">${s.stage}. ${esc(s.title)}</a>`
        : `<span class="title">${s.stage}. ${esc(s.title)}</span>`;
      return `<li data-state="${state}"><span class="bar" aria-hidden="true"></span>${title}<span class="state">${label}${s.iterations ? ` · ${s.iterations} ${s.iterations === 1 ? "iteration" : "iterations"}` : ""}</span></li>`;
    }).join("");
  }

  /* ---------------- chart: problems by stage and kind ---------------- */
  function chartKinds(d) {
    const el = $("#chart-kinds");
    $("#kind-legend").innerHTML = Object.entries(KINDS).map(([k, v]) =>
      `<li><svg aria-hidden="true" style="fill:var(${v.color})"><use href="assets/icons.svg?v=6#m-square"/></svg>${v.label} (${d.problem_totals[k]})</li>`).join("");
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), lab = 70, rowH = 40, H = d.problems_by_stage.length * (rowH + 14) + 8;
      const max = Math.max(...d.problems_by_stage.map((r) => Object.keys(KINDS).reduce((a, k) => a + r[k], 0)));
      const X = (v) => (v / max) * (W - lab - 40);
      let s = "";
      d.problems_by_stage.forEach((r, i) => {
        const y = 4 + i * (rowH + 14);
        s += `<text x="0" y="${y + rowH / 2 + 4}" style="font-size:13px;fill:var(--ink)">Stage ${r.stage}</text>`;
        let x = lab, total = 0;
        Object.entries(KINDS).forEach(([k, v]) => {
          if (!r[k]) return;
          const w = X(r[k]);
          s += `<rect x="${x}" y="${y}" width="${Math.max(w - 2, 2)}" height="${rowH}" rx="4" style="fill:var(${v.color})" tabindex="0" data-tip="<strong>Stage ${r.stage}</strong><br>${v.short}: ${r[k]}"/>`;
          if (w > 26) s += `<text x="${x + w / 2 - 1}" y="${y + rowH / 2 + 4}" text-anchor="middle" style="font-size:12px;fill:${k === "data" ? "#0b0b0b" : "#fff"};pointer-events:none">${r[k]}</text>`;
          x += w; total += r[k];
        });
        s += `<text x="${x + 8}" y="${y + rowH / 2 + 4}" style="font-size:12px;fill:var(--ink-2)">${total}</text>`;
      });
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
  }

  /* ---------------- chart: tests over iterations ---------------- */
  function chartTests(d) {
    const el = $("#chart-tests");
    responsive(el, () => {
      const pts = d.tests_timeline;
      const W = Math.max(300, el.clientWidth), H = 220, m = { l: 40, r: 20, t: 16, b: 34 };
      const max = Math.ceil(Math.max(...pts.map((p) => p.tests)) / 20) * 20;
      const X = (i) => m.l + (i / Math.max(1, pts.length - 1)) * (W - m.l - m.r);
      const Y = (v) => H - m.b - (v / max) * (H - m.t - m.b);
      let s = "";
      for (let v = 0; v <= max; v += max / 4) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" style="stroke:var(--grid)"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" style="font-size:11px;fill:var(--muted)">${v}</text>`;
      let path = "";
      pts.forEach((p, i) => { path += i ? `H${X(i)}V${Y(p.tests)}` : `M${X(0)} ${Y(p.tests)}`; });
      s += `<path d="${path}" style="fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round"/>`;
      pts.forEach((p, i) => {
        s += `<circle cx="${X(i)}" cy="${Y(p.tests)}" r="5" style="fill:var(--s1);stroke:var(--card);stroke-width:2" tabindex="0" data-tip="<strong>Iteration ${p.id}</strong><br>${p.tests} tests"/>`;
        s += `<text x="${X(i)}" y="${H - m.b + 18}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${p.id}</text>`;
      });
      const last = pts[pts.length - 1];
      s += `<text x="${X(pts.length - 1) - 8}" y="${Y(last.tests) - 12}" text-anchor="end" style="font-size:12px;fill:var(--ink-2)">${last.tests} tests</text>`;
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
  }

  /* ---------------- charts: published maps through our evaluation (3.2) ---------------- */
  const SHORT_SET = (n) => n.replace("ETIS-LaribPolypDB", "ETIS").replace("CVC-", "");
  function chartRuler(d) {
    const pm = d.published_maps, el = $("#chart-ruler");
    if (!pm) { el.closest("figure").hidden = true; return; }
    const rows = pm.rows.filter((r) => r.paper), tol = pm.tolerance, methods = [...new Set(rows.map((r) => r.method))];
    const sets = [...new Set(rows.map((r) => r.dataset))];
    const inRounding = rows.filter((r) => Math.abs(r.diff_mdice) <= 0.0005).length, pass = rows.filter((r) => r.within_tolerance).length;
    $("#ruler-summary").textContent = `${pass} of ${rows.length} within the tolerance, ${inRounding} of ${rows.length} within rounding.`;
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), narrow = W < 560, lab = narrow ? 70 : 96, rowH = 46, top = 8, H = methods.length * rowH + top + 34;
      const lim = 0.006, X = (v) => lab + ((v + lim) / (2 * lim)) * (W - lab - 16);
      let s = `<rect x="${X(-tol)}" y="${top}" width="${X(tol) - X(-tol)}" height="${methods.length * rowH}" style="fill:var(--s1);opacity:.08"/>`;
      s += `<rect x="${X(-0.0005)}" y="${top}" width="${X(0.0005) - X(-0.0005)}" height="${methods.length * rowH}" style="fill:var(--s1);opacity:.16"/>`;
      [-0.005, 0, 0.005].forEach((t) => {
        s += `<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${top + methods.length * rowH}" style="stroke:var(--grid)"/>`;
        s += `<text x="${X(t)}" y="${H - 14}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${t === 0 ? "0" : (t > 0 ? "+" : "−") + Math.abs(t).toFixed(3)}</text>`;
      });
      s += `<text x="${(lab + W) / 2}" y="${H - 1}" text-anchor="middle" style="font-size:11px;fill:var(--ink-2)">our mDice minus the published mDice</text>`;
      methods.forEach((m, i) => {
        const y = top + i * rowH + rowH / 2;
        s += `<text x="${lab - 10}" y="${y + 4}" text-anchor="end" style="font-size:12px;fill:var(--ink)">${m}</text>`;
        rows.filter((r) => r.method === m).forEach((r) => {
          const j = sets.indexOf(r.dataset), cy = y + (j - 2) * 6, out = Math.abs(r.diff_mdice) > 0.0005, cx = X(Math.max(-lim, Math.min(lim, r.diff_mdice)));
          const tip = `<strong>${r.method}, ${r.dataset}</strong><br>ours ${r.mdice.toFixed(4)} · paper ${r.paper.mdice.toFixed(3)}<br>difference ${r.diff_mdice >= 0 ? "+" : ""}${r.diff_mdice.toFixed(4)} · ${r.n_images} images`;
          s += `<circle cx="${cx.toFixed(1)}" cy="${cy}" r="5" style="fill:var(${out ? "--s2" : "--s1"});stroke:var(--card);stroke-width:2"/>`;
          s += `<circle cx="${cx.toFixed(1)}" cy="${cy}" r="11" style="fill:transparent" tabindex="0" data-tip="${esc(tip)}"/>`;
          if (out) {
            const num = `${r.diff_mdice >= 0 ? "+" : ""}${r.diff_mdice.toFixed(4)}`, full = `${SHORT_SET(r.dataset)} ${num}`;
            const text = cx + 12 + full.length * 6.4 < W ? full : num; // to the right of the dot, away from the cluster at 0
            s += `<text x="${cx + 12}" y="${cy + 4}" style="font-size:11px;fill:var(--ink-2)">${text}</text>`;
          }
        });
      });
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
    const f = (v, n) => (v == null ? "–" : v.toFixed(n));
    $("#table-ruler").innerHTML = `<table><caption>Authors' maps scored by our evaluation, against PraNet's Tables 1-2</caption><thead><tr><th scope="col">Method, test set</th><th scope="col" class="num">Images</th><th scope="col" class="num">mDice ours</th><th scope="col" class="num">mDice paper</th><th scope="col" class="num">Difference</th><th scope="col" class="num">Dice at 0.5</th></tr></thead><tbody>${
      rows.map((r) => `<tr><th scope="row">${r.method}, ${r.dataset}</th><td class="num" data-label="Images">${r.n_images}/${r.n_ground_truth}</td><td class="num" data-label="mDice ours">${f(r.mdice, 4)}</td><td class="num" data-label="mDice paper">${f(r.paper.mdice, 3)}</td><td class="num" data-label="Difference">${r.diff_mdice >= 0 ? "+" : ""}${f(r.diff_mdice, 4)}</td><td class="num" data-label="Dice at 0.5">${f(r.dice, 4)}</td></tr>`).join("")}</tbody></table>`;
  }

  function chartSkipped(d) {
    const el = $("#chart-skipped"), pm = d.published_maps;
    const rows = pm ? pm.rows.filter((r) => r.all_files) : [];
    if (!rows.length) { el.closest("figure").hidden = true; return; }
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), lab = W < 560 ? 70 : 96, rowH = 40, top = 6, H = rows.length * rowH + top + 32;
      const vals = rows.flatMap((r) => [r.mdice, r.all_files.mdice, r.paper.mdice]);
      const lo = Math.floor(Math.min(...vals) * 100 - 1) / 100, hi = Math.ceil(Math.max(...vals) * 100 + 1) / 100;
      const X = (v) => lab + ((v - lo) / (hi - lo)) * (W - lab - 20);
      let s = "";
      for (let t = lo; t <= hi + 1e-9; t += 0.02) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${top + rows.length * rowH}" style="stroke:var(--grid)"/><text x="${X(t)}" y="${H - 12}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${t.toFixed(2)}</text>`;
      rows.forEach((r, i) => {
        const y = top + i * rowH + rowH / 2, a = X(r.mdice), b = X(r.all_files.mdice);
        s += `<text x="${lab - 10}" y="${y + 4}" text-anchor="end" style="font-size:12px;fill:var(--ink)">${r.method}</text>`;
        s += `<line x1="${a}" x2="${b}" y1="${y}" y2="${y}" style="stroke:var(--ink-2);stroke-width:2"/>`;
        s += `<circle cx="${a}" cy="${y}" r="6" style="fill:var(--card);stroke:var(--s1);stroke-width:2.5" tabindex="0" data-tip="<strong>${r.method}</strong><br>${r.n_images} images (as published): ${r.mdice.toFixed(4)}<br>paper: ${r.paper.mdice.toFixed(3)}"/>`;
        s += `<circle cx="${b}" cy="${y}" r="6" style="fill:var(--s2);stroke:var(--card);stroke-width:2" tabindex="0" data-tip="<strong>${r.method}</strong><br>all ${r.all_files.n_images} images: ${r.all_files.mdice.toFixed(4)}<br>${(r.all_files.mdice - r.mdice).toFixed(4)} against the published set"/>`;
      });
      s += `<text x="${(lab + W) / 2}" y="${H}" text-anchor="middle" style="font-size:11px;fill:var(--ink-2)">mDice on CVC-ColonDB</text>`;
      el.innerHTML = `<ul class="legend"><li><svg aria-hidden="true" viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" style="fill:var(--card);stroke:var(--s1);stroke-width:2.5"/></svg>359 images, as published</li><li><svg aria-hidden="true" viewBox="0 0 14 14"><circle cx="7" cy="7" r="6" style="fill:var(--s2)"/></svg>All 380 images</li></ul><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelectorAll("svg")[1]);
    });
  }

  function chartMaps(d) {
    const el = $("#chart-maps"), pw = d.pranet_weights;
    if (!pw) { el.closest("figure").hidden = true; return; }
    const run = pw.runs.find((r) => r.device === "mps") || pw.runs[0], cpu = pw.runs.find((r) => r.device === "cpu");
    const gap = pw.device_gap ? Math.max(...pw.device_gap.map((g) => g.max_image_diff)) : null;
    const pass = run.sets.filter((e) => Math.abs(e.diff_paper) <= pw.tolerance).length;
    $("#maps-summary").textContent = `${pass} of ${run.sets.length} test sets within ${pw.tolerance} of the paper. Forward pass ${Math.round(Math.min(...run.sets.map((e) => e.forward_ms)))} ms on the Apple GPU${cpu ? `, ${Math.round(Math.min(...cpu.sets.map((e) => e.forward_ms)))} ms on the CPU` : ""}${gap !== null ? `; the two devices agree within ${gap.toFixed(5)} mDice per image` : ""}.`;
    const parts = [["maps_identical", "--s1", "identical"], ["maps_within_one_level", "--s3", "within one level"], ["maps_differ", "--s2", "differs by more"]];
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), lab = W < 560 ? 70 : 140, rowH = 30, H = run.sets.length * (rowH + 10) + 24;
      const X = (f) => lab + f * (W - lab - 12);
      let s = "";
      run.sets.forEach((e, i) => {
        const y = i * (rowH + 10);
        let acc = 0;
        s += `<text x="${lab - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" style="font-size:12px;fill:var(--ink)">${W < 560 ? SHORT_SET(e.dataset) : e.dataset}</text>`;
        parts.forEach(([k, c, name]) => {
          if (!e[k]) return;
          const f = e[k] / e.n_images, x = X(acc / e.n_images), w = X(f) - lab;
          s += `<rect x="${x + (acc ? 1 : 0)}" y="${y}" width="${Math.max(w - (acc ? 1 : 0), 1)}" height="${rowH}" rx="4" style="fill:var(${c})" tabindex="0" data-tip="<strong>${e.dataset}</strong><br>${e[k]} of ${e.n_images} maps ${name}<br>mDice ${e.mdice.toFixed(4)} · authors' maps ${e.authors_maps_mdice.toFixed(4)} · paper ${e.paper.toFixed(3)}"/>`;
          acc += e[k];
        });
      });
      [0, 0.5, 1].forEach((t) => { s += `<text x="${X(t)}" y="${H - 4}" text-anchor="${t === 0 ? "start" : t === 1 ? "end" : "middle"}" style="font-size:11px;fill:var(--muted)">${t * 100}%</text>`; });
      el.innerHTML = `<ul class="legend">${parts.map(([, c, name]) => `<li><svg aria-hidden="true" style="fill:var(${c})"><use href="assets/icons.svg?v=6#m-square"/></svg>${name[0].toUpperCase() + name.slice(1)}</li>`).join("")}</ul><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelectorAll("svg")[el.querySelectorAll("svg").length - 1]);
    });
  }

  /* ---------------- chart: CVC-ClinicDB sequences ---------------- */
  function chartSequences(inv) {
    const el = $("#chart-sequences"), seq = inv.clinicdb_sequences;
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), H = 240, m = { l: 34, r: 8, t: 10, b: 30 };
      const max = Math.ceil(Math.max(...seq.map((q) => q.train + q.val + q.test)) / 5) * 5;
      const bw = (W - m.l - m.r) / seq.length;
      const Y = (v) => H - m.b - (v / max) * (H - m.t - m.b);
      let s = "";
      for (let v = 0; v <= max; v += 10) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" style="stroke:var(--grid)"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end" style="font-size:11px;fill:var(--muted)">${v}</text>`;
      seq.forEach((q, i) => {
        const x = m.l + i * bw + 1.5, w = Math.max(bw - 3, 2);
        let base = 0;
        [["train", "--s1", "training"], ["val", "--s3", "our validation"], ["test", "--s2", "benchmark test"]].forEach(([k, c, name]) => {
          if (!q[k]) return;
          const y1 = Y(base + q[k]), y0 = Y(base);
          s += `<rect x="${x}" y="${y1}" width="${w}" height="${Math.max(y0 - y1 - 1.5, 1)}" rx="2" style="fill:var(${c})" tabindex="0" data-tip="<strong>Video ${q.sequence}</strong> (from frame ${q.first_frame})<br>${q[k]} frames in ${name}"/>`;
          base += q[k];
        });
        if (bw > 16 || i % 2 === 0) s += `<text x="${x + w / 2}" y="${H - m.b + 16}" text-anchor="middle" style="font-size:10px;fill:var(--muted)">${q.sequence}</text>`;
      });
      s += `<text x="${(m.l + W) / 2}" y="${H - 2}" text-anchor="middle" style="font-size:11px;fill:var(--ink-2)">video sequence</text>`;
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
    $("#table-sequences").innerHTML = `<table><caption>CVC-ClinicDB frames per video sequence</caption><thead><tr><th scope="col">Video</th><th scope="col" class="num">Training</th><th scope="col" class="num">Our validation</th><th scope="col" class="num">Benchmark test</th></tr></thead><tbody>${
      seq.map((q) => `<tr><th scope="row">${q.sequence}</th><td class="num" data-label="Training">${q.train}</td><td class="num" data-label="Our validation">${q.val}</td><td class="num" data-label="Benchmark test">${q.test}</td></tr>`).join("")}</tbody></table>`;
  }

  /* ---------------- chart: nearest-training similarity, every image ---------------- */
  function chartNearest(inv) {
    const el = $("#chart-nearest"), groups = Object.entries(inv.nearest);
    $("#nearest-count").textContent = groups.reduce((a, [, v]) => a + v.length, 0).toLocaleString("en");
    let seed = 1;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const jitter = groups.map(([, v]) => v.map(() => rnd() - 0.5));
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), narrow = W < 560;
      const lab = narrow ? 112 : 190, rowH = narrow ? 34 : 38, H = groups.length * rowH + 36;
      const lo = 0.6, hi = 1.0, X = (v) => lab + ((Math.max(v, lo) - lo) / (hi - lo)) * (W - lab - 12);
      let s = "";
      [0.6, 0.7, 0.8, 0.9, 1.0].forEach((t) => { s += `<line x1="${X(t)}" x2="${X(t)}" y1="0" y2="${H - 26}" style="stroke:var(--grid)"/><text x="${X(t)}" y="${H - 8}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${t.toFixed(1)}</text>`; });
      groups.forEach(([name, vals], i) => {
        const y = 10 + i * rowH + rowH / 2;
        const leak = name === "Test, CVC-ClinicDB";
        const shortName = narrow ? name.replace("Validation", "Val").replace("ETIS-LaribPolypDB", "ETIS").replace("CVC-", "") : name.replace("ETIS-LaribPolypDB", "ETIS");
        s += `<text x="${lab - 10}" y="${y + 4}" text-anchor="end" style="font-size:12px;fill:var(--ink)">${shortName}</text>`;
        vals.forEach((v, j) => { s += `<circle cx="${X(v).toFixed(1)}" cy="${(y + jitter[i][j] * rowH * 0.55).toFixed(1)}" r="2.6" style="fill:var(${leak ? "--s2" : "--s1"});opacity:.55"/>`; });
        const sorted = [...vals].sort((a, b) => a - b), med = sorted[Math.floor(sorted.length / 2)];
        s += `<line x1="${X(med)}" x2="${X(med)}" y1="${y - rowH * 0.36}" y2="${y + rowH * 0.36}" style="stroke:var(--ink);stroke-width:2.5"/>`;
        s += `<rect x="${lab}" y="${y - rowH / 2}" width="${W - lab - 12}" height="${rowH}" style="fill:transparent" tabindex="0" data-tip="<strong>${name}</strong><br>${vals.length} images · median ${med.toFixed(3)} · max ${sorted[sorted.length - 1].toFixed(3)}"/>`;
      });
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
  }

  /* ---------------- chart: set sizes ---------------- */
  function chartSizes(inv) {
    const el = $("#chart-sizes"), c = inv.counts;
    const rows = [
      ["Training", c.split.train, "--s1", "Kvasir + CVC-ClinicDB"],
      ["Our validation", c.split.val, "--s3", "90 Kvasir + 68 CVC-ClinicDB"],
      ...Object.entries(c.test).map(([k, v]) => [`Test, ${k.replace("ETIS-LaribPolypDB", "ETIS")}`, v, ["CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"].includes(k) ? "--s2" : "--s4",
        ["CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"].includes(k) ? "unseen hospital" : "seen hospital"]),
    ];
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), lab = W < 560 ? 116 : 150, rowH = 30, H = rows.length * (rowH + 8);
      const max = Math.max(...rows.map((r) => r[1])), X = (v) => (v / max) * (W - lab - 56);
      let s = "";
      rows.forEach(([name, v, color, note], i) => {
        const y = i * (rowH + 8);
        const label = W < 560 ? name.replace("Test, CVC-", "Test, ") : name;
        s += `<text x="${lab - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" style="font-size:12px;fill:var(--ink)">${label}</text>`;
        s += `<rect x="${lab}" y="${y}" width="${Math.max(X(v), 2)}" height="${rowH}" rx="4" style="fill:var(${color})" tabindex="0" data-tip="<strong>${name}</strong><br>${v.toLocaleString("en")} images · ${note}"/>`;
        s += `<text x="${lab + X(v) + 8}" y="${y + rowH / 2 + 4}" style="font-size:12px;fill:var(--ink-2)">${v.toLocaleString("en")}</text>`;
      });
      el.innerHTML = `<ul class="legend"><li><svg aria-hidden="true" style="fill:var(--s1)"><use href="assets/icons.svg?v=6#m-square"/></svg>Training</li><li><svg aria-hidden="true" style="fill:var(--s3)"><use href="assets/icons.svg?v=6#m-square"/></svg>Our validation</li><li><svg aria-hidden="true" style="fill:var(--s4)"><use href="assets/icons.svg?v=6#m-square"/></svg>Test, seen hospital</li><li><svg aria-hidden="true" style="fill:var(--s2)"><use href="assets/icons.svg?v=6#m-square"/></svg>Test, unseen hospital</li></ul><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
  }

  /* ---------------- model diagrams ---------------- */
  const fmtShape = (sh) => sh.join(" × ");
  const fmtM = (n) => `${(n / 1e6).toFixed(1)}M`;
  function modelGraph(el, m) {
    responsive(el, () => {
      const blocks = [
        { name: "input", shape: m.input, part: "io" },
        ...m.encoder.map((b) => ({ ...b, part: "enc" })),
        ...m.decoder.map((b) => ({ ...b, part: "dec" })),
        { name: "output", shape: m.output, part: "io" },
      ];
      const W = Math.max(300, el.clientWidth), narrow = W < 560, H = narrow ? 170 : 210, base = H - 26;
      const maxC = Math.max(...blocks.map((b) => b.shape[0])), maxR = Math.max(...blocks.map((b) => b.shape[1]));
      const hOf = (c) => 14 + (Math.log2(c + 1) / Math.log2(maxC + 1)) * (base - 34);
      const wRaw = blocks.map((b) => 6 + (Math.log2(b.shape[1]) / Math.log2(maxR)) * 26);
      const gap = narrow ? 5 : 10;
      const scale = (W - 8 - gap * (blocks.length - 1)) / wRaw.reduce((a, b) => a + b, 0);
      let x = 4, s = "";
      const pos = [];
      blocks.forEach((b, i) => {
        const w = wRaw[i] * scale, h = hOf(b.shape[0]);
        const color = b.part === "enc" ? "--s1" : b.part === "dec" ? "--s2" : "--ink-3";
        const dashed = m.status === "planned" ? "stroke-dasharray:4 3;" : "";
        s += `<rect class="block-rect" x="${x}" y="${base - h}" width="${w}" height="${h}" rx="5" style="fill:var(${color});fill-opacity:${b.part === "io" ? 0.25 : 0.85};stroke:var(${color});${dashed}" tabindex="0" data-tip="<strong>${esc(b.name)}</strong><br>${fmtShape(b.shape)}"/>`;
        if (!narrow || i % 2 === 0 || i === blocks.length - 1) s += `<text x="${x + w / 2}" y="${base + 16}" text-anchor="middle" style="font-size:10px;fill:var(--muted)">${b.shape[1]}</text>`;
        pos.push({ x, w, h, b });
        x += w + gap;
      });
      // skip connections for U-Nets: encoder level k to the decoder block with the same resolution
      if (m.id !== "pranet" && m.id !== "dinov3_linear") {
        pos.forEach((p) => {
          if (p.b.part !== "enc") return;
          const q = pos.find((d) => d.b.part === "dec" && d.b.shape[1] === p.b.shape[1]);
          if (!q) return;
          const x1 = p.x + p.w / 2, x2 = q.x + q.w / 2, top = base - Math.max(p.h, q.h) - 10 - (x2 - x1) * 0.06;
          s += `<path d="M${x1} ${base - p.h - 2} C ${x1} ${top}, ${x2} ${top}, ${x2} ${base - q.h - 2}" style="fill:none;stroke:var(--ink-3);stroke-dasharray:3 3"/>`;
        });
      }
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
  }

  function models(d) {
    const wrap = $("#models");
    wrap.innerHTML = d.models.map((m, i) => `
      <article class="panel model reveal">
        <div class="model-head"><h3>${esc(m.title)}</h3><span class="badge ${m.status === "built" ? "built" : ""}">${m.rung === "reference" ? "Reference" : `Rung ${m.rung}`} · ${m.status === "built" ? "built and measured" : "planned"}</span></div>
        <ul class="chips">
          <li><strong>${fmtShape(m.input)}</strong>input</li>
          <li><strong>${fmtShape(m.output)}</strong>output</li>
          ${m.params ? `<li><strong>${fmtM(m.params.total)}</strong>parameters</li><li><strong>${Math.round((100 * m.params.encoder) / m.params.total)}%</strong>in the encoder</li>` : `<li><strong>not yet measured</strong>parameters</li>`}
        </ul>
        <div class="graph" id="graph-${i}" role="img" aria-label="Diagram of ${esc(m.title)}"></div>
        <p class="note" style="margin-top:4px;font-size:0.8125rem">Numbers under the blocks: resolution in pixels per side. Hover or tap a block for channels × height × width.</p>
        <p class="note" style="margin-top:8px">${esc(m.note)}</p>
      </article>`).join("");
    d.models.forEach((m, i) => modelGraph($(`#graph-${i}`), m));

    const built = d.models.filter((m) => m.params);
    const el = $("#chart-params");
    responsive(el, () => {
      const W = Math.max(300, el.clientWidth), lab = W < 560 ? 124 : 200, rowH = 32, H = built.length * (rowH + 12);
      const max = Math.max(...built.map((m) => m.params.total)), X = (v) => (v / max) * (W - lab - 64);
      let s = "";
      built.forEach((m, i) => {
        const y = i * (rowH + 12), enc = m.params.encoder, rest = m.params.total - enc;
        const short = W < 560 ? (m.id === "smp_unet_r34" ? "U-Net, ResNet-34" : m.title.replace(" (released weights)", "")) : m.title.replace(" (released weights)", "");
        s += `<text x="${lab - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" style="font-size:12px;fill:var(--ink)">${esc(short)}</text>`;
        s += `<rect x="${lab}" y="${y}" width="${X(enc) - 2}" height="${rowH}" rx="4" style="fill:var(--s1)" tabindex="0" data-tip="<strong>${esc(m.title)}</strong><br>encoder ${fmtM(enc)}"/>`;
        s += `<rect x="${lab + X(enc)}" y="${y}" width="${Math.max(X(rest), 2)}" height="${rowH}" rx="4" style="fill:var(--s2)" tabindex="0" data-tip="<strong>${esc(m.title)}</strong><br>decoder and head ${fmtM(rest)}"/>`;
        s += `<text x="${lab + X(m.params.total) + 8}" y="${y + rowH / 2 + 4}" style="font-size:12px;fill:var(--ink-2)">${fmtM(m.params.total)}</text>`;
      });
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${s}</svg>`;
      tooltip(el).bind(el.querySelector("svg"));
    });
  }

  /* ---------------- timeline and problems table ---------------- */
  function kindTag(k) { return `<span class="kind-tag"><i style="background:var(${KINDS[k].color})"></i>${KINDS[k].short}</span>`; }

  function timeline(d) {
    const el = $("#timeline");
    el.innerHTML = d.iterations.map((it) => {
      const counts = Object.keys(KINDS).map((k) => [k, it.problems.filter((p) => p.kind === k).length]).filter(([, n]) => n);
      return `<li class="iter reveal" data-stage="${it.stage}" data-kinds="${[...new Set(it.problems.map((p) => p.kind))].join(" ")}">
        <div class="iter-head">
          <span class="iter-id">${esc(it.id)}</span>
          <div><h3>${esc(it.title)}</h3><div class="meta">Stage ${it.stage} · ${it.date}${it.tests ? ` · ${it.tests} tests` : ""}</div></div>
          <span class="count"><svg class="icon" aria-hidden="true"><use href="assets/icons.svg?v=6#i-bug"/></svg>${it.problems.length} found and fixed</span>
        </div>
        <p class="summary">${esc(it.summary)}</p>
        <ul class="chips">${it.results.map((r) => `<li><strong>${esc(r.value)}</strong>${esc(r.label)}</li>`).join("")}</ul>
        <details>
          <summary>Why, problems and fixes<svg class="icon" aria-hidden="true"><use href="assets/icons.svg?v=6#i-plus"/></svg></summary>
          <p class="why"><b>Why.</b> ${esc(it.why)}</p>
          <ul class="problems">${it.problems.map((p) => `<li class="problem" data-kind="${p.kind}"><span class="dot" style="background:var(${KINDS[p.kind].color})"></span><div><p class="what">${esc(p.what)} ${kindTag(p.kind)}</p><p class="how"><b>Found:</b> ${esc(p.found)}</p><p class="how"><b>Fix:</b> ${esc(p.fix)}</p></div></li>`).join("")}</ul>
          <ul class="files">${it.files.map((f) => `<li><a class="doc-link" href="../${esc(f)}"><svg class="icon" aria-hidden="true"><use href="assets/icons.svg?v=6#i-doc"/></svg>${esc(f)}</a></li>`).join("")}</ul>
        </details>
      </li>`;
    }).join("");

    const stages = [...new Set(d.iterations.map((it) => it.stage))];
    const stageEl = $("#stage-filter"), kindEl = $("#kind-filter");
    stageEl.innerHTML = `<button type="button" data-stage="all" aria-pressed="true">All stages</button>` + stages.map((s) => `<button type="button" data-stage="${s}" aria-pressed="false">Stage ${s}</button>`).join("");
    kindEl.innerHTML = `<button type="button" data-kind="all" aria-pressed="true">All problems</button>` + Object.entries(KINDS).map(([k, v]) => `<button type="button" data-kind="${k}" aria-pressed="false">${v.short}</button>`).join("");
    let fs = "all", fk = "all";
    const apply = () => {
      el.querySelectorAll(".iter").forEach((li) => {
        const okStage = fs === "all" || li.dataset.stage === fs;
        const okKind = fk === "all" || li.dataset.kinds.split(" ").includes(fk);
        li.hidden = !(okStage && okKind);
        li.querySelectorAll(".problem").forEach((p) => { p.hidden = fk !== "all" && p.dataset.kind !== fk; });
        if (fk !== "all" && okStage && okKind) li.querySelector("details").open = true;
      });
    };
    stageEl.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; fs = b.dataset.stage; stageEl.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); apply(); });
    kindEl.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; fk = b.dataset.kind; kindEl.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); apply(); });
  }

  function problemsTable(d) {
    const rows = d.iterations.flatMap((it) => it.problems.map((p) => ({ it, p })));
    $("#problems-table").innerHTML = `<table><caption>${rows.length} problems, found and fixed</caption><thead><tr><th scope="col">Step</th><th scope="col">Source</th><th scope="col">What</th><th scope="col">How it was found</th><th scope="col">Fix</th></tr></thead><tbody>${
      rows.map(({ it, p }) => `<tr><th scope="row">${esc(it.id)}</th><td data-label="Source">${kindTag(p.kind)}</td><td data-label="What">${esc(p.what)}</td><td data-label="How it was found">${esc(p.found)}</td><td data-label="Fix">${esc(p.fix)}</td></tr>`).join("")}</tbody></table>`;
  }

  /* ---------------- load ---------------- */
  Promise.all([
    fetch("data/journal.json", { cache: "no-cache" }).then((r) => r.json()),
    fetch("../reports/data_inventory.json", { cache: "no-cache" }).then((r) => r.json()),
  ]).then(([d, inv]) => {
    kpis(d); progress(d); chartKinds(d); chartTests(d); chartRuler(d); chartSkipped(d); chartMaps(d); chartSequences(inv); chartNearest(inv); chartSizes(inv);
    models(d); timeline(d); problemsTable(d);
  }).catch((err) => {
    $("#main").insertAdjacentHTML("afterbegin", `<p class="wrap note" style="padding-top:24px">The journal reads data/journal.json. Serve the repository root (python3 -m http.server) and open /site/journal.html. (${esc(err.message)})</p>`);
  });
})();
