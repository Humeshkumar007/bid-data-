/**
 * Curriculum Effectiveness Analysis - Interactive Web Application Logic
 * Integrates Chart.js visual analytics, live FastAPI prediction simulator,
 * SHAP waterfall explainability, and at-risk student monitoring.
 */

// Global Chart References
let chartBenchmark = null;
let chartOutcomePie = null;
let chartWeightScatter = null;
let chartLateImpact = null;
let chartActiveDaysTier = null;
let chartResourceBreakdown = null;
let chartTemporal = null;


// Application Initialization
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupThemeToggle();
  setupSimulatorControls();
  fetchAllData();

  // Attach refresh button
  const refreshBtn = document.getElementById("refreshDataBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      refreshBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Refreshing...`;
      fetchAllData().finally(() => {
        refreshBtn.innerHTML = `<i class="fa-solid fa-rotate"></i> Refresh`;
      });
    });
  }
});

/* ==========================================================================
   Navigation Tabs Handler
   ========================================================================== */
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panels = document.querySelectorAll(".tab-panel");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPanel = document.getElementById(tab.dataset.tab);
      if (targetPanel) {
        targetPanel.classList.add("active");
        // Trigger resize for Chart.js rendering
        window.dispatchEvent(new Event("resize"));
      }
    });
  });
}

/* ==========================================================================
   Theme Toggle (Dark/Light)
   ========================================================================== */
function setupThemeToggle() {
  const toggleBtn = document.getElementById("themeToggleBtn");
  const body = document.body;
  const savedTheme = localStorage.getItem("app_theme") || "dark";

  if (savedTheme === "light") {
    body.classList.remove("dark-theme");
    body.classList.add("light-theme");
    toggleBtn.innerHTML = `<i class="fa-solid fa-sun"></i>`;
  }

  toggleBtn.addEventListener("click", () => {
    if (body.classList.contains("dark-theme")) {
      body.classList.remove("dark-theme");
      body.classList.add("light-theme");
      toggleBtn.innerHTML = `<i class="fa-solid fa-sun"></i>`;
      localStorage.setItem("app_theme", "light");
    } else {
      body.classList.remove("light-theme");
      body.classList.add("dark-theme");
      toggleBtn.innerHTML = `<i class="fa-solid fa-moon"></i>`;
      localStorage.setItem("app_theme", "dark");
    }
    // Update chart text colors
    Chart.defaults.color = body.classList.contains("dark-theme") ? "#94a3b8" : "#475569";
    Chart.defaults.borderColor = body.classList.contains("dark-theme") ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.08)";
    refreshAllCharts();
  });
}

function refreshAllCharts() {
  [
    chartBenchmark, chartOutcomePie, chartWeightScatter, chartLateImpact,
    chartActiveDaysTier, chartResourceBreakdown, chartTemporal, chartGlobalShap
  ].forEach(chart => {
    if (chart) chart.update();
  });
}

/* ==========================================================================
   Data Fetching & Pipeline Synchronization
   ========================================================================== */
async function fetchAllData() {
  try {
    await Promise.all([
      loadPerformanceData(),
      loadAssessmentsData(),
      loadEngagementData(),
      loadModelEvaluation()
    ]);
  } catch (error) {
    console.error("Error loading curriculum analytics:", error);
  }
}

/* ==========================================================================
   Tab 1: Course Performance
   ========================================================================== */
async function loadPerformanceData() {
  const res = await fetch("/api/curriculum/performance");
  const data = await res.json();
  const kpis = data.overall_kpis;

  // Header quick ribbon
  document.getElementById("hmTotalStudents").textContent = kpis.total_enrollments.toLocaleString();
  document.getElementById("hmActiveCourses").textContent = `${kpis.active_courses_count} (AAA-GGG)`;

  // Tab 1 KPI Cards
  document.getElementById("kpiPassRate").textContent = `${kpis.overall_pass_rate}%`;
  document.getElementById("kpiDistinctionSub").textContent = `${kpis.overall_distinction_rate}% Distinction rate`;
  document.getElementById("kpiWithdrawalRate").textContent = `${kpis.overall_withdrawal_rate}%`;
  document.getElementById("kpiFailSub").textContent = `${kpis.overall_fail_rate}% Academic Failures`;
  document.getElementById("kpiAvgScore").textContent = kpis.avg_assessment_score.toString();
  document.getElementById("kpiAvgClicks").textContent = kpis.avg_vle_clicks.toLocaleString();
  document.getElementById("kpiActiveDaysSub").textContent = `${kpis.avg_active_days} avg active study days`;

  // Render Module Benchmark Bar Chart
  const modules = data.modules;
  renderBenchmarkChart(modules);

  // Render Cohort Academic Result Donut Chart
  renderOutcomePieChart(kpis);

  // Populate Module Table
  const tbody = document.getElementById("moduleTableBody");
  tbody.innerHTML = modules.map(m => `
    <tr>
      <td><strong>${m.code_module}</strong></td>
      <td>${m.enrolled.toLocaleString()}</td>
      <td><span class="badge badge-pass">${m.pass_rate}%</span></td>
      <td><span class="badge badge-distinction">${m.distinction_rate}%</span></td>
      <td><span class="badge badge-withdrawn">${m.withdrawal_rate}%</span></td>
      <td><span class="badge badge-fail">${m.fail_rate}%</span></td>
      <td><strong>${m.avg_score}</strong></td>
      <td>${m.avg_clicks.toLocaleString()}</td>
      <td>${m.assessment_completion}%</td>
    </tr>
  `).join("");
}

function renderBenchmarkChart(modules) {
  const ctx = document.getElementById("courseBenchmarkChart").getContext("2d");
  if (chartBenchmark) chartBenchmark.destroy();

  chartBenchmark = new Chart(ctx, {
    type: "bar",
    data: {
      labels: modules.map(m => `Module ${m.code_module}`),
      datasets: [
        {
          label: "Pass Rate (%)",
          data: modules.map(m => m.pass_rate),
          backgroundColor: "rgba(16, 185, 129, 0.75)",
          borderColor: "#10b981",
          borderWidth: 1.5,
          borderRadius: 6
        },
        {
          label: "Withdrawal Rate (%)",
          data: modules.map(m => m.withdrawal_rate),
          backgroundColor: "rgba(244, 63, 94, 0.75)",
          borderColor: "#f43f5e",
          borderWidth: 1.5,
          borderRadius: 6
        },
        {
          label: "Fail Rate (%)",
          data: modules.map(m => m.fail_rate),
          backgroundColor: "rgba(245, 158, 11, 0.75)",
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { boxWidth: 12, font: { family: "Outfit", size: 11 } } },
        tooltip: { padding: 10 }
      },
      scales: {
        y: { beginAtZero: true, max: 100, title: { display: true, text: "Percentage (%)" } }
      }
    }
  });
}

function renderOutcomePieChart(kpis) {
  const ctx = document.getElementById("outcomePieChart").getContext("2d");
  if (chartOutcomePie) chartOutcomePie.destroy();

  chartOutcomePie = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Pass", "Distinction", "Fail", "Withdrawn"],
      datasets: [{
        data: [
          (kpis.overall_pass_rate - kpis.overall_distinction_rate).toFixed(1),
          kpis.overall_distinction_rate,
          kpis.overall_fail_rate,
          kpis.overall_withdrawal_rate
        ],
        backgroundColor: [
          "#10b981",
          "#a855f7",
          "#f59e0b",
          "#f43f5e"
        ],
        borderColor: "rgba(17, 24, 39, 0.8)",
        borderWidth: 3,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: { position: "right", labels: { boxWidth: 14, font: { family: "Outfit", size: 12 } } }
      }
    }
  });
}

/* ==========================================================================
   Tab 2: Assessment Effectiveness
   ========================================================================== */
async function loadAssessmentsData() {
  const res = await fetch("/api/curriculum/assessments");
  const data = await res.json();

  // Populate Assessment KPIs
  const typeMap = {};
  data.assessment_type_summary.forEach(t => { typeMap[t.assessment_type] = t; });

  if (typeMap["TMA"]) {
    document.getElementById("tmaAvgScore").textContent = `${typeMap["TMA"].avg_score}`;
    document.getElementById("tmaSubmissions").textContent = `${typeMap["TMA"].pass_rate}% Pass Rate &bull; ${typeMap["TMA"].count} Assessments`;
  }
  if (typeMap["CMA"]) {
    document.getElementById("cmaAvgScore").textContent = `${typeMap["CMA"].avg_score}`;
    document.getElementById("cmaSubmissions").textContent = `${typeMap["CMA"].pass_rate}% Pass Rate &bull; ${typeMap["CMA"].count} Assessments`;
  }
  if (typeMap["Exam"]) {
    document.getElementById("examAvgScore").textContent = `${typeMap["Exam"].avg_score}`;
    document.getElementById("examPassRate").textContent = `${typeMap["Exam"].pass_rate}% Pass Rate &bull; Summative Final`;
  }

  const avgLate = (data.assessment_type_summary.reduce((acc, curr) => acc + curr.avg_late_rate, 0) / data.assessment_type_summary.length).toFixed(1);
  document.getElementById("avgLateSubRate").textContent = `${avgLate}%`;

  // Render Weight vs Difficulty Scatter Plot
  renderWeightScatter(data.scatter_data);

  // Render Late Submission Impact Chart
  renderLateImpactChart(data.late_submission_impact);

  // Populate Bottleneck Assessments Table
  const challengingBody = document.getElementById("challengingAssessmentsBody");
  challengingBody.innerHTML = data.challenging_assessments.map(a => `
    <tr>
      <td><code>#${a.id_assessment}</code></td>
      <td><strong>${a.code_module}</strong></td>
      <td>${a.code_presentation}</td>
      <td><span class="badge ${a.assessment_type === 'Exam' ? 'badge-fail' : 'badge-pass'}">${a.assessment_type}</span></td>
      <td><strong>${a.weight}%</strong></td>
      <td>${a.avg_score}</td>
      <td><span class="badge ${a.pass_rate < 70 ? 'badge-fail' : 'badge-pass'}">${a.pass_rate}%</span></td>
      <td>${a.late_submission_rate}%</td>
      <td><span class="badge badge-withdrawn">${a.difficulty_index}</span></td>
      <td><span class="status-pill status-danger"><i class="fa-solid fa-triangle-exclamation"></i> High Attrition Risk</span></td>
    </tr>
  `).join("");
}

function renderWeightScatter(scatterPoints) {
  const ctx = document.getElementById("weightDifficultyScatter").getContext("2d");
  if (chartWeightScatter) chartWeightScatter.destroy();

  const colorByType = { "TMA": "#6366f1", "CMA": "#a855f7", "Exam": "#f43f5e" };

  chartWeightScatter = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: ["TMA", "CMA", "Exam"].map(atype => ({
        label: atype,
        data: scatterPoints.filter(p => p.assessment_type === atype).map(p => ({
          x: p.weight,
          y: p.avg_score,
          title: `Module ${p.code_module} (${p.code_presentation})`,
          passRate: p.pass_rate,
          diffIndex: p.difficulty_index
        })),
        backgroundColor: colorByType[atype] || "#38bdf8",
        pointRadius: 6,
        pointHoverRadius: 9
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw;
              return `${p.title}: Weight ${p.x}%, Avg Score: ${p.y}, Pass Rate: ${p.passRate}%`;
            }
          }
        }
      },
      scales: {
        x: { title: { display: true, text: "Assessment Weight (%)" }, min: 0, max: 105 },
        y: { title: { display: true, text: "Average Score Achieved" }, min: 20, max: 100 }
      }
    }
  });
}

function renderLateImpactChart(lateData) {
  const ctx = document.getElementById("lateSubmissionImpactChart").getContext("2d");
  if (chartLateImpact) chartLateImpact.destroy();

  chartLateImpact = new Chart(ctx, {
    type: "bar",
    data: {
      labels: lateData.map(d => d.late_submission_rate),
      datasets: [
        {
          label: "Pass Rate (%)",
          data: lateData.map(d => d.pass_rate),
          backgroundColor: "#10b981",
          borderRadius: 6
        },
        {
          label: "Average Score",
          data: lateData.map(d => d.avg_score),
          backgroundColor: "#6366f1",
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top" } },
      scales: { y: { beginAtZero: true, max: 100 } }
    }
  });
}

/* ==========================================================================
   Tab 3: Engagement & VLE Dynamics
   ========================================================================== */
async function loadEngagementData() {
  const res = await fetch("/api/curriculum/engagement");
  const data = await res.json();

  // Active days tiers chart
  renderActiveDaysTierChart(data.active_days_tiers);

  // Resource breakdown chart
  renderResourceBreakdownChart(data.activity_breakdown);

  // Temporal milestones chart
  renderTemporalChart(data.outcome_engagement);
}

function renderActiveDaysTierChart(tiers) {
  const ctx = document.getElementById("activeDaysTierChart").getContext("2d");
  if (chartActiveDaysTier) chartActiveDaysTier.destroy();

  chartActiveDaysTier = new Chart(ctx, {
    type: "bar",
    data: {
      labels: tiers.map(t => t.active_days_tier),
      datasets: [
        {
          type: "bar",
          label: "Pass Rate (%)",
          data: tiers.map(t => t.pass_rate),
          backgroundColor: "rgba(99, 102, 241, 0.8)",
          borderRadius: 6,
          yAxisID: "y"
        },
        {
          type: "line",
          label: "Avg Clicks",
          data: tiers.map(t => t.avg_clicks),
          borderColor: "#06b6d4",
          backgroundColor: "#06b6d4",
          pointRadius: 5,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { type: "linear", position: "left", max: 100, title: { display: true, text: "Pass Rate (%)" } },
        y1: { type: "linear", position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Clicks" } }
      }
    }
  });
}

function renderResourceBreakdownChart(breakdown) {
  const ctx = document.getElementById("resourceBreakdownChart").getContext("2d");
  if (chartResourceBreakdown) chartResourceBreakdown.destroy();

  const labels = Object.keys(breakdown);
  const passAverages = labels.map(l => breakdown[l].pass_avg);
  const failAverages = labels.map(l => breakdown[l].fail_avg);

  chartResourceBreakdown = new Chart(ctx, {
    type: "radar",
    data: {
      labels: labels.map(l => l.toUpperCase()),
      datasets: [
        {
          label: "Passing Students",
          data: passAverages,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.2)",
          pointBackgroundColor: "#10b981"
        },
        {
          label: "At-Risk / Failing Students",
          data: failAverages,
          borderColor: "#f43f5e",
          backgroundColor: "rgba(244, 63, 94, 0.2)",
          pointBackgroundColor: "#f43f5e"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          grid: { color: "rgba(255, 255, 255, 0.08)" },
          ticks: { backdropColor: "transparent" }
        }
      }
    }
  });
}

function renderTemporalChart(outcomes) {
  const ctx = document.getElementById("temporalEngagementChart").getContext("2d");
  if (chartTemporal) chartTemporal.destroy();

  chartTemporal = new Chart(ctx, {
    type: "bar",
    data: {
      labels: outcomes.map(o => o.final_result),
      datasets: [
        { label: "Pre-Course", data: outcomes.map(o => o.avg_pre_clicks), backgroundColor: "#06b6d4" },
        { label: "Early Weeks (0-30d)", data: outcomes.map(o => o.avg_early_clicks), backgroundColor: "#6366f1" },
        { label: "Mid Term (31-100d)", data: outcomes.map(o => o.avg_mid_clicks), backgroundColor: "#a855f7" },
        { label: "Late Term (>100d)", data: outcomes.map(o => o.avg_late_clicks), backgroundColor: "#10b981" }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true },
        y: { stacked: true, title: { display: true, text: "Average Clicks Volume" } }
      }
    }
  });
}

/* ==========================================================================
   Tab 4: Machine Learning Benchmark & Live Prediction Simulator
   ========================================================================== */
async function loadModelEvaluation() {
  const res = await fetch("/api/model/evaluation");
  const data = await res.json();

  document.getElementById("hmModelAuc").textContent = (data.models[data.best_model_name].roc_auc).toString();

  const container = document.getElementById("modelsBenchmarkGrid");
  container.innerHTML = Object.entries(data.models).map(([name, m]) => {
    const isBest = name === data.best_model_name;
    return `
      <div class="model-benchmark-card ${isBest ? 'is-best' : ''}">
        <div class="mb-title"><i class="fa-solid fa-microchip"></i> ${name}</div>
        <div class="mb-metric-row"><span>Accuracy</span><span>${(m.accuracy * 100).toFixed(2)}%</span></div>
        <div class="mb-metric-row"><span>ROC-AUC</span><span>${m.roc_auc.toFixed(4)}</span></div>
        <div class="mb-metric-row"><span>F1-Score</span><span>${(m.f1_score * 100).toFixed(2)}%</span></div>
        <div class="mb-metric-row"><span>Precision</span><span>${(m.precision * 100).toFixed(2)}%</span></div>
        <div class="mb-metric-row"><span>Recall</span><span>${(m.recall * 100).toFixed(2)}%</span></div>
        <div class="mb-metric-row"><span>Fit Time</span><span>${m.training_time_seconds}s</span></div>
      </div>
    `;
  }).join("");
}

function setupSimulatorControls() {
  // Range sliders sync
  const sliderMap = [
    { id: "simActiveDays", label: "valActiveDays", unit: " days" },
    { id: "simAvgScore", label: "valAvgScore", unit: "" },
    { id: "simCompletionRate", label: "valCompletionRate", unit: "%" },
    { id: "simLateRate", label: "valLateRate", unit: "%" },
    { id: "simTotalClicks", label: "valTotalClicks", unit: "" }
  ];

  sliderMap.forEach(s => {
    const el = document.getElementById(s.id);
    const label = document.getElementById(s.label);
    if (el && label) {
      el.addEventListener("input", () => {
        label.textContent = el.value + s.unit;
      });
    }
  });

  // Preset Button
  const presetBtn = document.getElementById("loadPresetBtn");
  if (presetBtn) {
    presetBtn.addEventListener("click", () => {
      document.getElementById("simActiveDays").value = 12;
      document.getElementById("valActiveDays").textContent = "12 days";

      document.getElementById("simAvgScore").value = 42.0;
      document.getElementById("valAvgScore").textContent = "42.0";

      document.getElementById("simCompletionRate").value = 35;
      document.getElementById("valCompletionRate").textContent = "35%";

      document.getElementById("simLateRate").value = 60;
      document.getElementById("valLateRate").textContent = "60%";

      document.getElementById("simTotalClicks").value = 280;
      document.getElementById("valTotalClicks").textContent = "280";

      document.getElementById("simStudiedCredits").value = 120;
      document.getElementById("simPrevAttempts").value = 1;

      runPrediction();
    });
  }

  // Form submit
  const form = document.getElementById("predictionForm");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      runPrediction();
    });
  }

  // Initial simulation run
  setTimeout(runPrediction, 500);
}

async function runPrediction() {
  const simBtn = document.getElementById("simulateBtn");
  if (simBtn) simBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Calculating SHAP...`;

  const activeDays = parseInt(document.getElementById("simActiveDays").value, 10);
  const avgScore = parseFloat(document.getElementById("simAvgScore").value);
  const completionRate = parseFloat(document.getElementById("simCompletionRate").value) / 100.0;
  const lateRate = parseFloat(document.getElementById("simLateRate").value) / 100.0;
  const totalClicks = parseInt(document.getElementById("simTotalClicks").value, 10);
  const studiedCredits = parseInt(document.getElementById("simStudiedCredits").value, 10);
  const prevAttempts = parseInt(document.getElementById("simPrevAttempts").value, 10);
  const moduleCode = document.getElementById("simModule").value;
  const education = document.getElementById("simEducation").value;

  const payload = {
    code_module: moduleCode,
    gender: "F",
    highest_education: education,
    age_band: "0-35",
    disability: "N",
    studied_credits: studiedCredits,
    num_of_prev_attempts: prevAttempts,
    days_before_course_registered: 30,
    total_assessments_submitted: Math.round(completionRate * 5),
    avg_assessment_score: avgScore,
    weighted_score: avgScore * completionRate,
    late_submission_rate: lateRate,
    avg_submission_delay: lateRate > 0.3 ? 3.0 : -1.5,
    assessment_completion_rate: completionRate,
    tma_submitted: Math.max(1, Math.round(completionRate * 4)),
    tma_avg_score: avgScore,
    cma_submitted: 1,
    cma_avg_score: avgScore,
    total_vle_clicks: totalClicks,
    vle_active_days: activeDays,
    clicks_per_active_day: activeDays > 0 ? (totalClicks / activeDays) : 0,
    pre_course_clicks: Math.round(totalClicks * 0.08),
    early_course_clicks: Math.round(totalClicks * 0.3),
    click_forum: Math.round(totalClicks * 0.25),
    click_oucontent: Math.round(totalClicks * 0.45),
    click_quiz: Math.round(totalClicks * 0.1),
    click_resource: Math.round(totalClicks * 0.06),
    click_url: Math.round(totalClicks * 0.04)
  };

  try {
    const res = await fetch("/api/model/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    renderPredictionResult(result);
  } catch (err) {
    console.error("Prediction failed:", err);
  } finally {
    if (simBtn) simBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Outcome & Run SHAP`;
  }
}

function renderPredictionResult(res) {
  const pred = res.prediction;
  const isPass = pred.outcome === "Pass";

  // Score gauge
  const passProbEl = document.getElementById("predPassProb");
  passProbEl.textContent = `${pred.probability_pass}%`;
  passProbEl.parentElement.style.borderColor = pred.risk_color;
  passProbEl.parentElement.style.backgroundColor = isPass ? "rgba(16, 185, 129, 0.2)" : "rgba(244, 63, 94, 0.2)";

  // Risk Badge
  const badge = document.getElementById("riskBadge");
  badge.textContent = pred.risk_tier;
  badge.className = `status-pill ${pred.risk_tier === 'Low Risk' ? 'status-healthy' : pred.risk_tier === 'Medium Risk' ? 'status-warning' : 'status-danger'}`;

  // Outcome text
  document.getElementById("predOutcomeTitle").innerHTML = `Predicted Outcome: <span style="color: ${pred.risk_color};">${pred.outcome}</span>`;
  document.getElementById("predOutcomeDesc").textContent = isPass
    ? `Student exhibits robust engagement patterns and is on track for module completion.`
    : `High likelihood of module withdrawal or failure. Early intervention strongly recommended.`;

  // Probability Bar
  document.getElementById("probFillPass").style.width = `${pred.probability_pass}%`;
  document.getElementById("probFillFail").style.width = `${pred.probability_fail}%`;
  document.getElementById("probPassLabel").textContent = `Pass: ${pred.probability_pass}%`;
  document.getElementById("probFailLabel").textContent = `Fail / Risk: ${pred.probability_fail}%`;

  // Render SHAP Waterfall factors
  const factorsList = document.getElementById("shapFactorsList");
  factorsList.innerHTML = res.top_contributing_factors.map(f => `
    <div class="shap-item ${f.direction}">
      <span class="shap-feat-name">${f.feature.replace(/_/g, ' ')}</span>
      <span class="shap-effect-text">${f.effect}</span>
      <span class="shap-val-badge ${f.direction === 'positive' ? 'pos' : 'neg'}">
        ${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(3)}
      </span>
    </div>
  `).join("");

  // Render Interventions
  const interventionsList = document.getElementById("interventionsList");
  interventionsList.innerHTML = res.recommended_interventions.map(int => `
    <div class="intervention-card ${int.priority === 'High' ? 'high-priority' : int.priority === 'Medium' ? 'medium-priority' : ''}">
      <div class="int-title-row">
        <span>${int.type}</span>
        <span class="int-priority-pill ${int.priority === 'High' ? 'p-high' : int.priority === 'Medium' ? 'p-med' : 'p-low'}">${int.priority} Priority</span>
      </div>
      <div class="int-action"><strong>Action:</strong> ${int.action}</div>
      <div class="int-rationale"><strong>Rationale:</strong> ${int.rationale}</div>
    </div>
  `).join("");
}
