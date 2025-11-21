const API_BASE = "http://127.0.0.1:8000";

/* ============================================================
   HELPER — Safely get element (avoids errors on other pages)
============================================================== */
function get(id) {
    return document.getElementById(id);
}

/* ============================================================
   FILE UPLOAD
============================================================== */
async function uploadFile() {
    const fileInput = get("fileInput");
    const status = get("uploadStatus");

    if (!fileInput?.files.length) {
        status.innerHTML = "Please select a CSV file.";
        status.style.color = "red";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    status.textContent = "Uploading...";

    const res = await fetch(`${API_BASE}/upload/`, {
        method: "POST",
        body: formData
    });

    status.textContent = res.ok ? "Upload successful!" : "Upload failed!";
    status.style.color = res.ok ? "green" : "red";
}

/* ============================================================
   EDA SUMMARY + PLOTS
============================================================== */
async function loadEDASummary() {
    const box = get("summaryBox");
    if (!box) return;

    box.textContent = "Loading...";

    const res = await fetch(`${API_BASE}/eda/summary`);
    const data = await res.json();

    box.innerHTML = JSON.stringify(data, null, 2);
}

async function loadEDAPlots() {
    const container = get("plotsContainer");
    if (!container) return;

    container.textContent = "Loading plots...";

    const res = await fetch(`${API_BASE}/eda/plots`);
    const data = await res.json();

    container.innerHTML = "";

    for (const [name, base64] of Object.entries(data)) {
        const img = document.createElement("img");
        img.src = "data:image/png;base64," + base64;
        img.classList.add("plot-img");
        container.appendChild(img);
    }
}

/* ============================================================
   PREPROCESS
============================================================== */
async function runPreprocess() {
    const strategy = get("strategySelect")?.value;
    const status = get("preprocessStatus");
    if (!strategy || !status) return;

    status.textContent = "Processing...";

    const res = await fetch(`${API_BASE}/preprocess/missing?strategy=${strategy}`, {
        method: "POST"
    });

    if (!res.ok) {
        status.textContent = "Preprocessing failed!";
        status.style.color = "red";
        return;
    }

    const data = await res.json();
    status.innerHTML = `Preprocessing complete!<br>Saved File: ${data.output_file}`;
    status.style.color = "green";
}

/* ============================================================
   TRAIN MODEL
============================================================== */
async function trainModel() {
    const target = get("targetInput")?.value;
    const testSize = get("testSizeInput")?.value;
    const status = get("trainStatus");

    if (!target || !status) return;

    status.textContent = "Training model...";

    const res = await fetch(
        `${API_BASE}/train/?target=${target}&test_size=${testSize}`,
        { method: "POST" }
    );

    if (!res.ok) {
        status.textContent = "Training failed!";
        status.style.color = "red";
        return;
    }

    const data = await res.json();
    status.innerHTML = `Model trained!<br>R² Score: ${data.r2_score.toFixed(4)}`;
    status.style.color = "green";
}

/* ============================================================
   PREDICTION
============================================================== */
async function predictYield() {
    const status = get("predictionResult");
    if (!status) return;

    const payload = {
        soil_pH: parseFloat(get("soil_pH")?.value),
        soil_N: parseFloat(get("soil_N")?.value),
        soil_P: parseFloat(get("soil_P")?.value),
        rainfall_mm: parseFloat(get("rainfall_mm")?.value),
        temp_avg: parseFloat(get("temp_avg")?.value),
        fertilizer_kg_per_ha: parseFloat(get("fertilizer_kg_per_ha")?.value),
        irrigation_mm: parseFloat(get("irrigation_mm")?.value),
        pesticide_ml: parseFloat(get("pesticide_ml")?.value),
        month: parseInt(get("month")?.value),
        day_of_year: parseInt(get("day_of_year")?.value),
        year: parseInt(get("year")?.value),
        input_cost_total: parseFloat(get("input_cost_total")?.value),
        environmental_score: parseFloat(get("environmental_score")?.value),
        crop_type: get("crop_type")?.value
    };

    status.textContent = "Predicting...";

    const res = await fetch(`${API_BASE}/predict/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!res.ok) {
        status.textContent = "Prediction failed!";
        status.style.color = "red";
        return;
    }

    const data = await res.json();
    status.innerHTML = `Predicted Yield: <b>${data.prediction.toFixed(2)}</b> kg/ha`;
    status.style.color = "green";
}

/* ============================================================
   OPTIMIZATION
============================================================== */
async function optimizeInputs() {
    const box = get("optimizeResult");
    if (!box) return;

    const payload = {
        soil_pH: parseFloat(get("soil_pH")?.value),
        soil_N: parseFloat(get("soil_N")?.value),
        soil_P: parseFloat(get("soil_P")?.value),
        rainfall_mm: parseFloat(get("rainfall_mm")?.value),
        temp_avg: parseFloat(get("temp_avg")?.value),
        month: parseInt(get("month")?.value),
        day_of_year: parseInt(get("day_of_year")?.value),
        year: parseInt(get("year")?.value),
        crop_type: get("crop_type")?.value
    };

    box.textContent = "Optimizing...";

    const res = await fetch(`${API_BASE}/optimize/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!res.ok) {
        box.textContent = "Optimization failed!";
        box.style.color = "red";
        return;
    }

    const data = await res.json();

    box.innerHTML = `
Optimal Inputs:
-------------------------
Fertilizer: ${data.fertilizer_kg_per_ha} kg/ha  
Irrigation: ${data.irrigation_mm} mm  
Pesticide: ${data.pesticide_ml} ml  

Predicted Yield: ${data.yield.toFixed(2)} kg/ha  
Cost: ${data.input_cost_total.toFixed(2)} INR  
Environmental Score: ${data.environmental_score.toFixed(2)}
    `;
}

/* ============================================================
   SHAP DASHBOARD
============================================================== */

/* Load SHAP Summary Plot */
async function loadSummary() {
    const img = get("summary-img");
    if (!img) return;

    img.src = ""; // clear old
    img.alt = "Loading...";

    const res = await fetch(`${API_BASE}/shap/summary`);
    const data = await res.json();

    img.src = "data:image/png;base64," + data.image;
}

/* Load Feature Importance Bar Plot */
async function loadBar() {
    const img = get("bar-img");
    if (!img) return;

    img.src = ""; // clear old
    img.alt = "Loading...";

    const res = await fetch(`${API_BASE}/shap/bar`);
    const data = await res.json();

    img.src = "data:image/png;base64," + data.image;
}
