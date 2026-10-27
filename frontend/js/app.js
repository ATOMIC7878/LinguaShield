const ageProfileSelect = document.getElementById('ageProfile');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');
const textPayload = document.getElementById('textPayload');
const filePayload = document.getElementById('filePayload');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const submitTextBtn = document.getElementById('submitTextBtn');
const submitFileBtn = document.getElementById('submitFileBtn');
const resultAnchor = document.getElementById('resultAnchor');
const riskBadge = document.getElementById('riskBadge');
const outputExtractedText = document.getElementById('outputExtractedText');
const emergencySafetyHub = document.getElementById('emergencySafetyHub');
const seniorSummaryBlock = document.getElementById('seniorSummaryBlock');
const seniorInstructionText = document.getElementById('seniorInstructionText');
const technicalMetricsBlock = document.getElementById('technicalMetricsBlock');

const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const urgencyBar = document.getElementById('urgencyBar');
const urgencyValue = document.getElementById('urgencyValue');
const authorityBar = document.getElementById('authorityBar');
const authorityValue = document.getElementById('authorityValue');
const isolationBar = document.getElementById('isolationBar');
const isolationValue = document.getElementById('isolationValue');

const localImagePreviewCard = document.getElementById('localImagePreviewCard');
const livePreviewImg = document.getElementById('livePreviewImg');

const BACKEND_URL = "http://127.0.0.1:8000";

ageProfileSelect.addEventListener('change', (e) => {
    const selectedMode = e.target.value;
    if (selectedMode === 'senior') {
        document.body.classList.add('senior-mode');
        if (!resultAnchor.classList.contains('hidden')) technicalMetricsBlock.classList.add('hidden');
    } else {
        document.body.classList.remove('senior-mode');
        if (!resultAnchor.classList.contains('hidden')) technicalMetricsBlock.classList.remove('hidden');
    }
});

tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.getAttribute('data-target')).classList.add('active');
    });
});

filePayload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];

        fileNameDisplay.textContent = `Target File: ${file.name}`;
        fileNameDisplay.style.backgroundColor = "#D1FAE5";
        fileNameDisplay.style.color = "#065F46";

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function (event) {
                livePreviewImg.src = event.target.result;
                localImagePreviewCard.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            localImagePreviewCard.classList.add('hidden');
            livePreviewImg.src = "";
        }
    }
});

function renderAuditReport(data) {
    resultAnchor.classList.remove('hidden');
    const currentAgeMode = ageProfileSelect.value;
    const isThreat = data.is_scam;

    if (isThreat) {
        riskBadge.textContent = data.risk_classification;
        riskBadge.classList.add('scam-alert');
        emergencySafetyHub.classList.remove('hidden');
    } else {
        riskBadge.textContent = "SAFE / LOW RISK";
        riskBadge.classList.remove('scam-alert');
        emergencySafetyHub.classList.add('hidden');
    }

    // Force explicit text injection from data metrics payload keys
    outputExtractedText.textContent = data.extracted_text || data.processed_text;

    confidenceBar.style.width = `${data.confidence}%`;
    confidenceValue.textContent = `${data.confidence}%`;
    urgencyBar.style.width = `${data.manipulation_vectors.urgency}%`;
    urgencyValue.textContent = `${data.manipulation_vectors.urgency}%`;
    authorityBar.style.width = `${data.manipulation_vectors.authority}%`;
    authorityValue.textContent = `${data.manipulation_vectors.authority}%`;
    isolationBar.style.width = `${data.manipulation_vectors.isolation}%`;
    isolationValue.textContent = `${data.manipulation_vectors.isolation}%`;

    if (currentAgeMode === 'senior') {
        technicalMetricsBlock.classList.add('hidden');
        seniorSummaryBlock.classList.remove('hidden');
        if (isThreat) {
            seniorInstructionText.textContent = "🛑 WARNING: This message looks like a fake scam attempt! It is trying to scare you or pretend to be someone in authority. Do not send any money, do not click any links, and do not call back any numbers listed inside it.";
            seniorSummaryBlock.style.backgroundColor = "#FEF2F2";
            seniorSummaryBlock.style.borderLeftColor = "#DC2626";
            seniorInstructionText.style.color = "#7F1D1D";
        } else {
            seniorInstructionText.textContent = "💚 SAFE: Our system checked the message phrasing patterns and found no common scam signatures. It looks clean and safe to read.";
            seniorSummaryBlock.style.backgroundColor = "#F0FDF4";
            seniorSummaryBlock.style.borderLeftColor = "#10B981";
            seniorInstructionText.style.color = "#065F46";
        }
    } else {
        technicalMetricsBlock.classList.remove('hidden');
        seniorSummaryBlock.classList.add('hidden');
    }
    resultAnchor.scrollIntoView({ behavior: 'smooth' });
}

submitTextBtn.addEventListener('click', async () => {
    const rawText = textPayload.value.trim();
    if (!rawText) return alert("Please enter text content before running the audit.");
    setLoading(submitTextBtn, true, "Auditing...");
    try {
        const response = await fetch(`${BACKEND_URL}/api/analyze/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: rawText, age_group: ageProfileSelect.value })
        });
        if (!response.ok) throw new Error("API Exception.");
        renderAuditReport(await response.json());
    } catch (error) {
        alert("Failed to connect to local backend server.");
    } finally {
        setLoading(submitTextBtn, false, "Run Threat Audit");
    }
});

submitFileBtn.addEventListener('click', async () => {
    if (filePayload.files.length === 0) return alert("Please select an image or audio file first.");
    const formData = new FormData();
    formData.append("file", filePayload.files[0]);
    formData.append("age_group", ageProfileSelect.value);
    setLoading(submitFileBtn, true, "Extracting Asset...");
    try {
        const response = await fetch(`${BACKEND_URL}/api/analyze/file`, { method: 'POST', body: formData });
        if (!response.ok) throw new Error("File integration pipeline error.");
        renderAuditReport(await response.json());
    } catch (error) {
        alert("Multi-modal parser extraction error.");
    } finally {
        setLoading(submitFileBtn, false, "Process Multi-Modal Asset");
    }
});

function setLoading(btn, isLoading, txt) {
    btn.disabled = isLoading;
    btn.textContent = txt;
    btn.style.opacity = isLoading ? "0.6" : "1.0";
}