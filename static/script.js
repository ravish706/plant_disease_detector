const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const uploadSection = document.querySelector('.upload-section');
const resetBtns = document.querySelectorAll('.reset-btn');


uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

uploadZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadSection.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (!file) return;

    loading.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        resultsSection.style.display = 'block';
        if (data.error) {
            showError(data.error);
        } else {
            showResults(data);
        }
    })
    .catch(error => {
        loading.style.display = 'none';
        resultsSection.style.display = 'block';
        showError('An error occurred during analysis.');
        console.error('Error:', error);
    });
});

resetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        previewSection.style.display = 'none';
        resultsSection.style.display = 'none';
        uploadSection.style.display = 'block';
        fileInput.value = '';
    });
});

function showResults(data) {
    document.getElementById('disease-name').textContent = data.class;
    document.getElementById('confidence-text').textContent = `Confidence: ${data.confidence}%`;
    document.getElementById('confidence-fill').style.width = `${data.confidence}%`;
    
    const recommendation = document.getElementById('recommendation');
    // This is a placeholder. The backend does not provide this information.
    recommendation.innerHTML = `
        <h4>💡 Treatment Recommendation</h4>
        <p><strong>Diagnosis:</strong> ${data.class}</p>
        <p><strong>Action:</strong> Please consult a specialist for treatment options.</p>
    `;

    const resultCard = document.querySelector('.result-card');
    if (data.class.toLowerCase().includes("healthy")) {
        resultCard.style.background = 'linear-gradient(135deg, #f0fdf4, #dcfce7)';
        resultCard.style.borderLeftColor = '#10b981';
        recommendation.style.background = '#f0fdf4';
        recommendation.style.borderLeftColor = '#10b981';
    } else {
        resultCard.style.background = 'linear-gradient(135deg, #fef2f2, #fee2e2)';
        resultCard.style.borderLeftColor = '#ef4444';
        recommendation.style.background = '#fef2f2';
        recommendation.style.borderLeftColor = '#ef4444';
    }
}

function showError(message) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.innerHTML = `<p class="error">${message}</p>`;
}

document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-5px)';
        card.style.transition = 'transform 0.3s ease';
    });
    
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0)';
    });
});