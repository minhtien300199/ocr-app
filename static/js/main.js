let currentImage = null;

// Tab switching
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all tabs
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        // Add active class to clicked tab
        button.classList.add('active');
        document.getElementById(`${button.dataset.tab}-tab`).classList.add('active');
    });
});

// File Upload handling
document.getElementById('imageInput').addEventListener('change', handleFileSelect);

// Drag and Drop handling
const dropZone = document.getElementById('dragDropZone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFileSelect({ target: { files: files } });
    }
});

// Clipboard handling
document.addEventListener('paste', (e) => {
    const clipboardZone = document.getElementById('clipboardZone');
    const items = e.clipboardData.items;

    for (let item of items) {
        if (item.type.startsWith('image/')) {
            const file = item.getAsFile();
            handleFileSelect({ target: { files: [file] } });
            clipboardZone.classList.add('active');
            setTimeout(() => clipboardZone.classList.remove('active'), 300);
            break;
        }
    }
});

// URL handling
function loadImageFromUrl() {
    const urlInput = document.getElementById('imageUrl');
    const url = urlInput.value.trim();

    if (!url) {
        alert('Please enter an image URL');
        return;
    }

    // Show loading state
    const preview = document.getElementById('imagePreview');
    preview.style.display = 'block';
    preview.src = 'loading.gif'; // You might want to add a loading gif

    // Load image
    fetch(url)
        .then(response => response.blob())
        .then(blob => {
            const file = new File([blob], "image.jpg", { type: "image/jpeg" });
            currentImage = file;
            displayPreview(file);
        })
        .catch(error => {
            alert('Error loading image from URL: ' + error.message);
            preview.style.display = 'none';
        });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        currentImage = file;
        displayPreview(file);
    }
}

function displayPreview(file) {
    const preview = document.getElementById('imagePreview');
    const reader = new FileReader();

    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
    };

    reader.readAsDataURL(file);
}

function processImage(modelType) {
    if (!currentImage) {
        alert('Please select or paste an image first');
        return;
    }

    const resultsDiv = document.getElementById('results');
    const formData = new FormData();
    formData.append('file', currentImage);

    // Get selected text level
    const textLevel = document.getElementById('textLevel').value;
    formData.append('text_level', textLevel);

    // Show loading state
    resultsDiv.innerHTML = 'Processing...';

    fetch(`/api/${modelType}`, {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultsDiv.innerHTML = `Error: ${data.error}`;
            } else if (data.results) {
                if (data.results.length === 0) {
                    resultsDiv.innerHTML = 'No results found or feature not implemented yet.';
                } else {
                    let resultHtml = '<ul>';
                    data.results.forEach(result => {
                        resultHtml += `<li>Text: ${result.text} (Confidence: ${(result.confidence * 100).toFixed(2)}%)</li>`;
                    });
                    resultHtml += '</ul>';
                    resultsDiv.innerHTML = resultHtml;
                }
            } else {
                resultsDiv.innerHTML = 'Unexpected response format';
            }
        })
        .catch(error => {
            resultsDiv.innerHTML = `Error: ${error.message}`;
        });
} 