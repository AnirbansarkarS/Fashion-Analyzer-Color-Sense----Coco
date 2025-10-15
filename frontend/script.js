document.addEventListener('DOMContentLoaded', () => {
    const rateButton = document.getElementById('rate-button');
    const resultsSection = document.getElementById('results-section');
    const spinner = document.getElementById('spinner');
    const resultsContent = document.getElementById('results-content');
    const dropZone = document.getElementById('drop-zone');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const fileInput = document.getElementById('file-input');
    const uploadedImage = document.getElementById('uploaded-image');

    let imageUploaded = false;
    let uploadedFile = null; // Store the uploaded file for backend use

    // Handle image upload
    function handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];
        if (!file.type.startsWith('image/')) return;

        uploadedFile = file; // Save file for upload later

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.src = e.target.result;
            uploadedImage.classList.remove('hidden');
            uploadedImage.classList.add('visible');
            dropZoneContent.classList.add('hidden');
            imageUploaded = true;
        };
        reader.readAsDataURL(file);
    }

    // Connect with backend on "Rate" click
    rateButton.addEventListener('click', async () => {
        if (!imageUploaded || !uploadedFile) {
            alert('Please upload an image first!');
            return;
        }

        resultsSection.classList.remove('hidden');
        spinner.classList.remove('hidden');
        resultsContent.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', uploadedFile);

            const response = await fetch('http://127.0.0.1:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            spinner.classList.add('hidden');
            resultsContent.classList.remove('hidden');

            // --- Update UI with backend data ---
            const scoreCircle = resultsContent.querySelector('.score-circle');
            const recommendationsBox = resultsContent.querySelector('.recommendations-box');

            if (scoreCircle) {
                scoreCircle.textContent = data.score ? `${data.score}/10` : "--";
                scoreCircle.style.animation = 'fadeIn 0.8s ease-out forwards';
            }

            if (recommendationsBox) {
                recommendationsBox.innerHTML = "";
                if (data.recommendations && data.recommendations.length > 0) {
                    data.recommendations.forEach(rec => {
                        const p = document.createElement('p');
                        p.textContent = `• ${rec}`;
                        recommendationsBox.appendChild(p);
                    });
                } else {
                    recommendationsBox.textContent = "No recommendations found.";
                }
                recommendationsBox.style.animation = 'slideUp 0.8s ease-out forwards';
            }

        } catch (error) {
            spinner.classList.add('hidden');
            alert('⚠ Error connecting to backend or analyzing image.');
            console.error(error);
        }
    });

    // --- Drag and Drop logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }, false);

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
    });
});
document.addEventListener("mousemove", (e) => {
    const x = (e.clientX / window.innerWidth) * 100;
    const y = (e.clientY / window.innerHeight) * 100;
    document.body.style.background = `linear-gradient(${x + 100}deg, #d4e0ff, #f8e8ff, #c9f7f5)`;
});
const dropZone = document.getElementById("drop-zone");
const uploadedImage = document.getElementById("uploaded-image");

dropZone.addEventListener("click", () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.click();

    fileInput.onchange = () => {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
                uploadedImage.classList.add("visible");
            };
            reader.readAsDataURL(file);
        }
    };
});
