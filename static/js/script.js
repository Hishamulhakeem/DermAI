// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const fileDropArea = document.getElementById('file-drop-area');
    const imagePreview = document.getElementById('image-preview');
    const selectedFileName = document.getElementById('selected-file-name');
    const predictButton = document.getElementById('predict-button');
    const loadingSpinner = document.getElementById('loading-spinner');
    const buttonText = document.getElementById('button-text');

    const resultCard = document.getElementById('result-card');
    const bestDiseaseName = document.getElementById('best-disease-name');
    const top3List = document.getElementById('top3-list');
    
    const hospitalCard = document.getElementById('hospital-card');
    const locationStatus = document.getElementById('location-status');
    const mapContainer = document.getElementById('map-container');
    
    const darkModeToggle = document.getElementById('dark-mode-toggle');


    // --- Dark Mode Toggle ---
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDarkMode = document.body.classList.contains('dark-mode');
        // Store user preference
        localStorage.setItem('dermAI-dark-mode', isDarkMode ? 'enabled' : 'disabled');
        darkModeToggle.querySelector('.icon').textContent = isDarkMode ? '☀️' : '🌙';
    });

    // Apply stored theme on load
    if (localStorage.getItem('dermAI-dark-mode') === 'enabled') {
        document.body.classList.add('dark-mode');
        darkModeToggle.querySelector('.icon').textContent = '☀️';
    }


    // --- Image Upload and Preview Logic ---

    // Click handler for the drop area
    fileDropArea.addEventListener('click', () => imageUpload.click());

    // Drag-and-drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, () => fileDropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, () => fileDropArea.classList.remove('highlight'), false);
    });

    fileDropArea.addEventListener('drop', handleDrop, false);

    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        imageUpload.files = files; // Assign files to the input element
        handleFiles(files);
    }

    // Input change handler
    imageUpload.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length === 0) {
            imagePreview.style.display = 'none';
            predictButton.disabled = true;
            selectedFileName.textContent = '';
            return;
        }

        const file = files[0];
        const fileName = file.name;
        
        // Basic file type check
        if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
            alert('Invalid file type. Please upload a PNG, JPG, or JPEG image.');
            imageUpload.value = ''; // Clear the input
            imagePreview.style.display = 'none';
            predictButton.disabled = true;
            selectedFileName.textContent = '';
            return;
        }
        
        // Show file name
        selectedFileName.textContent = `Selected: ${fileName}`;
        
        // Show image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);

        predictButton.disabled = false;
        resultCard.classList.add('hidden'); // Hide previous results
    }


    // --- Form Submission / AJAX Prediction ---

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image first.');
            return;
        }

        // 1. Show Loading State
        predictButton.disabled = true;
        buttonText.textContent = 'Analyzing...';
        loadingSpinner.style.display = 'inline-block';
        resultCard.classList.add('hidden');
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            // 2. Handle Response
            if (response.ok && result.status === 'success') {
                displayPredictionResults(result);
            } else {
                alert(`Prediction Failed: ${result.error || 'Unknown Error'}`);
            }

        } catch (error) {
            console.error('Prediction network error:', error);
            alert('A network error occurred while connecting to the server.');

        } finally {
            // 3. Reset Loading State
            predictButton.disabled = false;
            buttonText.textContent = 'Analyze Skin';
            loadingSpinner.style.display = 'none';
        }
    });

    function displayPredictionResults(data) {
        // Best Prediction: Only show the disease name
        bestDiseaseName.textContent = data.best_disease_name;
        
        // Top 3 List: Only show disease names
        top3List.innerHTML = '';
        data.top_3_predictions.forEach((p, index) => {
            const listItem = document.createElement('li');
            // Add a visual cue for the top prediction
            const prefix = index === 0 ? '🥇 ' : (index === 1 ? '🥈 ' : '🥉 ');
            // The confidence-value span is omitted completely
            listItem.innerHTML = `
                <span class="disease-name">${prefix}${p.disease}</span>
            `;
            top3List.appendChild(listItem);
        });

        resultCard.classList.remove('hidden');
        // Scroll to the results for better UX on mobile
        resultCard.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Geolocation and Map Embedding (No Changes) ---

    function getNearbyHospitals(lat, lon) {
        hospitalCard.classList.remove('hidden');
        locationStatus.innerHTML = `<p>Location found: **Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}**</p>`;

        // --- OpenStreetMap Alternative (No API Key Required) ---
        const osmUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${lon - 0.01},${lat - 0.005},${lon + 0.01},${lat + 0.005}&layer=mapnik&marker=${lat},${lon}`;
        
        const mapIframe = document.createElement('iframe');
        mapIframe.src = osmUrl;
        mapIframe.allowfullscreen = '';
        mapIframe.loading = 'lazy';
        mapIframe.referrerpolicy = 'no-referrer-when-downgrade';

        mapContainer.innerHTML = ''; // Clear previous map
        mapContainer.appendChild(mapIframe);
    }

    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    getNearbyHospitals(lat, lon);
                },
                (error) => {
                    // Handle errors like permission denied
                    hospitalCard.classList.remove('hidden');
                    let message = 'Location access denied.';
                    if (error.code === error.PERMISSION_DENIED) {
                        message = 'Location permission was denied. Please allow location access to see nearby hospitals.';
                    } else if (error.code === error.POSITION_UNAVAILABLE) {
                        message = 'Location information is unavailable.';
                    } else if (error.code === error.TIMEOUT) {
                        message = 'The request to get user location timed out.';
                    }
                    locationStatus.innerHTML = `<p class="error">${message}</p>`;
                    mapContainer.innerHTML = '<p class="note">Map not available without location.</p>';
                    console.error('Geolocation Error:', error);
                }
            );
        } else {
            hospitalCard.classList.remove('hidden');
            locationStatus.innerHTML = '<p class="error">Geolocation is not supported by this browser.</p>';
            mapContainer.innerHTML = '<p class="note">Map not available.</p>';
        }
    }

    // Initialize the location feature when the page loads
    getLocation();

});