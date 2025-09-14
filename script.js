// Mentra Reality Pipeline Demo Integration
document.addEventListener('DOMContentLoaded', function() {
    // Simple scroll effect for navigation
    const nav = document.querySelector('.nav');
    
    window.addEventListener('scroll', () => {
        const currentScrollY = window.scrollY;
        
        if (currentScrollY > 100) {
            nav.style.background = 'rgba(26, 26, 26, 0.98)';
        } else {
            nav.style.background = 'rgba(26, 26, 26, 0.95)';
        }
    });

    // Simple fade-in animation for content sections
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe content sections for animation
    const contentSections = document.querySelectorAll('.content-section, .demo-section');
    contentSections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        observer.observe(section);
    });

    // Simple CTA button interaction
    const ctaButtons = document.querySelectorAll('.cta-button');
    ctaButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            // Simple click effect
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 150);
        });
    });

    // Subtle floating animation for device mockup
    const deviceMockup = document.querySelector('.device-mockup');
    if (deviceMockup) {
        setInterval(() => {
            deviceMockup.style.transform = `translateY(${Math.sin(Date.now() * 0.001) * 5}px)`;
        }, 16);
    }

    // Demo functionality
    initializeDemo();

    // Simple loading animation
    window.addEventListener('load', () => {
        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity 0.6s ease';
        
        setTimeout(() => {
            document.body.style.opacity = '1';
        }, 100);
    });
});

// Demo Integration
function initializeDemo() {
    const uploadArea = document.getElementById('uploadArea');
    const videoInput = document.getElementById('videoInput');
    const demoVideo = document.getElementById('demoVideo');
    const emotionCanvas = document.getElementById('emotionCanvas');
    const processingOverlay = document.getElementById('processingOverlay');
    const startDemoBtn = document.getElementById('startDemo');
    const processVideoBtn = document.getElementById('processVideo');
    const stopDemoBtn = document.getElementById('stopDemo');
    const demoResults = document.getElementById('demoResults');
    
    let currentVideo = null;
    let isProcessing = false;
    let emotionData = [];
    let roomData = [];
    let trajectoryData = [];

    // Upload area click handler
    uploadArea.addEventListener('click', () => {
        videoInput.click();
    });

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#00d4ff';
        uploadArea.style.background = 'rgba(0, 212, 255, 0.1)';
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        uploadArea.style.background = 'rgba(255, 255, 255, 0.03)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        uploadArea.style.background = 'rgba(255, 255, 255, 0.03)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            handleVideoFile(files[0]);
        }
    });

    // Video input handler
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoFile(e.target.files[0]);
        }
    });

    // Button handlers
    startDemoBtn.addEventListener('click', startDemo);
    processVideoBtn.addEventListener('click', processVideo);
    stopDemoBtn.addEventListener('click', stopDemo);

    function handleVideoFile(file) {
        const url = URL.createObjectURL(file);
        demoVideo.src = url;
        demoVideo.style.display = 'block';
        processVideoBtn.style.display = 'inline-block';
        startDemoBtn.style.display = 'none';
        
        // Update upload area
        uploadArea.innerHTML = `
            <div class="upload-icon">✅</div>
            <p>Video loaded: ${file.name}</p>
        `;
    }

    function startDemo() {
        // Simulate starting the pipeline
        processingOverlay.style.display = 'flex';
        startDemoBtn.style.display = 'none';
        processVideoBtn.style.display = 'inline-block';
        
        // Simulate pipeline initialization
        setTimeout(() => {
            processingOverlay.style.display = 'none';
            showNotification('Pipeline initialized successfully!');
        }, 2000);
    }

    function processVideo() {
        if (isProcessing) return;
        
        isProcessing = true;
        processingOverlay.style.display = 'flex';
        processVideoBtn.disabled = true;
        stopDemoBtn.style.display = 'inline-block';
        
        // Try to use real pipeline, fallback to simulation
        processVideoWithPipeline();
    }
    
    async function processVideoWithPipeline() {
        try {
            // Check if backend is available
            const response = await fetch('/api/demo_mode');
            const data = await response.json();
            
            if (data.pipeline_available) {
                // Use real pipeline
                await processWithRealPipeline();
            } else {
                // Use demo mode
                simulateVideoProcessing();
            }
        } catch (error) {
            console.log('Backend not available, using simulation');
            simulateVideoProcessing();
        }
    }
    
    async function processWithRealPipeline() {
        const formData = new FormData();
        const videoFile = videoInput.files[0];
        
        if (!videoFile) {
            showNotification('No video file selected');
            return;
        }
        
        formData.append('video', videoFile);
        
        try {
            const response = await fetch('/api/process_video', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Process real results
                processRealResults(result.results);
            } else {
                throw new Error(result.error || 'Processing failed');
            }
        } catch (error) {
            console.error('Pipeline processing error:', error);
            showNotification('Pipeline error, using simulation');
            simulateVideoProcessing();
        }
    }
    
    function processRealResults(results) {
        // Process real pipeline results
        emotionData = results.emotions || [];
        roomData = results.rooms || [];
        trajectoryData = results.trajectory || [];
        
        // Update UI with real data
        updateEmotionChart();
        updateRoomMap();
        updateTrajectoryViz();
        
        // Show results
        showResults();
        showNotification('Video processing completed with real pipeline!');
        
        // Reset processing state
        isProcessing = false;
        processingOverlay.style.display = 'none';
        processVideoBtn.disabled = false;
        stopDemoBtn.style.display = 'none';
        startDemoBtn.style.display = 'inline-block';
    }

    function stopDemo() {
        isProcessing = false;
        processingOverlay.style.display = 'none';
        processVideoBtn.disabled = false;
        stopDemoBtn.style.display = 'none';
        startDemoBtn.style.display = 'inline-block';
        
        // Reset video
        demoVideo.pause();
        demoVideo.currentTime = 0;
    }

    function simulateVideoProcessing() {
        const duration = demoVideo.duration || 10; // Default 10 seconds if no duration
        const processingTime = Math.min(duration * 1000, 10000); // Max 10 seconds
        
        // Simulate real-time processing
        let progress = 0;
        const interval = setInterval(() => {
            if (!isProcessing) {
                clearInterval(interval);
                return;
            }
            
            progress += 100;
            const percent = Math.min((progress / processingTime) * 100, 100);
            
            // Update processing text
            const processingText = processingOverlay.querySelector('p');
            processingText.textContent = `Processing video... ${Math.round(percent)}%`;
            
            // Simulate emotion detection
            if (progress % 1000 === 0) {
                simulateEmotionDetection();
            }
            
            // Simulate room mapping
            if (progress % 1500 === 0) {
                simulateRoomMapping();
            }
            
            // Simulate trajectory tracking
            if (progress % 800 === 0) {
                simulateTrajectoryTracking();
            }
            
            if (percent >= 100) {
                clearInterval(interval);
                finishProcessing();
            }
        }, 100);
    }

    function simulateEmotionDetection() {
        const emotions = ['Happy', 'Neutral', 'Sad', 'Surprised', 'Angry', 'Fearful', 'Disgusted'];
        const emotion = emotions[Math.floor(Math.random() * emotions.length)];
        const confidence = Math.random() * 0.4 + 0.6; // 60-100% confidence
        
        emotionData.push({
            emotion: emotion,
            confidence: confidence,
            timestamp: Date.now()
        });
        
        // Draw emotion overlay on canvas
        drawEmotionOverlay(emotion, confidence);
    }

    function simulateRoomMapping() {
        const rooms = ['Living Room', 'Kitchen', 'Bedroom', 'Bathroom', 'Office'];
        const room = rooms[Math.floor(Math.random() * rooms.length)];
        const emotion = emotionData[emotionData.length - 1]?.emotion || 'Neutral';
        
        roomData.push({
            room: room,
            emotion: emotion,
            timestamp: Date.now()
        });
    }

    function simulateTrajectoryTracking() {
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const z = Math.random() * 10;
        
        trajectoryData.push({
            x: x,
            y: y,
            z: z,
            timestamp: Date.now()
        });
    }

    function drawEmotionOverlay(emotion, confidence) {
        const canvas = emotionCanvas;
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match video
        canvas.width = demoVideo.videoWidth || 640;
        canvas.height = demoVideo.videoHeight || 480;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw emotion indicator
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        // Draw circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, 50, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(0, 212, 255, ${confidence * 0.8})`;
        ctx.fill();
        
        // Draw emotion text
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(emotion, centerX, centerY + 5);
        
        // Draw confidence
        ctx.font = '12px Inter';
        ctx.fillText(`${Math.round(confidence * 100)}%`, centerX, centerY + 25);
        
        canvas.style.display = 'block';
    }

    function finishProcessing() {
        isProcessing = false;
        processingOverlay.style.display = 'none';
        processVideoBtn.disabled = false;
        stopDemoBtn.style.display = 'none';
        startDemoBtn.style.display = 'inline-block';
        
        // Show results
        showResults();
        showNotification('Video processing completed!');
    }

    function showResults() {
        demoResults.style.display = 'block';
        
        // Update spatial heatmap
        updateSpatialHeatmap();
        
        // Update room analysis
        updateRoomAnalysis();
        
        // Update real estate echo
        updateRealEstateEcho();
        
        // Scroll to results
        demoResults.scrollIntoView({ behavior: 'smooth' });
    }

    function updateSpatialHeatmap() {
        const heatmap = document.getElementById('spatialHeatmap');
        heatmap.innerHTML = 'Spatial Intensity';
    }

    function updateRoomAnalysis() {
        const analysis = document.getElementById('roomAnalysis');
        
        const roomData = [
            { name: 'Living Room', score: 78, area: 320, accessibility: 'Wheelchair Accessible' },
            { name: 'Kitchen', score: 92, area: 180, accessibility: 'Limited Access' },
            { name: 'Bedroom', score: 64, area: 226, accessibility: 'Wheelchair Accessible' },
            { name: 'Bathroom', score: 81, area: 95, accessibility: 'Limited Access' }
        ];
        
        let analysisHTML = '';
        
        roomData.forEach(room => {
            analysisHTML += `
                <div style="margin-bottom: 1rem; padding: 0.75rem; border: 1px solid rgba(255,255,255,0.2); border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: bold; color: #ffffff; font-size: 0.875rem;">${room.name}</span>
                        <span style="color: #00d4ff; font-size: 0.875rem; font-weight: bold;">${room.score}%</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #888888; margin-bottom: 0.5rem;">
                        ${room.area} sq ft • ${room.accessibility}
                    </div>
                    <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;">
                        <div style="height: 100%; width: ${room.score}%; background: #00d4ff; border-radius: 2px;"></div>
                    </div>
                </div>
            `;
        });
        
        analysis.innerHTML = analysisHTML;
    }

    function updateRealEstateEcho() {
        const echo = document.getElementById('realEstateEcho');
        
        const echoData = {
            totalArea: 1200,
            accessibilityScore: 85,
            spatialEfficiency: 92,
            naturalLight: 78,
            flowScore: 88
        };
        
        let echoHTML = `
            <div style="font-size: 2rem; font-weight: bold; color: #00d4ff; margin-bottom: 1rem;">
                ${echoData.totalArea} sq ft
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #00d4ff;">
                        ${echoData.accessibilityScore}%
                    </div>
                    <div style="font-size: 0.75rem; color: #ffffff; margin-top: 0.25rem;">
                        Accessibility
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #00d4ff;">
                        ${echoData.spatialEfficiency}%
                    </div>
                    <div style="font-size: 0.75rem; color: #ffffff; margin-top: 0.25rem;">
                        Spatial Efficiency
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #00d4ff;">
                        ${echoData.naturalLight}%
                    </div>
                    <div style="font-size: 0.75rem; color: #ffffff; margin-top: 0.25rem;">
                        Natural Light
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #00d4ff;">
                        ${echoData.flowScore}%
                    </div>
                    <div style="font-size: 0.75rem; color: #ffffff; margin-top: 0.25rem;">
                        Flow Score
                    </div>
                </div>
            </div>
        `;
        
        echo.innerHTML = echoHTML;
    }

    function showNotification(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: rgba(0, 212, 255, 0.9);
            color: #000;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
