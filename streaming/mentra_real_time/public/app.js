// Global state
let isStreaming = false;
let streamStartTime = null;
let durationInterval = null;
let eventSource = null;
let hlsPlayer = null;

// DOM Elements
const elements = {
    connectionStatus: document.querySelector('.connection-status'),
    statusText: document.querySelector('.status-text'),
    frameCounter: document.querySelector('.frame-counter'),
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    fullscreenBtn: document.getElementById('fullscreenBtn'),
    streamStatus: document.getElementById('streamStatus'),
    streamId: document.getElementById('streamId'),
    streamFormat: document.getElementById('streamFormat'),
    streamDuration: document.getElementById('streamDuration'),
    urlSection: document.getElementById('urlSection'),
    hlsUrl: document.getElementById('hlsUrl'),
    dashUrl: document.getElementById('dashUrl'),
    hlsUrlContainer: document.getElementById('hlsUrlContainer'),
    dashUrlContainer: document.getElementById('dashUrlContainer'),
    logsContainer: document.getElementById('logsContainer'),
    streamFrame: document.getElementById('streamFrame'),
    hlsVideo: document.getElementById('hlsVideo'),
    placeholder: document.getElementById('placeholder'),
    captureStep: document.getElementById('captureStep'),
    streamStep: document.getElementById('streamStep'),
    processStep: document.getElementById('processStep')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventSource();
    setupEventListeners();
    checkInitialStatus();
});

// Setup event listeners
function setupEventListeners() {
    elements.startBtn.addEventListener('click', startStream);
    elements.stopBtn.addEventListener('click', stopStream);
    elements.fullscreenBtn.addEventListener('click', toggleFullscreen);
}

// Initialize Server-Sent Events connection
function initializeEventSource() {
    eventSource = new EventSource('/api/stream/events');
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUIFromStatus(data);
    };
    
    eventSource.onerror = (error) => {
        console.error('SSE connection error:', error);
        updateConnectionStatus(false);
        addLog('error', 'Connection to server lost. Retrying...');
        
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
            if (eventSource.readyState === EventSource.CLOSED) {
                initializeEventSource();
            }
        }, 3000);
    };
}

// Check initial status
async function checkInitialStatus() {
    try {
        const response = await fetch('/api/stream/status');
        const data = await response.json();
        updateUIFromStatus(data);
    } catch (error) {
        console.error('Failed to check initial status:', error);
        updateConnectionStatus(false);
    }
}

// Start streaming
async function startStream() {
    if (isStreaming) return;
    
    elements.startBtn.disabled = true;
    addLog('info', 'Starting stream...');
    updatePipelineStep('capture', 'active');
    
    try {
        const response = await fetch('/api/stream/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLog('success', 'Stream started successfully');
            updatePipelineStep('stream', 'active');
        } else {
            // Handle WebSocket disconnection specifically
            if (data.reconnectRequired) {
                addLog('error', data.error);
                addLog('warning', data.suggestion || 'Please reconnect your glasses');
                updateConnectionStatus(false);
                resetStreamUI();
                // Show alert for reconnection required
                alert('Glasses disconnected!\n\n' + (data.suggestion || 'Please close and reopen the app in MentraOS.'));
            } else {
                throw new Error(data.error || 'Failed to start stream');
            }
        }
    } catch (error) {
        addLog('error', `Failed to start stream: ${error.message}`);
        elements.startBtn.disabled = false;
        updatePipelineStep('capture', 'ready');
    }
}

// Stop streaming
async function stopStream() {
    if (!isStreaming) return;
    
    elements.stopBtn.disabled = true;
    addLog('info', 'Stopping stream...');
    
    try {
        const response = await fetch('/api/stream/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLog('success', 'Stream stopped');
            resetPipelineSteps();
        } else {
            throw new Error(data.error || 'Failed to stop stream');
        }
    } catch (error) {
        addLog('error', `Failed to stop stream: ${error.message}`);
        elements.stopBtn.disabled = false;
    }
}

// Update UI from status data
function updateUIFromStatus(data) {
    // Update connection status
    updateConnectionStatus(data.connected);
    
    if (!data.connected) {
        resetStreamUI();
        return;
    }
    
    // Update stream status
    const status = data.status || 'idle';
    elements.streamStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    
    // Handle streaming state
    if (status === 'active' || status === 'starting') {
        if (!isStreaming) {
            isStreaming = true;
            streamStartTime = Date.now();
            startDurationTimer();
            elements.startBtn.style.display = 'none';
            elements.stopBtn.style.display = 'inline-flex';
            elements.stopBtn.disabled = false;
            updatePipelineStep('capture', 'complete');
            updatePipelineStep('stream', 'active');
        }
        
        // Update frame counter
        if (data.frameCount !== undefined) {
            elements.frameCounter.textContent = `Frames: ${data.frameCount}`;
        }
        
        // Display video stream - Always use previewUrl with iframe (Mentra recommendation)
        if (data.previewUrl) {
            // Only update if URL has changed to avoid iframe reloads
            if (!elements.streamFrame.src.includes(data.previewUrl)) {
                displayPreviewStream(data.previewUrl);
                addLog('info', 'Using preview stream (recommended by Mentra)');
            }
            elements.streamFormat.textContent = 'Live Preview';
        } else if (data.hlsUrl) {
            // Fallback to HLS if no preview URL (though Mentra says HLS doesn't work well)
            addLog('warning', 'No preview URL available, attempting HLS (may not work)');
            displayHLSStream(data.hlsUrl);
            elements.streamFormat.textContent = 'HLS (Limited)';
        }
        
        // Update URLs
        if (data.hlsUrl) {
            elements.hlsUrl.value = data.hlsUrl;
            elements.hlsUrlContainer.style.display = 'block';
            elements.urlSection.style.display = 'block';
        }
        
        if (data.dashUrl) {
            elements.dashUrl.value = data.dashUrl;
            elements.dashUrlContainer.style.display = 'block';
            elements.urlSection.style.display = 'block';
        }
        
        // Update stream ID
        if (data.streamId) {
            elements.streamId.textContent = data.streamId.substring(0, 12) + '...';
            elements.streamId.title = data.streamId;
        }
        
    } else if (status === 'idle' || status === 'error' || status === 'stopping') {
        resetStreamUI();
        
        if (status === 'error' && data.error) {
            addLog('error', data.error);
        }
    }
    
    // Handle existing stream notification
    if (data.existingStream) {
        addLog('warning', 'Reconnected to existing stream');
    }
}

// Display preview stream (WebRTC/iframe) - Recommended by Mentra
function displayPreviewStream(url) {
    elements.placeholder.style.display = 'none';
    elements.hlsVideo.style.display = 'none';
    
    // Set iframe attributes for better compatibility
    elements.streamFrame.setAttribute('allow', 'autoplay; fullscreen; camera; microphone');
    elements.streamFrame.setAttribute('allowfullscreen', 'true');
    elements.streamFrame.setAttribute('frameborder', '0');
    
    // Update src and display
    elements.streamFrame.src = url;
    elements.streamFrame.style.display = 'block';
    
    addLog('success', 'Preview stream connected via iframe (Mentra recommended method)');
    addLog('info', `Preview URL: ${url.substring(0, 50)}...`);
}

// Display HLS stream
function displayHLSStream(url) {
    elements.placeholder.style.display = 'none';
    elements.streamFrame.style.display = 'none';
    elements.hlsVideo.style.display = 'block';
    
    if (Hls.isSupported()) {
        if (hlsPlayer) {
            hlsPlayer.destroy();
        }
        
        hlsPlayer = new Hls({
            enableWorker: true,
            lowLatencyMode: true,
            backBufferLength: 90
        });
        
        hlsPlayer.loadSource(url);
        hlsPlayer.attachMedia(elements.hlsVideo);
        
        hlsPlayer.on(Hls.Events.MANIFEST_PARSED, () => {
            elements.hlsVideo.play().catch(e => {
                console.log('Autoplay blocked, user interaction required');
                addLog('warning', 'Click play button to start video');
            });
        });
        
        hlsPlayer.on(Hls.Events.ERROR, (event, data) => {
            if (data.fatal) {
                addLog('error', `HLS Error: ${data.type}`);
            }
        });
        
        addLog('success', 'HLS stream connected');
    } else if (elements.hlsVideo.canPlayType('application/vnd.apple.mpegurl')) {
        // Native HLS support (Safari)
        elements.hlsVideo.src = url;
        elements.hlsVideo.play().catch(e => {
            console.log('Autoplay blocked');
        });
        addLog('success', 'Native HLS stream connected');
    } else {
        addLog('error', 'HLS is not supported in this browser');
    }
}

// Reset stream UI
function resetStreamUI() {
    isStreaming = false;
    streamStartTime = null;
    stopDurationTimer();
    
    elements.startBtn.style.display = 'inline-flex';
    elements.startBtn.disabled = false;
    elements.stopBtn.style.display = 'none';
    elements.stopBtn.disabled = false;
    
    elements.streamFrame.src = '';
    elements.streamFrame.style.display = 'none';
    elements.hlsVideo.style.display = 'none';
    elements.placeholder.style.display = 'flex';
    
    elements.streamId.textContent = '-';
    elements.streamFormat.textContent = '-';
    elements.streamDuration.textContent = '00:00';
    elements.frameCounter.textContent = 'Frames: 0';
    
    elements.urlSection.style.display = 'none';
    elements.hlsUrlContainer.style.display = 'none';
    elements.dashUrlContainer.style.display = 'none';
    
    resetPipelineSteps();
    
    if (hlsPlayer) {
        hlsPlayer.destroy();
        hlsPlayer = null;
    }
}

// Update connection status
function updateConnectionStatus(connected) {
    if (connected) {
        elements.connectionStatus.classList.add('connected');
        elements.statusText.textContent = 'Connected';
        addLog('success', 'Glasses connected');
    } else {
        elements.connectionStatus.classList.remove('connected');
        elements.statusText.textContent = 'Disconnected';
    }
}

// Pipeline step management
function updatePipelineStep(stepId, status) {
    const step = elements[stepId + 'Step'];
    if (!step) return;
    
    step.classList.remove('active', 'complete');
    
    if (status === 'active') {
        step.classList.add('active');
        step.querySelector('.step-status').textContent = 'Active';
    } else if (status === 'complete') {
        step.classList.add('complete');
        step.querySelector('.step-status').textContent = 'Complete';
    } else {
        step.querySelector('.step-status').textContent = 'Ready';
    }
}

function resetPipelineSteps() {
    ['capture', 'stream', 'process'].forEach(step => {
        updatePipelineStep(step, 'ready');
    });
}

// Duration timer
function startDurationTimer() {
    stopDurationTimer();
    updateDuration();
    durationInterval = setInterval(updateDuration, 1000);
}

function stopDurationTimer() {
    if (durationInterval) {
        clearInterval(durationInterval);
        durationInterval = null;
    }
}

function updateDuration() {
    if (!streamStartTime) return;
    
    const duration = Math.floor((Date.now() - streamStartTime) / 1000);
    const minutes = Math.floor(duration / 60);
    const seconds = duration % 60;
    elements.streamDuration.textContent = 
        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// Fullscreen toggle
function toggleFullscreen() {
    const videoContainer = document.querySelector('.video-player');
    
    if (!document.fullscreenElement) {
        videoContainer.requestFullscreen().catch(err => {
            addLog('error', `Failed to enter fullscreen: ${err.message}`);
        });
    } else {
        document.exitFullscreen();
    }
}

// Copy to clipboard
function copyToClipboard(inputId) {
    const input = document.getElementById(inputId);
    input.select();
    document.execCommand('copy');
    
    // Visual feedback
    const button = input.nextElementSibling;
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    setTimeout(() => {
        button.textContent = originalText;
    }, 2000);
}
window.copyToClipboard = copyToClipboard; // Make it globally accessible

// Logging
function addLog(type, message) {
    const time = new Date().toLocaleTimeString('en-US', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-message">${message}</span>
    `;
    
    elements.logsContainer.appendChild(logEntry);
    
    // Keep only last 50 logs
    while (elements.logsContainer.children.length > 50) {
        elements.logsContainer.removeChild(elements.logsContainer.firstChild);
    }
    
    // Auto-scroll to bottom
    elements.logsContainer.scrollTop = elements.logsContainer.scrollHeight;
}
