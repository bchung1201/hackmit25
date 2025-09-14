# Mentra Real-Time Video Stream

A real-time video streaming application for Mentra Live glasses with ML processing pipeline support.

## Features

- ✅ Real-time video capture from Mentra Live glasses
- ✅ Live video display via HLS/WebRTC
- ✅ Stream status monitoring
- ✅ Processing pipeline visualization
- ✅ Activity logging
- ✅ Stream URL export for external processing

## Prerequisites

1. **Mentra Live Beta Glasses** - Connected and configured
2. **MentraOS App** - Installed on your phone (iOS/Android)
3. **Mentra Developer Account** - Get credentials at https://console.mentra.glass/
4. **Node.js/Bun** - For running the server

## Setup

### 1. Install Dependencies

```bash
bun install
# or
npm install
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
# Get these from https://console.mentra.glass/
PACKAGE_NAME=your-package-name-here
MENTRAOS_API_KEY=your-api-key-here
PORT=3000
```

### 3. Connect Your Glasses

1. Turn on your Mentra Live glasses (hold left button)
2. Wait for "System Ready" announcement
3. Open MentraOS app on your phone
4. Pair the glasses if not already paired
5. Connect to **MentraHackathon** WiFi (password: hackathon)
6. In glass settings, set "Camera Button Action" to "Use in Apps"

### 4. Start the Server

```bash
bun run dev
# or
npm run dev
```

### 5. Open the Web Interface

Navigate to `http://localhost:3000` in your browser.

## Usage

1. **Start Stream** - Click the "Start Stream" button to begin capturing video
2. **View Stream** - Video will appear in the player once the stream is active
3. **Monitor Status** - Check connection status, frame count, and stream details
4. **Copy URLs** - Get HLS/DASH URLs for external processing
5. **Stop Stream** - Click "Stop Stream" when done

## Stream Types

### Managed Stream (Default)
- Uses Cloudflare for distribution
- Provides HLS and DASH URLs
- WebRTC preview available
- Lower latency

### Direct RTMP (Future)
- Stream directly to custom RTMP servers
- Full control over the stream pipeline

## Processing Pipeline

The application is designed to support three stages:

1. **Capture** - Video capture from glasses
2. **Stream** - RTMP/HLS distribution
3. **Process** - ML/AI processing (to be implemented)

## API Endpoints

- `GET /` - Web interface
- `POST /api/stream/start` - Start streaming
- `POST /api/stream/stop` - Stop streaming
- `GET /api/stream/status` - Get current status
- `GET /api/stream/events` - SSE endpoint for real-time updates

## ML Processing Integration

To add ML processing:

1. Create a processing service that consumes the HLS/DASH stream
2. Use the provided stream URLs from the web interface
3. See `examples/ml_processor.py` for a sample implementation

## Troubleshooting

### Glasses Not Connecting
- Ensure glasses are on "MentraHackathon" WiFi
- Restart glasses by holding left button
- Check Build number is at least 13 in MentraOS app

### Stream Not Starting
- Verify .env configuration
- Check console for error messages
- Ensure no other app is streaming

### Video Not Displaying
- Try refreshing the page
- Check browser console for errors
- Ensure you're using a modern browser with HLS support

## Architecture

```
Mentra Live Glasses
        ↓
   RTMP Stream
        ↓
Cloudflare (Managed)
        ↓
   HLS/DASH/WebRTC
        ↓
   Web Interface
        ↓
  ML Processing
```

## Resources

- [Mentra Documentation](https://docs.mentra.glass/)
- [Developer Console](https://console.mentra.glass/)
- [Discord Community](https://mentra.glass/discord)

## License

MIT