# Mentra Streaming Implementation Notes

## Important: Use Preview URL with iframe (NOT HLS)

Based on feedback from Mentra developers at HackMIT, the HLS URL is not working reliably. Instead, use the **previewUrl** in an iframe element.

## Key Changes Made

### 1. Stream Display Priority
- **Primary**: Use `previewUrl` (or `webrtcUrl`) in an iframe
- **Fallback**: HLS only if no preview URL available (but likely won't work)

### 2. iframe Implementation
```html
<iframe id="streamFrame" 
        allow="autoplay; fullscreen; camera; microphone" 
        allowfullscreen="true"
        frameborder="0">
</iframe>
```

### 3. Server-side Handling
The server checks for both `previewUrl` and `webrtcUrl` fields:
```typescript
session.previewUrl = data.previewUrl ?? data.webrtcUrl ?? null;
```

### 4. Client-side Updates
- Prioritizes preview URL over HLS
- Adds proper iframe attributes for compatibility
- Provides clear logging about which stream type is being used

## Stream Types

1. **Preview Stream (RECOMMENDED)**
   - Low latency WebRTC stream
   - Displayed in iframe
   - URL format: Usually a Cloudflare Stream URL

2. **HLS Stream (LIMITED/BROKEN)**
   - Higher latency
   - Currently not working per Mentra team
   - Kept as fallback only

## Troubleshooting

### If stream doesn't display:
1. Check console for "Preview URL received" message
2. Verify iframe src is being set correctly
3. Ensure glasses are properly connected
4. Try reconnecting the app in MentraOS

### WebSocket Issues:
- The code now handles WebSocket disconnections gracefully
- Clear error messages guide users to reconnect
- Stale sessions are automatically cleaned up

## Testing Checklist
- [ ] Glasses connect successfully
- [ ] Preview URL is received in console logs
- [ ] iframe displays the stream
- [ ] Stream controls work (start/stop)
- [ ] Reconnection after disconnect works
