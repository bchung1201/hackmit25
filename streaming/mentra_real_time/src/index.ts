import { AppServer, AppSession, StreamType } from '@mentra/sdk';
import express from 'express';
import path from 'path';
import cors from 'cors';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const PACKAGE_NAME = process.env.PACKAGE_NAME ?? (() => { throw new Error('PACKAGE_NAME is not set in .env file'); })();
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY ?? (() => { throw new Error('MENTRAOS_API_KEY is not set in .env file'); })();
const PORT = parseInt(process.env.PORT || '3000');

// Extend AppSession type to include our custom properties
declare module "@mentra/sdk" {
    interface AppSession {
        streamStatus: 'idle' | 'starting' | 'active' | 'stopping' | 'error';
        streamId: string | null;
        hlsUrl: string | null;
        dashUrl: string | null;
        previewUrl: string | null;
        thumbnailUrl: string | null;
        error: string | null;
        frameCount: number;
        lastFrameTimestamp: number | null;
    }
}

class RealTimeStreamApp extends AppServer {
    private userSessions = new Map<string, AppSession>();
    private activeStreams = new Map<string, any>();

    constructor() {
        super({
            packageName: PACKAGE_NAME,
            apiKey: MENTRAOS_API_KEY,
            port: PORT,
            publicDir: path.join(__dirname, '../public'),
        });

        this.setupRoutes();
    }

    private setupRoutes() {
        const app = this.getExpressApp();
        
        // Enable CORS for all routes
        app.use(cors());
        app.use(express.json());
        app.use(express.static(path.join(__dirname, '../public')));

        // Main page
        app.get('/', (req, res) => {
            res.sendFile(path.join(__dirname, '../public/index.html'));
        });

        // API: Start streaming
        app.post('/api/stream/start', async (req: any, res) => {
            const userId = req.authUserId;
            const session = userId ? this.userSessions.get(userId) : null;
            
            if (!session) {
                return res.status(401).json({ error: 'No active session. Please connect your glasses.' });
            }

            // Check if WebSocket is connected (if we can access it)
            if ((session as any).ws && (session as any).ws.readyState !== 1) {
                console.log('WebSocket not connected, readyState:', (session as any).ws?.readyState);
                // Remove stale session
                this.userSessions.delete(userId);
                return res.status(503).json({ 
                    error: 'Connection lost. Please reopen the app in MentraOS.',
                    reconnectRequired: true,
                    suggestion: 'Close and reopen the app in MentraOS' 
                });
            }

            try {
                console.log('Starting managed stream...');
                session.streamStatus = 'starting';
                
                // Start a managed stream (uses Cloudflare for HLS/DASH distribution)
                await session.camera.startManagedStream();
                
                res.json({ 
                    success: true, 
                    message: 'Stream starting...',
                    status: session.streamStatus 
                });
            } catch (error: any) {
                console.error('Error starting stream:', error);
                
                // Check if it's a WebSocket connection error
                if (error.message && error.message.includes('WebSocket not connected')) {
                    session.streamStatus = 'error';
                    session.error = 'Glasses disconnected. Please reconnect your glasses and try again.';
                    
                    // Try to remove and re-add the session to force reconnection
                    this.userSessions.delete(userId);
                    
                    res.status(503).json({ 
                        error: 'Glasses disconnected. Please reconnect your glasses and try again.',
                        status: session.streamStatus,
                        reconnectRequired: true,
                        suggestion: 'Close and reopen the app in MentraOS'
                    });
                } else {
                    session.streamStatus = 'error';
                    session.error = error.message || 'Failed to start stream';
                    res.status(500).json({ 
                        error: session.error,
                        status: session.streamStatus 
                    });
                }
            }
        });

        // API: Stop streaming
        app.post('/api/stream/stop', async (req: any, res) => {
            const userId = req.authUserId;
            const session = userId ? this.userSessions.get(userId) : null;
            
            if (!session) {
                return res.status(401).json({ error: 'No active session' });
            }

            try {
                console.log('Stopping stream...');
                session.streamStatus = 'stopping';
                
                await session.camera.stopManagedStream();
                
                session.streamStatus = 'idle';
                session.hlsUrl = null;
                session.dashUrl = null;
                session.previewUrl = null;
                session.streamId = null;
                session.frameCount = 0;
                
                res.json({ 
                    success: true, 
                    message: 'Stream stopped',
                    status: session.streamStatus 
                });
            } catch (error: any) {
                console.error('Error stopping stream:', error);
                res.status(500).json({ 
                    error: error.message || 'Failed to stop stream' 
                });
            }
        });

        // API: Get stream status
        app.get('/api/stream/status', (req: any, res) => {
            const userId = req.authUserId;
            const session = userId ? this.userSessions.get(userId) : null;
            
            if (!session) {
                return res.json({ 
                    connected: false,
                    status: 'disconnected',
                    message: 'No glasses connected' 
                });
            }

            res.json({
                connected: true,
                status: session.streamStatus,
                streamId: session.streamId,
                hlsUrl: session.hlsUrl,
                dashUrl: session.dashUrl,
                previewUrl: session.previewUrl,
                thumbnailUrl: session.thumbnailUrl,
                frameCount: session.frameCount,
                lastFrameTimestamp: session.lastFrameTimestamp,
                error: session.error
            });
        });

        // Server-Sent Events for real-time updates
        app.get('/api/stream/events', (req: any, res) => {
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            });

            const userId = req.authUserId;
            
            // Send initial status
            const session = userId ? this.userSessions.get(userId) : null;
            const initialData = session ? {
                connected: true,
                status: session.streamStatus,
                hlsUrl: session.hlsUrl,
                previewUrl: session.previewUrl,
                frameCount: session.frameCount
            } : {
                connected: false,
                status: 'disconnected'
            };
            
            res.write(`data: ${JSON.stringify(initialData)}\n\n`);

            // Store the SSE connection
            const streamKey = `${userId}-${Date.now()}`;
            this.activeStreams.set(streamKey, res);

            // Heartbeat to keep connection alive
            const heartbeat = setInterval(() => {
                res.write(': heartbeat\n\n');
            }, 30000);

            // Clean up on disconnect
            req.on('close', () => {
                clearInterval(heartbeat);
                this.activeStreams.delete(streamKey);
            });
        });
    }

    protected async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
        console.log(`New session started for user: ${userId}`);
        
        // Initialize session properties
        session.streamStatus = 'idle';
        session.streamId = null;
        session.hlsUrl = null;
        session.dashUrl = null;
        session.previewUrl = null;
        session.thumbnailUrl = null;
        session.error = null;
        session.frameCount = 0;
        session.lastFrameTimestamp = null;

        // Store the session
        this.userSessions.set(userId, session);

        // Subscribe to stream events
        session.subscribe(StreamType.MANAGED_STREAM_STATUS);

        // Listen for stream status updates
        const statusUnsubscribe = session.camera.onManagedStreamStatus((data) => {
            console.log('Stream status update:', data);
            
            session.streamStatus = data.status === 'active' ? 'active' : 
                                  data.status === 'initializing' ? 'starting' : 
                                  'idle';
            session.streamId = data.streamId ?? null;
            session.hlsUrl = data.hlsUrl ?? null;
            session.dashUrl = data.dashUrl ?? null;
            // Check for both previewUrl and webrtcUrl (Mentra recommends using this in iframe)
            session.previewUrl = data.previewUrl ?? (data as any).webrtcUrl ?? null;
            session.thumbnailUrl = data.thumbnailUrl ?? null;
            
            // Log the preview URL for debugging (Mentra recommended to use this)
            if (session.previewUrl) {
                console.log('Preview URL received (use this in iframe):', session.previewUrl);
            }
            
            if (data.status === 'active') {
                session.frameCount++;
                session.lastFrameTimestamp = Date.now();
            }

            // Broadcast update to all SSE connections
            this.broadcastStreamUpdate(userId, {
                connected: true,
                status: session.streamStatus,
                streamId: session.streamId,
                hlsUrl: session.hlsUrl,
                dashUrl: session.dashUrl,
                previewUrl: session.previewUrl,
                thumbnailUrl: session.thumbnailUrl,
                frameCount: session.frameCount,
                lastFrameTimestamp: session.lastFrameTimestamp
            });
        });

        // Clean up on disconnect
        const disconnectUnsubscribe = session.events.onDisconnected(() => {
            console.log(`Session disconnected for user: ${userId}`);
            //this.userSessions.delete(userId);
            
            // Broadcast disconnection
            this.broadcastStreamUpdate(userId, {
                connected: false,
                status: 'disconnected'
            });
        });

        this.addCleanupHandler(() => {
            statusUnsubscribe();
            disconnectUnsubscribe();
        });

        // Check for existing streams
        try {
            const streamInfo = await session.camera.checkExistingStream();
            if (streamInfo.hasActiveStream && streamInfo.streamInfo) {
                console.log('Found existing stream:', streamInfo.streamInfo);
                
                if (streamInfo.streamInfo.type === 'managed') {
                    session.streamStatus = 'active';
                    session.streamId = streamInfo.streamInfo.streamId || null;
                    session.hlsUrl = streamInfo.streamInfo.hlsUrl || null;
                    session.dashUrl = streamInfo.streamInfo.dashUrl || null;
                    // Mentra recommends using previewUrl (might also be called webrtcUrl)
                    session.previewUrl = streamInfo.streamInfo.previewUrl || (streamInfo.streamInfo as any).webrtcUrl || null;
                    
                    // Notify UI about existing stream
                    this.broadcastStreamUpdate(userId, {
                        connected: true,
                        status: session.streamStatus,
                        streamId: session.streamId,
                        hlsUrl: session.hlsUrl,
                        dashUrl: session.dashUrl,
                        previewUrl: session.previewUrl,
                        existingStream: true
                    });
                }
            }
        } catch (error) {
            console.error('Error checking existing stream:', error);
        }

        // Show a welcome message on the glasses
        session.layouts.showTextWall('📹 Real-Time Stream Ready\n\nPress button to control');
    }

    private broadcastStreamUpdate(userId: string, data: any) {
        // Send updates to all SSE connections for this user
        this.activeStreams.forEach((res, key) => {
            if (key.startsWith(userId)) {
                res.write(`data: ${JSON.stringify(data)}\n\n`);
            }
        });
    }

    protected async onStop(sessionId: string, userId: string, reason: string): Promise<void> {
        console.log(`Session stopped for user ${userId}: ${reason}`);
        
        // Clean up session
        //this.userSessions.delete(userId);
        
        // Notify all connections
        this.broadcastStreamUpdate(userId, {
            connected: false,
            status: 'disconnected',
            reason
        });
        
        await super.onStop(sessionId, userId, reason);
    }
}

// Start the server
const app = new RealTimeStreamApp();
app.start().catch(console.error);

console.log(`
🚀 Mentra Real-Time Stream Server
==================================
Server running on port ${PORT}
Open http://localhost:${PORT} in your browser

Make sure you have:
1. Connected your Mentra Live glasses via the MentraOS app
2. Set up your .env file with PACKAGE_NAME and MENTRAOS_API_KEY
3. Your glasses are on the MentraHackathon WiFi network
`);
