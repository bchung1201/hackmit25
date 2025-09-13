"""
Wispr voice integration for hands-free interaction
Handles speech-to-text and text-to-speech for voice commands
"""

import asyncio
import logging
import whisper
import pyaudio
import wave
import tempfile
import os
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class VoiceCommand:
    """Represents a voice command"""
    text: str
    confidence: float
    timestamp: float
    duration: float

@dataclass
class VoiceResponse:
    """Represents a voice response"""
    text: str
    audio_data: Optional[bytes] = None
    timestamp: float = 0.0

class WisprVoiceProcessor:
    """
    Wispr-based voice processing for hands-free interaction
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        sample_rate: int = 16000,
        chunk_size: int = 1024
    ):
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Initialize Whisper model
        self.whisper_model = whisper.load_model(model_size)
        
        # Audio configuration
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Audio streaming
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        
        # Voice command callbacks
        self.command_callbacks: list[Callable[[VoiceCommand], None]] = []
        
        # Voice activity detection
        self.vad_threshold = 0.01
        self.silence_duration = 1.0  # seconds
        self.min_audio_duration = 0.5  # seconds
        
    async def start_listening(self):
        """Start listening for voice commands"""
        logger.info("Starting voice command listening...")
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_listening = True
            
            # Start audio processing loop
            await self._audio_processing_loop()
            
        except Exception as e:
            logger.error(f"Failed to start voice listening: {e}")
            await self.stop_listening()
            raise
    
    async def stop_listening(self):
        """Stop listening for voice commands"""
        logger.info("Stopping voice command listening...")
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    async def _audio_processing_loop(self):
        """Main audio processing loop"""
        audio_buffer = []
        silence_start = None
        is_recording = False
        
        while self.is_listening:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    audio_buffer.append(audio_data)
                    
                    # Convert to numpy array for analysis
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_level = np.sqrt(np.mean(audio_array**2))
                    
                    # Voice activity detection
                    if audio_level > self.vad_threshold:
                        if not is_recording:
                            is_recording = True
                            silence_start = None
                            logger.debug("Voice activity detected, starting recording")
                    else:
                        if is_recording:
                            if silence_start is None:
                                silence_start = asyncio.get_event_loop().time()
                            elif asyncio.get_event_loop().time() - silence_start > self.silence_duration:
                                # End of speech detected
                                await self._process_audio_command(audio_buffer)
                                audio_buffer = []
                                is_recording = False
                                silence_start = None
                                logger.debug("Voice activity ended, processing command")
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_audio_command(self, audio_buffer: list):
        """Process recorded audio as voice command"""
        if len(audio_buffer) < self.min_audio_duration * self.sample_rate / self.chunk_size:
            logger.debug("Audio too short, ignoring")
            return
        
        try:
            # Combine audio chunks
            audio_data = b''.join(audio_buffer)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Write WAV file
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_data)
                
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    temp_file.name,
                    language=self.language
                )
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                # Create voice command
                command = VoiceCommand(
                    text=result["text"].strip(),
                    confidence=result.get("confidence", 0.0),
                    timestamp=asyncio.get_event_loop().time(),
                    duration=len(audio_buffer) * self.chunk_size / self.sample_rate
                )
                
                logger.info(f"Voice command detected: '{command.text}' (confidence: {command.confidence:.2f})")
                
                # Notify callbacks
                for callback in self.command_callbacks:
                    try:
                        callback(command)
                    except Exception as e:
                        logger.error(f"Voice command callback error: {e}")
                
        except Exception as e:
            logger.error(f"Voice command processing error: {e}")
    
    def add_command_callback(self, callback: Callable[[VoiceCommand], None]):
        """Add callback for voice commands"""
        self.command_callbacks.append(callback)
    
    async def speak_response(self, text: str) -> VoiceResponse:
        """
        Convert text to speech and play through Mentra glasses
        
        Args:
            text: Text to convert to speech
            
        Returns:
            VoiceResponse object
        """
        logger.info(f"Speaking response: {text}")
        
        try:
            # For now, we'll use a simple text-to-speech approach
            # In a real implementation, you'd integrate with Mentra's audio output
            
            # Generate audio using a TTS service
            audio_data = await self._text_to_speech(text)
            
            response = VoiceResponse(
                text=text,
                audio_data=audio_data,
                timestamp=asyncio.get_event_loop().time()
            )
            
            # Play audio through Mentra glasses
            await self._play_audio(audio_data)
            
            logger.info("Response spoken successfully")
            return response
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            raise
    
    async def _text_to_speech(self, text: str) -> bytes:
        """Convert text to speech audio data"""
        # This is a placeholder - in practice, you'd use a TTS service
        # For development, we'll return empty audio data
        await asyncio.sleep(0.1)  # Simulate processing time
        return b''  # Empty audio data
    
    async def _play_audio(self, audio_data: bytes):
        """Play audio through Mentra glasses"""
        # This is a placeholder - in practice, you'd stream audio to Mentra glasses
        logger.info("Playing audio through Mentra glasses")
        await asyncio.sleep(0.5)  # Simulate audio playback time
    
    async def process_voice_command(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> VoiceResponse:
        """
        Process a voice command with context
        
        Args:
            command: Voice command to process
            context: Context from current scene analysis
            
        Returns:
            Voice response
        """
        logger.info(f"Processing voice command: {command.text}")
        
        # Parse command intent
        intent = self._parse_command_intent(command.text)
        
        # Generate appropriate response based on intent and context
        response_text = await self._generate_command_response(intent, context)
        
        # Convert to speech
        response = await self.speak_response(response_text)
        
        return response
    
    def _parse_command_intent(self, command_text: str) -> Dict[str, Any]:
        """Parse voice command to extract intent"""
        command_lower = command_text.lower()
        
        intent = {
            "action": "unknown",
            "target": None,
            "parameters": {}
        }
        
        # Simple intent parsing
        if "describe" in command_lower or "what" in command_lower:
            intent["action"] = "describe"
        elif "where" in command_lower or "find" in command_lower:
            intent["action"] = "locate"
        elif "navigate" in command_lower or "go" in command_lower:
            intent["action"] = "navigate"
        elif "count" in command_lower:
            intent["action"] = "count"
        elif "accessibility" in command_lower:
            intent["action"] = "accessibility"
        elif "help" in command_lower:
            intent["action"] = "help"
        
        return intent
    
    async def _generate_command_response(
        self,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate response text based on intent and context"""
        
        action = intent["action"]
        
        if action == "describe":
            return "I can see a room with various objects. Let me analyze the scene and provide a detailed description."
        elif action == "locate":
            return "I'll help you locate what you're looking for. Let me scan the current view."
        elif action == "navigate":
            return "I'll provide navigation instructions based on what I can see in the current view."
        elif action == "count":
            return "I'll count the objects visible in the current scene."
        elif action == "accessibility":
            return "I'll assess the accessibility features and potential barriers in this area."
        elif action == "help":
            return "I can help you understand your environment. Try asking me to describe the room, locate objects, or provide navigation instructions."
        else:
            return "I didn't understand that command. You can ask me to describe the scene, locate objects, or provide navigation help."

# Mock implementation for development without audio hardware
class MockWisprVoiceProcessor:
    """Mock Wispr voice processor for development/testing"""
    
    def __init__(self, *args, **kwargs):
        self.command_callbacks = []
        self.is_listening = False
    
    async def start_listening(self):
        """Mock start listening"""
        logger.info("Mock voice listening started")
        self.is_listening = True
        
        # Simulate voice commands for testing
        await self._simulate_voice_commands()
    
    async def stop_listening(self):
        """Mock stop listening"""
        logger.info("Mock voice listening stopped")
        self.is_listening = False
    
    async def _simulate_voice_commands(self):
        """Simulate voice commands for testing"""
        test_commands = [
            "Describe this room to me",
            "Where is the nearest exit?",
            "Count the objects in this scene",
            "What accessibility features do you see?",
            "Help me navigate to the kitchen"
        ]
        
        for command_text in test_commands:
            if not self.is_listening:
                break
                
            await asyncio.sleep(5)  # Wait 5 seconds between commands
            
            command = VoiceCommand(
                text=command_text,
                confidence=0.95,
                timestamp=asyncio.get_event_loop().time(),
                duration=2.0
            )
            
            logger.info(f"Simulated voice command: {command_text}")
            
            # Notify callbacks
            for callback in self.command_callbacks:
                try:
                    callback(command)
                except Exception as e:
                    logger.error(f"Mock voice command callback error: {e}")
    
    def add_command_callback(self, callback: Callable[[VoiceCommand], None]):
        """Add callback for voice commands"""
        self.command_callbacks.append(callback)
    
    async def speak_response(self, text: str) -> VoiceResponse:
        """Mock speech response"""
        logger.info(f"Mock speaking: {text}")
        await asyncio.sleep(1.0)  # Simulate speech time
        
        return VoiceResponse(
            text=text,
            audio_data=b'',
            timestamp=asyncio.get_event_loop().time()
        )
    
    async def process_voice_command(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> VoiceResponse:
        """Mock command processing"""
        logger.info(f"Mock processing command: {command.text}")
        await asyncio.sleep(0.5)
        
        response_text = f"Mock response to: {command.text}"
        return await self.speak_response(response_text)
