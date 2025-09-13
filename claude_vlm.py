"""
Claude VLM integration for environment understanding
Handles visual language model processing for scene analysis and description
"""

import asyncio
import logging
import base64
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

@dataclass
class ObjectDescription:
    """Description of a segmented object"""
    object_id: int
    category: str
    description: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    attributes: Dict[str, Any]

@dataclass
class SceneDescription:
    """Complete scene description"""
    overall_description: str
    objects: List[ObjectDescription]
    spatial_relationships: List[str]
    room_type: Optional[str]
    safety_concerns: List[str]
    accessibility_features: List[str]
    timestamp: float

class ClaudeVLMProcessor:
    """
    Claude VLM processor for environment understanding
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        
    async def analyze_scene(
        self,
        frame: np.ndarray,
        segmentation_result: Optional[Any] = None,
        user_query: Optional[str] = None
    ) -> SceneDescription:
        """
        Analyze a scene using Claude VLM
        
        Args:
            frame: Input video frame
            segmentation_result: Optional segmentation results
            user_query: Optional specific query about the scene
            
        Returns:
            SceneDescription object
        """
        logger.info("Analyzing scene with Claude VLM")
        
        try:
            # Encode frame as base64
            frame_base64 = self._encode_frame(frame)
            
            # Prepare prompt
            prompt = self._build_analysis_prompt(segmentation_result, user_query)
            
            # Call Claude API
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": frame_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            scene_description = self._parse_claude_response(response.content[0].text)
            
            logger.info("Scene analysis completed successfully")
            return scene_description
            
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            raise
    
    async def answer_question(
        self,
        frame: np.ndarray,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer a specific question about the scene
        
        Args:
            frame: Input video frame
            question: Question to answer
            context: Optional context from previous analysis
            
        Returns:
            Answer string
        """
        logger.info(f"Answering question: {question}")
        
        try:
            # Encode frame as base64
            frame_base64 = self._encode_frame(frame)
            
            # Build context-aware prompt
            context_text = ""
            if context:
                context_text = f"Context: {json.dumps(context, indent=2)}\n\n"
            
            prompt = f"""{context_text}Question: {question}

Please provide a clear, concise answer based on what you can see in the image. If the question cannot be answered from the image alone, please say so."""
            
            # Call Claude API
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": frame_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            answer = response.content[0].text
            logger.info("Question answered successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Question answering error: {e}")
            raise
    
    async def generate_navigation_instructions(
        self,
        frame: np.ndarray,
        destination: str,
        current_location: Optional[str] = None
    ) -> List[str]:
        """
        Generate navigation instructions based on current view
        
        Args:
            frame: Current view frame
            destination: Where the user wants to go
            current_location: Optional current location description
            
        Returns:
            List of navigation instructions
        """
        logger.info(f"Generating navigation instructions to: {destination}")
        
        try:
            frame_base64 = self._encode_frame(frame)
            
            location_context = f"Current location: {current_location}\n" if current_location else ""
            
            prompt = f"""{location_context}Destination: {destination}

Based on what you can see in this image, provide step-by-step navigation instructions to reach the destination. Focus on visible landmarks, doors, hallways, and other navigational features."""
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": frame_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            instructions_text = response.content[0].text
            instructions = self._parse_instructions(instructions_text)
            
            logger.info(f"Generated {len(instructions)} navigation instructions")
            return instructions
            
        except Exception as e:
            logger.error(f"Navigation instruction generation error: {e}")
            raise
    
    async def assess_accessibility(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Assess accessibility features and barriers in the scene
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with accessibility assessment
        """
        logger.info("Assessing accessibility features")
        
        try:
            frame_base64 = self._encode_frame(frame)
            
            prompt = """Analyze this image for accessibility features and potential barriers. Consider:

1. Wheelchair accessibility (ramps, elevators, wide doorways)
2. Visual accessibility (braille, high contrast, clear signage)
3. Hearing accessibility (visual alerts, captioning)
4. Mobility barriers (stairs, narrow passages, obstacles)
5. Safety features (handrails, non-slip surfaces)

Provide a detailed assessment with specific observations."""
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": frame_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            assessment_text = response.content[0].text
            assessment = self._parse_accessibility_assessment(assessment_text)
            
            logger.info("Accessibility assessment completed")
            return assessment
            
        except Exception as e:
            logger.error(f"Accessibility assessment error: {e}")
            raise
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_base64
    
    def _build_analysis_prompt(
        self,
        segmentation_result: Optional[Any],
        user_query: Optional[str]
    ) -> str:
        """Build analysis prompt for Claude"""
        
        if user_query:
            return f"""Please analyze this image and answer the following question: {user_query}

Provide a detailed response based on what you can see in the image."""
        
        base_prompt = """Please analyze this image and provide a comprehensive description including:

1. Overall scene description
2. Objects visible in the scene (with approximate locations)
3. Spatial relationships between objects
4. Room/space type and purpose
5. Any safety concerns or hazards
6. Accessibility features or barriers

Be specific and detailed in your observations."""
        
        if segmentation_result:
            base_prompt += f"""

Note: This image has been segmented with {len(segmentation_result.masks)} objects detected. Please focus your analysis on these segmented regions."""
        
        return base_prompt
    
    def _parse_claude_response(self, response_text: str) -> SceneDescription:
        """Parse Claude's response into structured data"""
        # This is a simplified parser - in practice, you'd want more robust parsing
        lines = response_text.split('\n')
        
        overall_description = ""
        objects = []
        spatial_relationships = []
        room_type = None
        safety_concerns = []
        accessibility_features = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "overall" in line.lower() or "scene" in line.lower():
                current_section = "overall"
                overall_description = line
            elif "object" in line.lower():
                current_section = "objects"
            elif "spatial" in line.lower() or "relationship" in line.lower():
                current_section = "spatial"
            elif "room" in line.lower() or "space" in line.lower():
                current_section = "room"
            elif "safety" in line.lower() or "hazard" in line.lower():
                current_section = "safety"
            elif "accessibility" in line.lower():
                current_section = "accessibility"
            else:
                if current_section == "overall":
                    overall_description += " " + line
                elif current_section == "objects":
                    # Simple object parsing
                    objects.append(ObjectDescription(
                        object_id=len(objects),
                        category="unknown",
                        description=line,
                        confidence=0.8,
                        bbox=(0, 0, 0, 0),
                        attributes={}
                    ))
                elif current_section == "spatial":
                    spatial_relationships.append(line)
                elif current_section == "room":
                    room_type = line
                elif current_section == "safety":
                    safety_concerns.append(line)
                elif current_section == "accessibility":
                    accessibility_features.append(line)
        
        return SceneDescription(
            overall_description=overall_description,
            objects=objects,
            spatial_relationships=spatial_relationships,
            room_type=room_type,
            safety_concerns=safety_concerns,
            accessibility_features=accessibility_features,
            timestamp=asyncio.get_event_loop().time()
        )
    
    def _parse_instructions(self, instructions_text: str) -> List[str]:
        """Parse navigation instructions into list"""
        lines = instructions_text.split('\n')
        instructions = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or 
                        line.startswith(('-', '•', '*'))):
                instructions.append(line)
        
        return instructions
    
    def _parse_accessibility_assessment(self, assessment_text: str) -> Dict[str, Any]:
        """Parse accessibility assessment into structured data"""
        return {
            "assessment_text": assessment_text,
            "wheelchair_accessible": "wheelchair" in assessment_text.lower(),
            "visual_accessibility": "braille" in assessment_text.lower() or "contrast" in assessment_text.lower(),
            "safety_features": "handrail" in assessment_text.lower() or "non-slip" in assessment_text.lower(),
            "barriers": "stairs" in assessment_text.lower() or "narrow" in assessment_text.lower(),
            "timestamp": asyncio.get_event_loop().time()
        }

# Mock implementation for development without Claude API
class MockClaudeVLMProcessor:
    """Mock Claude VLM processor for development/testing"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    async def analyze_scene(
        self,
        frame: np.ndarray,
        segmentation_result: Optional[Any] = None,
        user_query: Optional[str] = None
    ) -> SceneDescription:
        """Generate mock scene description"""
        logger.info("Generating mock scene analysis")
        
        await asyncio.sleep(0.5)  # Simulate processing time
        
        objects = [
            ObjectDescription(
                object_id=0,
                category="chair",
                description="A wooden chair with four legs",
                confidence=0.9,
                bbox=(100, 200, 80, 120),
                attributes={"material": "wood", "color": "brown"}
            ),
            ObjectDescription(
                object_id=1,
                category="table",
                description="A rectangular wooden table",
                confidence=0.85,
                bbox=(200, 150, 150, 100),
                attributes={"material": "wood", "color": "brown"}
            )
        ]
        
        return SceneDescription(
            overall_description="A typical office or study room with wooden furniture",
            objects=objects,
            spatial_relationships=["The chair is positioned near the table"],
            room_type="office",
            safety_concerns=[],
            accessibility_features=["Wide doorways", "Good lighting"],
            timestamp=asyncio.get_event_loop().time()
        )
    
    async def answer_question(
        self,
        frame: np.ndarray,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate mock answer"""
        logger.info(f"Generating mock answer for: {question}")
        await asyncio.sleep(0.3)
        return f"Mock answer for: {question}"
    
    async def generate_navigation_instructions(
        self,
        frame: np.ndarray,
        destination: str,
        current_location: Optional[str] = None
    ) -> List[str]:
        """Generate mock navigation instructions"""
        logger.info(f"Generating mock navigation to: {destination}")
        await asyncio.sleep(0.4)
        return [
            "1. Walk straight ahead",
            "2. Turn left at the corridor",
            f"3. Continue until you reach {destination}"
        ]
    
    async def assess_accessibility(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """Generate mock accessibility assessment"""
        logger.info("Generating mock accessibility assessment")
        await asyncio.sleep(0.3)
        return {
            "assessment_text": "Mock accessibility assessment",
            "wheelchair_accessible": True,
            "visual_accessibility": True,
            "safety_features": True,
            "barriers": False,
            "timestamp": asyncio.get_event_loop().time()
        }
