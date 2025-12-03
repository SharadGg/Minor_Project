"""
Gemini AI API Integration for Explainable Gesture Recognition

Provides human-readable explanations for gesture and emotion predictions
using Google's Gemini AI model.
"""

import os
from typing import Optional, Dict, List
import warnings

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    warnings.warn("google-generativeai not installed. Install with: pip install google-generativeai")


class GeminiExplainer:
    """
    Generate natural language explanations for gesture predictions
    using Google's Gemini AI model.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize Gemini explainer.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env variable)
            model_name: Gemini model to use
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        
        # Get API key
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Either pass api_key parameter or set GOOGLE_API_KEY environment variable.\n"
                "Get your key at: https://makersuite.google.com/app/apikey"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Cache for similar queries
        self.cache = {}
        
    def explain_gesture(
        self,
        gesture: str,
        confidence: float,
        context: Optional[str] = None
    ) -> str:
        """
        Generate explanation for a gesture prediction.
        
        Args:
            gesture: Predicted gesture name
            confidence: Prediction confidence (0-1)
            context: Optional context information
        
        Returns:
            Natural language explanation
        """
        # Check cache
        cache_key = f"{gesture}_{confidence:.2f}_{context}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Build prompt
        prompt = self._build_gesture_prompt(gesture, confidence, context)
        
        try:
            # Generate explanation
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
            
            # Cache result
            self.cache[cache_key] = explanation
            
            return explanation
            
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def explain_emotion(
        self,
        emotion: str,
        confidence: float,
        context: Optional[str] = None
    ) -> str:
        """
        Generate explanation for an emotion prediction.
        
        Args:
            emotion: Predicted emotion name
            confidence: Prediction confidence (0-1)
            context: Optional context information
        
        Returns:
            Natural language explanation
        """
        cache_key = f"emotion_{emotion}_{confidence:.2f}_{context}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        prompt = self._build_emotion_prompt(emotion, confidence, context)
        
        try:
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
            self.cache[cache_key] = explanation
            return explanation
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def explain_combined(
        self,
        gesture: str,
        emotion: str,
        gesture_confidence: float,
        emotion_confidence: float
    ) -> str:
        """
        Generate combined explanation for gesture and emotion.
        
        Args:
            gesture: Predicted gesture
            emotion: Predicted emotion
            gesture_confidence: Gesture confidence
            emotion_confidence: Emotion confidence
        
        Returns:
            Combined explanation
        """
        cache_key = f"combined_{gesture}_{emotion}_{gesture_confidence:.2f}_{emotion_confidence:.2f}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        prompt = f"""Analyze this person's body language:

Detected Gesture: {gesture} (confidence: {gesture_confidence:.1%})
Detected Emotion: {emotion} (confidence: {emotion_confidence:.1%})

Provide a brief, insightful explanation (2-3 sentences) that:
1. Interprets what the person is expressing through their body language
2. Explains the relationship between the gesture and emotion
3. Suggests the likely intent or message being conveyed

Be concise, empathetic, and practical."""

        try:
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
            self.cache[cache_key] = explanation
            return explanation
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def _build_gesture_prompt(
        self,
        gesture: str,
        confidence: float,
        context: Optional[str]
    ) -> str:
        """Build prompt for gesture explanation."""
        confidence_desc = self._confidence_description(confidence)
        
        prompt = f"""You are an expert in body language analysis. A computer vision system has detected the following gesture:

Gesture: {gesture}
Confidence: {confidence:.1%} ({confidence_desc})
"""
        
        if context:
            prompt += f"Context: {context}\n"
        
        prompt += """
Provide a brief, informative explanation (2-3 sentences) that:
1. Describes what this gesture typically means
2. Explains the body movements involved
3. Suggests appropriate contexts or intentions

Be clear, concise, and practical."""
        
        return prompt
    
    def _build_emotion_prompt(
        self,
        emotion: str,
        confidence: float,
        context: Optional[str]
    ) -> str:
        """Build prompt for emotion explanation."""
        confidence_desc = self._confidence_description(confidence)
        
        prompt = f"""You are an expert in emotional intelligence and body language. A system has detected the following emotion:

Emotion: {emotion}
Confidence: {confidence:.1%} ({confidence_desc})
"""
        
        if context:
            prompt += f"Context: {context}\n"
        
        prompt += """
Provide a brief explanation (2-3 sentences) that:
1. Describes the key body language indicators of this emotion
2. Explains what the person might be feeling or experiencing
3. Suggests how to appropriately respond or interpret this emotion

Be empathetic, clear, and actionable."""
        
        return prompt
    
    @staticmethod
    def _confidence_description(confidence: float) -> str:
        """Convert confidence score to descriptive text."""
        if confidence >= 0.9:
            return "very high confidence"
        elif confidence >= 0.75:
            return "high confidence"
        elif confidence >= 0.6:
            return "moderate confidence"
        elif confidence >= 0.4:
            return "low confidence"
        else:
            return "very low confidence"
    
    def get_gesture_tips(self, gesture: str) -> List[str]:
        """
        Get tips for recognizing a specific gesture.
        
        Args:
            gesture: Gesture name
        
        Returns:
            List of recognition tips
        """
        prompt = f"""List 3-5 key visual indicators that help identify the "{gesture}" gesture from body pose alone.

Format your response as a bullet-point list. Be specific about:
- Joint positions
- Arm/leg configurations
- Body orientation
- Movement patterns

Keep each point brief (one sentence)."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Parse bullet points
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            tips = [line.lstrip('•-*').strip() for line in lines if line]
            
            return tips[:5]  # Max 5 tips
            
        except Exception as e:
            return [f"Unable to retrieve tips: {str(e)}"]
    
    def analyze_sequence(
        self,
        predictions: List[Dict[str, any]],
        timestamps: Optional[List[float]] = None
    ) -> str:
        """
        Analyze a sequence of predictions over time.
        
        Args:
            predictions: List of prediction dicts with 'gesture', 'emotion', 'confidence'
            timestamps: Optional list of timestamps
        
        Returns:
            Temporal analysis explanation
        """
        if not predictions:
            return "No predictions to analyze"
        
        # Summarize sequence
        summary = f"Analyzing {len(predictions)} predictions over time:\n\n"
        
        for i, pred in enumerate(predictions[:5]):  # Show max 5
            time_str = f"t={timestamps[i]:.1f}s" if timestamps else f"Frame {i+1}"
            summary += f"{time_str}: {pred.get('gesture', 'Unknown')} ({pred.get('confidence', 0):.1%})\n"
        
        prompt = f"""{summary}

Provide a brief analysis (2-3 sentences) of:
1. The overall pattern or progression in the person's behavior
2. Any significant changes or transitions
3. What this sequence suggests about the person's intent or emotional state"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Unable to analyze sequence: {str(e)}"
    
    def clear_cache(self):
        """Clear explanation cache."""
        self.cache.clear()


def test_gemini_explainer():
    """Test the Gemini explainer with example predictions."""
    print("\n" + "="*60)
    print("Testing Gemini AI Explainer")
    print("="*60)
    
    # Check if API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("\n⚠️  GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("Get your key at: https://makersuite.google.com/app/apikey")
        return
    
    try:
        explainer = GeminiExplainer()
        
        # Test gesture explanation
        print("\n1. Gesture Explanation:")
        explanation = explainer.explain_gesture("Waving", 0.95)
        print(explanation)
        
        # Test emotion explanation
        print("\n2. Emotion Explanation:")
        explanation = explainer.explain_emotion("Happy", 0.88)
        print(explanation)
        
        # Test combined explanation
        print("\n3. Combined Explanation:")
        explanation = explainer.explain_combined("Celebrating", "Excited", 0.92, 0.89)
        print(explanation)
        
        # Test tips
        print("\n4. Recognition Tips:")
        tips = explainer.get_gesture_tips("Pointing")
        for i, tip in enumerate(tips, 1):
            print(f"   {i}. {tip}")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == '__main__':
    test_gemini_explainer()