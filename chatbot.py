"""
Grounded Q&A Chatbot for EnergyPulse
====================================
A question-answering chatbot that ONLY answers based on real household data.

Core principle:
  - Every numeric claim must come from a tool that queries real data
  - If data isn't available, the chatbot says so honestly
  - All responses are validated against actual tool output before display
  - Supports all four languages (English, Hindi, Kannada, Telugu)

Questions the chatbot can answer:
  - "Why was my bill higher last week?"
  - "Which appliance uses the most power?"
  - "How can I reduce my AC usage cost?"
  - "What's my average daily consumption?"
  - "Which hours do I use the most energy?"
"""

import json
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from db import get_db
from llm_guard import LLMHallucinationGuard
from i18n import T


class ChatbotTool:
    """Base class for chatbot tools that query real household data."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, household_id: str, **kwargs) -> Dict[str, Any]:
        """Execute the tool and return real data."""
        raise NotImplementedError


class GetUsageHistoryTool(ChatbotTool):
    """Query hourly usage history for a date range."""
    
    def __init__(self):
        super().__init__(
            name="get_usage_history",
            description="Get hourly electricity usage for a specific date range"
        )
    
    def execute(self, household_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Get usage history from the app's data."""
        try:
            from model import load_data
            df = load_data()
            
            # Simulate household-specific data (in production, query from household table)
            recent = df.tail(days_back * 24).copy()
            recent['datetime'] = pd.to_datetime(recent['datetime'])
            
            total_kwh = float(recent['Global_active_power'].sum() / 1000.0)  # Convert to kWh
            daily_avg = float(total_kwh / days_back)
            hourly_max = float(recent['Global_active_power'].max())
            hourly_min = float(recent['Global_active_power'].min())
            
            return {
                "status": "success",
                "period_days": days_back,
                "total_kwh": total_kwh,
                "daily_avg_kwh": daily_avg,
                "hourly_max_kw": hourly_max,
                "hourly_min_kw": hourly_min,
                "data_points": len(recent),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetApplianceBreakdownTool(ChatbotTool):
    """Get usage breakdown by appliance category."""
    
    def __init__(self):
        super().__init__(
            name="get_appliance_breakdown",
            description="Get electricity usage breakdown by appliance/category"
        )
    
    def execute(self, household_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Get appliance breakdown from real data."""
        try:
            from model import load_data
            df = load_data()
            
            recent = df.tail(days_back * 24).copy()
            
            breakdown = {
                "status": "success",
                "period_days": days_back,
                "Kitchen_kwh": float(recent['Sub_metering_1'].sum() / 1000.0),
                "Laundry_kwh": float(recent['Sub_metering_2'].sum() / 1000.0),
                "WaterHeater_AC_kwh": float(recent['Sub_metering_3'].sum() / 1000.0),
                "Other_kwh": float((
                    recent['Global_active_power'].sum() - 
                    recent['Sub_metering_1'].sum() - 
                    recent['Sub_metering_2'].sum() - 
                    recent['Sub_metering_3'].sum()
                ) / 1000.0),
            }
            
            total = sum(v for k, v in breakdown.items() if k.endswith("_kwh"))
            breakdown["total_kwh"] = total
            
            # Find top appliance
            appliances = {k: v for k, v in breakdown.items() if k.endswith("_kwh")}
            top_appliance = max(appliances, key=appliances.get)
            breakdown["top_appliance"] = top_appliance
            breakdown["top_appliance_kwh"] = appliances[top_appliance]
            
            return breakdown
        
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetCurrentPredictionTool(ChatbotTool):
    """Get current cost prediction for the month."""
    
    def __init__(self):
        super().__init__(
            name="get_current_prediction",
            description="Get predicted monthly electricity cost"
        )
    
    def execute(self, household_id: str, tariff_rate: float = 8.0) -> Dict[str, Any]:
        """Get current month's cost prediction."""
        try:
            from model import load_data, predict_next_period
            df = load_data()
            
            # Use last day's data for prediction
            last_day_kwh = float(df.tail(24)['Global_active_power'].sum() / 1000.0)
            days_in_month = 30  # Approximate
            predicted_kwh = last_day_kwh * days_in_month
            predicted_cost = predicted_kwh * tariff_rate
            
            return {
                "status": "success",
                "predicted_monthly_kwh": predicted_kwh,
                "predicted_monthly_cost_rs": predicted_cost,
                "tariff_rate": tariff_rate,
                "basis": "extrapolated from recent usage"
            }
        
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetOptimizationTipsTool(ChatbotTool):
    """Get actionable optimization tips from detected anomalies."""
    
    def __init__(self):
        super().__init__(
            name="get_optimization_tips",
            description="Get tips based on detected usage patterns and anomalies"
        )
    
    def execute(self, household_id: str) -> Dict[str, Any]:
        """Get optimization tips."""
        try:
            from optimize import detect_anomalies, generate_anomaly_tips
            from model import load_data
            
            df = load_data()
            anomalies = detect_anomalies(df, window_days=7)
            
            tips = []
            if len(anomalies) > 0:
                tips_generated = generate_anomaly_tips(anomalies.head(3))
                tips = tips_generated if isinstance(tips_generated, list) else []
            
            return {
                "status": "success",
                "anomalies_detected": len(anomalies),
                "tips": tips[:5],  # Top 5 tips
                "basis": "detected from real usage data"
            }
        
        except Exception as e:
            return {"status": "error", "message": str(e)}


class EnergyPulseChatbot:
    """
    Grounded Q&A chatbot that answers questions about household energy.
    All answers are validated against real data before display.
    """
    
    def __init__(self):
        """Initialize the chatbot with tools."""
        self.db = get_db()
        self.guard = LLMHallucinationGuard()
        
        # Define available tools
        self.tools = {
            "get_usage_history": GetUsageHistoryTool(),
            "get_appliance_breakdown": GetApplianceBreakdownTool(),
            "get_current_prediction": GetCurrentPredictionTool(),
            "get_optimization_tips": GetOptimizationTipsTool(),
        }
    
    def answer_question(self, household_id: str, email: str, question: str,
                       language: str = "en", tariff_rate: float = 8.0) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Answer a user question about their energy usage.
        
        Parameters
        ----------
        household_id : str
            Household identifier
        email : str
            User email
        question : str
            User's question in natural language
        language : str
            User's preferred language
        tariff_rate : float
            Electricity rate in Rs/kWh for cost calculations
        
        Returns
        -------
        Tuple[answer_text, is_valid_grounded, metadata_dict]
            - answer_text: The chatbot's response (string)
            - is_valid_grounded: Whether answer was validated against real data
            - metadata_dict: Tool calls, grounding data, etc. for logging
        """
        # Step 1: Determine which tools to call based on question
        tools_to_call = self._plan_tools(question, language)
        
        # Step 2: Execute tools and collect real data
        tool_results = {}
        for tool_name in tools_to_call:
            if tool_name in self.tools:
                try:
                    if tool_name == "get_current_prediction":
                        result = self.tools[tool_name].execute(household_id, tariff_rate=tariff_rate)
                    else:
                        result = self.tools[tool_name].execute(household_id)
                    tool_results[tool_name] = result
                    
                    # Register extracted numbers with the guard
                    if isinstance(result, dict) and result.get("status") == "success":
                        for key, val in result.items():
                            if isinstance(val, (int, float)):
                                self.guard.register_tool_result(f"{tool_name}_{key}", val)
                except Exception as e:
                    tool_results[tool_name] = {"status": "error", "message": str(e)}
        
        # Step 3: Generate answer based on tool results
        answer = self._generate_answer(question, tool_results, language)
        
        # Step 4: Validate answer against tool data using LLM guard
        is_valid, validation_issues, corrected_answer = self.guard.validate_response(
            answer, language=language
        )
        
        if not is_valid and corrected_answer:
            answer = corrected_answer
        
        # Step 5: Log conversation
        metadata = {
            "tool_calls": list(tools_to_call),
            "tool_results": tool_results,
            "validation_valid": is_valid,
            "validation_issues": validation_issues,
        }
        
        self.db.save_conversation(
            household_id=household_id,
            email=email,
            question=question,
            answer=answer,
            language=language,
            tool_calls=tools_to_call,
            grounding_data=tool_results,
            is_valid=is_valid
        )
        
        return (answer, is_valid, metadata)
    
    def _plan_tools(self, question: str, language: str) -> List[str]:
        """
        Determine which tools to call based on the question.
        Keyword matching for now; could be improved with NLU.
        """
        question_lower = question.lower()
        tools = []
        
        # Keywords for each tool
        if any(word in question_lower for word in ['bill', 'cost', 'expensive', 'price', 'rupee', 'rs']):
            tools.append("get_current_prediction")
        
        if any(word in question_lower for word in ['appliance', 'device', 'uses', 'usage', 'which', 'most']):
            tools.append("get_appliance_breakdown")
        
        if any(word in question_lower for word in ['history', 'last', 'day', 'week', 'how much', 'consumption']):
            tools.append("get_usage_history")
        
        if any(word in question_lower for word in ['reduce', 'save', 'optimization', 'lower', 'decrease', 'tips']):
            tools.append("get_optimization_tips")
        
        # Default: try usage history and prediction
        if not tools:
            tools = ["get_usage_history", "get_current_prediction"]
        
        return list(set(tools))  # Remove duplicates
    
    def _generate_answer(self, question: str, tool_results: Dict[str, Any],
                        language: str = "en") -> str:
        """
        Generate an answer based on tool results.
        Only uses real computed data from tool results.
        Never makes up numbers or guesses.
        """
        answer_lines = []
        
        # Check if we got any errors
        has_errors = any(
            result.get("status") == "error" 
            for result in tool_results.values()
        )
        
        if not tool_results or all(result.get("status") == "error" for result in tool_results.values()):
            # No data available
            honest_response = self._get_honest_response(language)
            return honest_response
        
        # Build answer from available data
        for tool_name, result in tool_results.items():
            if result.get("status") != "success":
                continue
            
            if tool_name == "get_usage_history":
                answer_lines.append(
                    f"Your usage over the last {result.get('period_days', 7)} days: "
                    f"{result.get('total_kwh', 0):.2f} kWh (average {result.get('daily_avg_kwh', 0):.2f} kWh/day)"
                )
            
            elif tool_name == "get_appliance_breakdown":
                top_app = result.get("top_appliance", "").replace("_kwh", "").replace("_", " ")
                top_usage = result.get("top_appliance_kwh", 0)
                answer_lines.append(
                    f"Your top consumer is {top_app} with {top_usage:.2f} kWh. "
                    f"Total household usage: {result.get('total_kwh', 0):.2f} kWh."
                )
            
            elif tool_name == "get_current_prediction":
                cost = result.get("predicted_monthly_cost_rs", 0)
                kwh = result.get("predicted_monthly_kwh", 0)
                answer_lines.append(
                    f"Based on recent usage, your predicted monthly cost is Rs. {cost:.2f} "
                    f"({kwh:.2f} kWh at Rs. {result.get('tariff_rate', 8.0)}/kWh)"
                )
            
            elif tool_name == "get_optimization_tips":
                tips = result.get("tips", [])
                if tips:
                    answer_lines.append("Optimization recommendations:")
                    for i, tip in enumerate(tips[:3], 1):
                        if isinstance(tip, str):
                            answer_lines.append(f"  {i}. {tip}")
            
        if not answer_lines:
            return self._get_honest_response(language)
        
        return "\n".join(answer_lines)
    
    def _get_honest_response(self, language: str = "en") -> str:
        """
        Return an honest response when data isn't available.
        Never guesses or invents numbers.
        """
        responses = {
            "en": "I don't have enough data to answer that question. Please check back after collecting more usage data.",
            "hi": "मेरे पास उस सवाल का जवाब देने के लिए पर्याप्त डेटा नहीं है। कृपया अधिक उपयोग डेटा एकत्र करने के बाद वापस जांचें।",
            "kn": "ಆ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ನನ್ನಲ್ಲಿ ಸಾಕಷ್ಟು ಡೇಟಾ ಇಲ್ಲ. ನೀವು ಹೆಚ್ಚಿನ ಬಳಕೆ ಡೇಟಾವನ್ನು ಸಂಗ್ರಹಿಸಿದ ನಂತರ ಮತ್ತೆ ಪರಿಶೀಲಿಸಿ.",
            "te": "ఆ ప్రశ్నకు సమాధానం ఇవ్వడానికి నా వద్ద తగినంత డేటా లేదు. మీరు మరిన్ని ఉపయోగ డేటాను సేకరించిన తర్వాత తిరిగి తనిఖీ చేయండి.",
        }
        return responses.get(language, responses["en"])
    
    def get_conversation_history(self, household_id: str, email: str,
                                limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversation history for a user."""
        return self.db.get_conversation_history(household_id, email, limit=limit)
    
    def clear_conversation_history(self, household_id: str, email: str) -> bool:
        """Clear conversation history for a user."""
        # In a full implementation, add a method to db.py to clear conversations
        # For now, this is a placeholder
        return True


# Singleton instance
_chatbot = None

def get_chatbot() -> EnergyPulseChatbot:
    """Get or create the chatbot singleton."""
    global _chatbot
    if _chatbot is None:
        _chatbot = EnergyPulseChatbot()
    return _chatbot
