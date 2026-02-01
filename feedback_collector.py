"""
📊 FEEDBACK STORAGE
===================
Handles feedback collection and storage.

Storage: feedback.json (JSON file for simplicity)
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class FeedbackCollector:
    """Collects and stores user feedback"""
    
    def __init__(self, feedback_file: str = "feedback.json"):
        """Initialize feedback collector"""
        self.feedback_file = Path(feedback_file)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create feedback file if it doesn't exist"""
        if not self.feedback_file.exists():
            self.feedback_file.write_text(json.dumps({"feedback_log": []}, indent=2))
    
    def save_feedback(
        self,
        query: str,
        answer: str,
        helpful: bool = None,
        rating: int = None,
        issues: List[str] = None,
        suggestion: str = "",
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save feedback to file.
        
        Args:
            query: User's question
            answer: Bot's answer
            helpful: True/False if answer was helpful
            rating: 1-5 star rating
            issues: List of issue types
            suggestion: User's suggestion text
            metadata: Additional metadata (mode, query_type, etc.)
            
        Returns:
            Feedback ID
        """
        # Load existing feedback
        data = json.loads(self.feedback_file.read_text())
        
        # Create feedback entry
        feedback_id = str(uuid.uuid4())[:8]
        feedback = {
            "id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer[:500],  # First 500 chars
            "helpful": helpful,
            "rating": rating,
            "issues": issues or [],
            "suggestion": suggestion,
            "metadata": metadata or {}
        }
        
        # Append and save
        data["feedback_log"].append(feedback)
        self.feedback_file.write_text(json.dumps(data, indent=2))
        
        return feedback_id
    
    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback entries"""
        data = json.loads(self.feedback_file.read_text())
        return data.get("feedback_log", [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        feedback_list = self.get_all_feedback()
        
        if not feedback_list:
            return {
                "total": 0,
                "helpful_rate": 0,
                "avg_rating": 0,
                "common_issues": {}
            }
        
        total = len(feedback_list)
        helpful_count = sum(1 for f in feedback_list if f.get("helpful"))
        ratings = [f.get("rating") for f in feedback_list if f.get("rating")]
        
        # Count issues
        issue_counts = {}
        for f in feedback_list:
            for issue in f.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            "total": total,
            "helpful_rate": helpful_count / total if total > 0 else 0,
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "common_issues": issue_counts
        }
    
    def clear_old_feedback(self, days: int = 30):
        """Remove feedback older than specified days"""
        from datetime import timedelta
        
        data = json.loads(self.feedback_file.read_text())
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered = [
            f for f in data["feedback_log"]
            if datetime.fromisoformat(f["timestamp"]) > cutoff_date
        ]
        
        data["feedback_log"] = filtered
        self.feedback_file.write_text(json.dumps(data, indent=2))
        
        return len(data["feedback_log"]) - len(filtered)
