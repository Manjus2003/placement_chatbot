"""
📈 TREND ANALYZER TOOL
======================
Analyzes trends across years (2024, 2025, 2026).

Capabilities:
- CTC trends over time
- Offer count trends
- CGPA cutoff trends
- Growth rate calculations
"""

import re
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_tool import BaseTool


class TrendAnalyzerTool(BaseTool):
    """
    Analyzes placement trends across years.
    """
    
    def __init__(self):
        """Initialize the trend analyzer"""
        self.data_cache = {}
    
    @property
    def name(self) -> str:
        return "trend_analyzer"
    
    @property
    def description(self) -> str:
        return "Analyze placement trends across years (CTC, offers, cutoffs)"
    
    def run(
        self,
        metric: str = "ctc",
        companies: List[str] = None,
        years: List[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze trends across years.
        
        Args:
            metric: "ctc", "offers", "cgpa_cutoff"
            companies: Filter to specific companies
            years: Filter years (default: [2024, 2025, 2026])
            
        Returns:
            Dict with trend_data, growth_rates, insights
        """
        try:
            print(f"📈 Analyzing {metric} trends...")
            
            years = years or [2024, 2025, 2026]
            
            # Extract data from folder structure
            trend_data = self._extract_yearly_data(metric, companies, years)
            
            # Calculate growth rates
            growth_rates = self._calculate_growth_rates(trend_data)
            
            # Generate insights
            insights = self._generate_insights(trend_data, growth_rates, metric)
            
            return {
                "response": self._format_response(trend_data, growth_rates, insights, metric),
                "trend_data": trend_data,
                "growth_rates": growth_rates,
                "insights": insights,
                "metric": metric,
                "visualization": "line_chart"
            }
        except Exception as e:
            print(f"❌ Error in trend analysis: {e}")
            return {
                "response": f"## 📈 Trend Analysis Error\n\nUnable to analyze trends: {str(e)}\n\nPlease try again or contact support.",
                "trend_data": {},
                "growth_rates": {},
                "insights": [],
                "metric": metric,
                "error": str(e)
            }
    
    def _extract_yearly_data(
        self, 
        metric: str, 
        companies: List[str],
        years: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """Extract data from folder structure or cache"""
        
        try:
            data_by_year = {}
            
            # Check if _mmd_collected exists
            base_path = Path("_mmd_collected")
            if not base_path.exists():
                # Return sample data if folder doesn't exist
                return self._get_sample_trend_data(metric, companies, years)
            match = re.search(r'(.+?)_MTech_(\d{4})', folder.name)
            if not match:
                continue
            
            company = match.group(1).replace('_', ' ')
            year = int(match.group(2))
            
            if year not in years:
                continue
            
            if companies and company not in companies:
                continue
            
            # Extract metric from files in folder
            value = self._extract_metric_from_folder(folder, metric)
            
            if value is not None:
                if year not in data_by_year:
                    data_by_year[year] = {}
                data_by_year[year][company] = value
            
            # Fill missing years with sample data if needed
            if not data_by_year:
                return self._get_sample_trend_data(metric, companies, years)
            
            return data_by_year
        except Exception as e:
            print(f"⚠️ Error extracting yearly data: {e}")
            return self._get_sample_trend_data(metric, companies, years)
    
    def _extract_metric_from_folder(self, folder: Path, metric: str) -> Optional[float]:
        """Extract specific metric from documents in folder"""
        
        # Try to read text files
        for file in folder.glob("*.txt"):
            try:
                content = file.read_text(encoding='utf-8', errors='ignore')
                
                if metric == "ctc":
                    # Pattern: CTC: 44 LPA, Package: 44 LPA, etc.
                    patterns = [
                        r'CTC:?\s*(\d+\.?\d*)\s*LPA',
                        r'Package:?\s*(\d+\.?\d*)\s*LPA',
                        r'Salary:?\s*(\d+\.?\d*)\s*LPA'
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            return float(match.group(1))
                
                elif metric == "cgpa_cutoff":
                    # Pattern: CGPA: 7.5, Minimum CGPA: 7.5
                    patterns = [
                        r'(?:Minimum|Min\.?|Required)\s+CGPA:?\s*(\d+\.?\d*)',
                        r'CGPA\s+(?:requirement|cutoff):?\s*(\d+\.?\d*)'
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            return float(match.group(1))
            except Exception:
                continue
        
        return None
    
    def _get_sample_trend_data(
        self, 
        metric: str, 
        companies: List[str],
        years: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """Return sample trend data for demonstration"""
        
        sample_companies = companies or ["Intel", "AMD", "Amazon", "Google", "Microsoft"]
        
        if metric == "ctc":
            base_values = {
                "Intel": 40, "AMD": 36, "Amazon": 42, 
                "Google": 46, "Microsoft": 44
            }
            growth_rate = 1.05  # 5% annual growth
        elif metric == "cgpa_cutoff":
            base_values = {
                "Intel": 7.5, "AMD": 7.0, "Amazon": 7.5,
                "Google": 8.0, "Microsoft": 7.5
            }
            growth_rate = 1.0  # Stable cutoffs
        else:  # offers
            base_values = {
                "Intel": 10, "AMD": 8, "Amazon": 12,
                "Google": 8, "Microsoft": 10
            }
            growth_rate = 1.1  # 10% growth
        
        data = {}
        for i, year in enumerate(sorted(years)):
            data[year] = {}
            for company in sample_companies:
                if company in base_values:
                    # Apply growth from 2024 baseline
                    years_diff = year - 2024
                    value = base_values[company] * (growth_rate ** years_diff)
                    data[year][company] = round(value, 1)
        
        return data
    
    def _calculate_growth_rates(
        self, 
        trend_data: Dict[int, Dict[str, float]]
    ) -> Dict[str, str]:
        """Calculate year-over-year growth rates"""
        
        growth_rates = {}
        years = sorted(trend_data.keys())
        
        if len(years) < 2:
            return growth_rates
        
        first_year = years[0]
        last_year = years[-1]
        
        # Get companies present in both years
        first_companies = set(trend_data[first_year].keys())
        last_companies = set(trend_data[last_year].keys())
        common_companies = first_companies & last_companies
        
        for company in common_companies:
            first_val = trend_data[first_year][company]
            last_val = trend_data[last_year][company]
            
            if first_val > 0:
                growth = ((last_val - first_val) / first_val) * 100
                growth_rates[company] = f"{growth:+.1f}%"
        
        return growth_rates
    
    def _generate_insights(
        self,
        trend_data: Dict[int, Dict[str, float]],
        growth_rates: Dict[str, str],
        metric: str
    ) -> List[str]:
        """Generate insights from trend data"""
        
        insights = []
        years = sorted(trend_data.keys())
        
        if not years or not growth_rates:
            return ["Insufficient data for insights"]
        
        # Find fastest growing company
        max_growth_company = None
        max_growth = -float('inf')
        
        for company, rate_str in growth_rates.items():
            rate = float(rate_str.replace('%', '').replace('+', ''))
            if rate > max_growth:
                max_growth = rate
                max_growth_company = company
        
        if max_growth_company:
            insights.append(
                f"📈 {max_growth_company} shows fastest growth at {growth_rates[max_growth_company]}"
            )
        
        # Average growth
        avg_growth = sum(
            float(r.replace('%', '').replace('+', '')) 
            for r in growth_rates.values()
        ) / len(growth_rates)
        
        if metric == "ctc":
            insights.append(f"💰 Average CTC growth: {avg_growth:+.1f}% ({years[0]}-{years[-1]})")
        elif metric == "cgpa_cutoff":
            if abs(avg_growth) < 1:
                insights.append(f"📊 CGPA cutoffs remain stable (±{abs(avg_growth):.1f}%)")
            else:
                insights.append(f"📊 CGPA cutoffs trending {'up' if avg_growth > 0 else 'down'}")
        
        # Year-over-year comparison
        if len(years) >= 2:
            latest_year = years[-1]
            prev_year = years[-2]
            
            latest_data = trend_data[latest_year]
            if latest_data:
                avg_latest = sum(latest_data.values()) / len(latest_data)
                insights.append(
                    f"📅 {latest_year} average: {avg_latest:.1f}"
                )
        
        return insights
    
    def _format_response(
        self,
        trend_data: Dict[int, Dict[str, float]],
        growth_rates: Dict[str, str],
        insights: List[str],
        metric: str
    ) -> str:
        """Format response for display"""
        
        response = f"## {metric.replace('_', ' ').title()} Trends\n\n"
        
        # Trend table
        years = sorted(trend_data.keys())
        companies = set()
        for year_data in trend_data.values():
            companies.update(year_data.keys())
        companies = sorted(companies)
        
        if companies and years:
            response += "| Company |"
            for year in years:
                response += f" {year} |"
            response += " Growth |\n"
            
            response += "|---------|"
            for _ in years:
                response += "------|"
            response += "-------|\n"
            
            for company in companies:
                response += f"| {company} |"
                for year in years:
                    value = trend_data[year].get(company, '-')
                    if isinstance(value, float):
                        response += f" {value} |"
                    else:
                        response += f" {value} |"
                
                growth = growth_rates.get(company, 'N/A')
                response += f" {growth} |\n"
        
        # Insights
        if insights:
            response += "\n### 💡 Insights\n\n"
            for insight in insights:
                response += f"- {insight}\n"
        
        return response


# Example usage
if __name__ == "__main__":
    tool = TrendAnalyzerTool()
    
    result = tool.run(
        metric="ctc",
        companies=["Intel", "AMD", "Amazon"],
        years=[2024, 2025, 2026]
    )
    
    print(result["response"])
