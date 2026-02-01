"""
🏢 COMPANY CLUSTERING TOOL
===========================
Groups similar companies for easier comparison.

Capabilities:
- Cluster by domain (Software, Hardware, Consulting)
- Cluster by CTC range
- Cluster by culture/values
- Find similar companies
"""

import re
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from .base_tool import BaseTool


class CompanyClusterTool(BaseTool):
    """
    Clusters companies by various attributes.
    """
    
    # Domain keywords
    DOMAIN_KEYWORDS = {
        "Software/IT": [
            "software", "cloud", "saas", "platform", "technology", "digital",
            "internet", "e-commerce", "app", "web"
        ],
        "Hardware/Semiconductor": [
            "chip", "semiconductor", "processor", "silicon", "hardware",
            "vlsi", "embedded", "electronics", "circuit"
        ],
        "Automotive": [
            "automotive", "vehicle", "mobility", "car", "automobile", "ev"
        ],
        "Finance/Banking": [
            "bank", "finance", "fintech", "investment", "trading", "capital"
        ],
        "Healthcare/Medical": [
            "healthcare", "medical", "health", "pharma", "hospital", "biotech"
        ],
        "Consulting": [
            "consulting", "advisory", "strategy", "management consulting"
        ],
        "Manufacturing": [
            "manufacturing", "industrial", "production", "factory"
        ],
        "Telecom/Networking": [
            "telecom", "network", "communication", "wireless", "5g"
        ]
    }
    
    # CTC ranges (in LPA)
    CTC_RANGES = [
        (0, 10, "Entry Level"),
        (10, 20, "Mid Range"),
        (20, 40, "High Range"),
        (40, 100, "Premium"),
        (100, float('inf'), "Top Tier")
    ]
    
    # Culture keywords
    CULTURE_KEYWORDS = {
        "Innovation-Focused": ["innovation", "research", "cutting-edge", "pioneering"],
        "Fast-Paced": ["fast-paced", "dynamic", "agile", "startup"],
        "Work-Life Balance": ["work-life", "flexible", "remote", "hybrid"],
        "Structured": ["structured", "process", "methodical", "established"],
        "Growth-Oriented": ["growth", "learning", "development", "career"]
    }
    
    def __init__(self, vector_search_tool=None):
        """
        Initialize company clustering tool.
        
        Args:
            vector_search_tool: VectorSearchTool for querying documents
        """
        self.vector_search_tool = vector_search_tool
    
    @property
    def name(self) -> str:
        return "company_cluster"
    
    @property
    def description(self) -> str:
        return "Group similar companies by domain, CTC, or culture"
    
    def run(
        self,
        cluster_by: str = "domain",
        company_name: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Cluster companies.
        
        Args:
            cluster_by: "domain", "ctc", "culture", or "similar"
            company_name: If clustering by similar, specify company name
            
        Returns:
            Dict with clusters and insights
        """
        try:
            print(f"🏢 Clustering companies by {cluster_by}...")
        
        # Get all companies
        companies = self._get_all_companies()
        
        # Perform clustering
        if cluster_by == "domain":
            clusters = self._cluster_by_domain(companies)
        elif cluster_by == "ctc":
            clusters = self._cluster_by_ctc(companies)
        elif cluster_by == "culture":
            clusters = self._cluster_by_culture(companies)
        elif cluster_by == "similar" and company_name:
            clusters = self._find_similar_companies(company_name, companies)
        else:
            clusters = {"error": "Invalid cluster_by parameter"}
        
        # Generate insights
        insights = self._generate_insights(clusters, cluster_by)
        
            return {
                "response": self._format_response(clusters, cluster_by, insights),
                "clusters": clusters,
                "insights": insights,
                "visualization": "grouped_bar_chart"
            }
        except Exception as e:
            print(f"❌ Error in company clustering: {e}")
            return {
                "response": f"## 🏢 Company Clustering Error\n\nUnable to cluster companies: {str(e)}\n\nPlease try again or contact support.",
                "clusters": {},
                "insights": {},
                "error": str(e)
            }
    
    def _get_all_companies(self) -> List[str]:
        """Get list of all companies"""
        
        if self.vector_search_tool:
            try:
                return self.vector_search_tool.get_all_companies()
            except:
                pass
        
        # Default sample companies
        return [
            "Amazon", "Google", "Microsoft", "Intel", "AMD", "Nvidia",
            "Qualcomm", "Bosch", "Samsung", "IBM", "Oracle", "SAP",
            "Dell", "HP", "Cisco", "Juniper", "Ericsson", "Nokia",
            "Goldman Sachs", "JPMorgan", "Morgan Stanley", "McKinsey",
            "BCG", "Deloitte", "Accenture"
        ]
    
    def _cluster_by_domain(self, companies: List[str]) -> Dict[str, List[str]]:
        """Cluster companies by domain"""
        
        clusters = defaultdict(list)
        
        for company in companies:
            domain = self._identify_domain(company)
            clusters[domain].append(company)
        
        return dict(clusters)
    
    def _identify_domain(self, company: str) -> str:
        """Identify company domain using keywords"""
        
        company_lower = company.lower()
        
        # Get company context
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} company profile domain industry",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "").lower()
            except:
                context = company_lower
        else:
            context = company_lower
        
        # Match against domain keywords
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in context)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "Other"
    
    def _cluster_by_ctc(self, companies: List[str]) -> Dict[str, List[str]]:
        """Cluster companies by CTC range"""
        
        clusters = defaultdict(list)
        
        for company in companies:
            ctc = self._extract_ctc(company)
            ctc_range = self._get_ctc_range(ctc)
            clusters[ctc_range].append({
                "company": company,
                "ctc": ctc
            })
        
        return dict(clusters)
    
    def _extract_ctc(self, company: str) -> float:
        """Extract CTC for company"""
        
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} CTC package salary compensation",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "")
                
                # Extract CTC using regex (looking for patterns like "15 LPA", "Rs 20 lakhs")
                patterns = [
                    r'(\d+(?:\.\d+)?)\s*LPA',
                    r'Rs\.?\s*(\d+(?:\.\d+)?)\s*lakhs',
                    r'₹\s*(\d+(?:\.\d+)?)\s*lakhs'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            except:
                pass
        
        # Return sample CTC based on company tier
        return self._get_sample_ctc(company)
    
    def _get_sample_ctc(self, company: str) -> float:
        """Return sample CTC based on company reputation"""
        
        company_lower = company.lower()
        
        # Top tier
        if any(kw in company_lower for kw in ['google', 'amazon', 'microsoft', 'nvidia']):
            return 45.0
        
        # High tier
        elif any(kw in company_lower for kw in ['intel', 'amd', 'qualcomm', 'apple']):
            return 30.0
        
        # Mid tier
        elif any(kw in company_lower for kw in ['bosch', 'samsung', 'dell', 'hp']):
            return 18.0
        
        # Default
        else:
            return 12.0
    
    def _get_ctc_range(self, ctc: float) -> str:
        """Get CTC range label"""
        
        for min_ctc, max_ctc, label in self.CTC_RANGES:
            if min_ctc <= ctc < max_ctc:
                return label
        
        return "Unknown"
    
    def _cluster_by_culture(self, companies: List[str]) -> Dict[str, List[str]]:
        """Cluster companies by culture/values"""
        
        clusters = defaultdict(list)
        
        for company in companies:
            cultures = self._identify_culture(company)
            for culture in cultures:
                clusters[culture].append(company)
        
        return dict(clusters)
    
    def _identify_culture(self, company: str) -> List[str]:
        """Identify company culture"""
        
        if self.vector_search_tool:
            try:
                result = self.vector_search_tool.run(
                    query=f"{company} culture values work environment",
                    company_filter=company,
                    top_k=1
                )
                context = result.get("context", "").lower()
            except:
                context = company.lower()
        else:
            context = company.lower()
        
        # Match against culture keywords
        matched_cultures = []
        for culture, keywords in self.CULTURE_KEYWORDS.items():
            if any(kw in context for kw in keywords):
                matched_cultures.append(culture)
        
        return matched_cultures if matched_cultures else ["General"]
    
    def _find_similar_companies(
        self,
        target_company: str,
        companies: List[str]
    ) -> Dict[str, Any]:
        """Find companies similar to target company"""
        
        # Get target company attributes
        target_domain = self._identify_domain(target_company)
        target_ctc = self._extract_ctc(target_company)
        target_cultures = self._identify_culture(target_company)
        
        # Find similar companies
        similar = []
        for company in companies:
            if company.lower() == target_company.lower():
                continue
            
            similarity_score = 0
            
            # Domain match (40 points)
            if self._identify_domain(company) == target_domain:
                similarity_score += 40
            
            # CTC similarity (30 points)
            company_ctc = self._extract_ctc(company)
            ctc_diff = abs(company_ctc - target_ctc)
            if ctc_diff < 5:
                similarity_score += 30
            elif ctc_diff < 10:
                similarity_score += 20
            elif ctc_diff < 20:
                similarity_score += 10
            
            # Culture match (30 points)
            company_cultures = self._identify_culture(company)
            culture_overlap = len(set(target_cultures) & set(company_cultures))
            similarity_score += min(30, culture_overlap * 10)
            
            if similarity_score >= 40:  # Threshold
                similar.append({
                    "company": company,
                    "similarity_score": similarity_score,
                    "domain": self._identify_domain(company),
                    "ctc": company_ctc
                })
        
        # Sort by similarity
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "target": target_company,
            "similar_companies": similar[:5]
        }
    
    def _generate_insights(
        self,
        clusters: Dict[str, Any],
        cluster_by: str
    ) -> Dict[str, Any]:
        """Generate insights from clusters"""
        
        insights = {}
        
        if cluster_by == "domain":
            # Count companies per domain
            largest_domain = max(clusters, key=lambda k: len(clusters[k]))
            insights["largest_domain"] = largest_domain
            insights["domain_count"] = len(clusters[largest_domain])
        
        elif cluster_by == "ctc":
            # Find most common range
            largest_range = max(clusters, key=lambda k: len(clusters[k]))
            insights["most_common_range"] = largest_range
            insights["companies_in_range"] = len(clusters[largest_range])
        
        elif cluster_by == "culture":
            # Find most common culture
            largest_culture = max(clusters, key=lambda k: len(clusters[k]))
            insights["most_common_culture"] = largest_culture
        
        elif cluster_by == "similar":
            if "similar_companies" in clusters:
                insights["similar_count"] = len(clusters["similar_companies"])
        
        return insights
    
    def _format_response(
        self,
        clusters: Dict[str, Any],
        cluster_by: str,
        insights: Dict[str, Any]
    ) -> str:
        """Format response for display"""
        
        response = f"## 🏢 Company Clustering: {cluster_by.title()}\n\n"
        
        if cluster_by == "similar":
            # Format similar companies
            target = clusters.get("target", "Unknown")
            response += f"### Companies Similar to {target}\n\n"
            
            for item in clusters.get("similar_companies", [])[:5]:
                response += f"**{item['company']}** (Similarity: {item['similarity_score']}%)\n"
                response += f"- Domain: {item['domain']}\n"
                response += f"- CTC: ₹{item['ctc']} LPA\n\n"
        
        else:
            # Format clusters
            for cluster_name, items in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
                response += f"### {cluster_name}\n\n"
                
                if cluster_by == "ctc":
                    # Show companies with CTC
                    for item in items[:10]:
                        company = item["company"]
                        ctc = item["ctc"]
                        response += f"- {company} (₹{ctc} LPA)\n"
                else:
                    # Show company list
                    companies_str = ", ".join(items[:10])
                    if len(items) > 10:
                        companies_str += f" ... and {len(items) - 10} more"
                    response += f"{companies_str}\n"
                
                response += "\n"
        
        # Add insights
        if insights:
            response += "### 💡 Key Insights\n\n"
            for key, value in insights.items():
                response += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        return response


# Example usage
if __name__ == "__main__":
    tool = CompanyClusterTool()
    
    # Domain clustering
    result = tool.run(cluster_by="domain")
    print(result["response"])
    
    # Similar companies
    result = tool.run(cluster_by="similar", company_name="Intel")
    print(result["response"])
