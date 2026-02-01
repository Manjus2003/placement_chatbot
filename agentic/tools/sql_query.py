"""
🗄️ SQL QUERY TOOL
=================
Queries structured placement data using SQL.

This tool:
1. Maintains a SQLite database of placement statistics
2. Converts natural language to SQL queries
3. Returns structured data for aggregations
"""

import os
import re
import json
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from .base_tool import BaseTool


class SQLQueryTool(BaseTool):
    """
    SQL-based querying of structured placement data.
    
    Uses SQLite for local storage and LLM for natural language to SQL.
    """
    
    def __init__(self, db_path: str = "placement_data.db"):
        """
        Initialize the SQL query tool.
        
        Args:
            db_path: Path to SQLite database
        """
        load_dotenv(dotenv_path="environment.env")
        
        self.db_path = db_path
        self.conn = None
        
        # Initialize LLM for NL to SQL
        self.llm_client = None
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
            except Exception:
                pass
        
        # Create database if it doesn't exist
        self._ensure_database()
        print("✅ SQLQueryTool initialized")
    
    @property
    def name(self) -> str:
        return "sql_query"
    
    @property
    def description(self) -> str:
        return "Query structured placement statistics (averages, counts, rankings)"
    
    def _ensure_database(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create placements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS placements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                year INTEGER,
                ctc_lpa REAL,
                roles TEXT,
                eligibility_cgpa REAL,
                branches TEXT,
                selection_rounds INTEGER,
                students_selected INTEGER,
                students_applied INTEGER,
                visit_date TEXT,
                category TEXT
            )
        """)
        
        # Create index for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_company ON placements(company)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_year ON placements(year)
        """)
        
        self.conn.commit()
    
    def populate_from_chunks(self, chunks_json_path: str):
        """
        Populate database from chunks JSON file.
        
        Extracts structured data from chunk metadata.
        """
        if not os.path.exists(chunks_json_path):
            print(f"⚠️ Chunks file not found: {chunks_json_path}")
            return
        
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        cursor = self.conn.cursor()
        
        # Track companies we've added
        added_companies = set()
        
        for chunk in chunks:
            company = chunk.get("company", "")
            if not company or company in added_companies:
                continue
            
            # Extract data from chunk
            text = chunk.get("text", chunk.get("content", "")).lower()
            
            # Extract CTC
            ctc = self._extract_number(text, [
                r'ctc[:\s]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*lpa',
                r'package[:\s]+(\d+(?:\.\d+)?)'
            ])
            
            # Extract CGPA
            cgpa = self._extract_number(text, [
                r'cgpa[:\s]+(\d+(?:\.\d+)?)',
                r'eligibility[:\s]+(\d+(?:\.\d+)?)',
                r'minimum[:\s]+(\d+(?:\.\d+)?)'
            ])
            
            # Extract year from metadata or folder name
            year = chunk.get("year", 2025)
            
            # Insert into database
            cursor.execute("""
                INSERT OR REPLACE INTO placements 
                (company, year, ctc_lpa, eligibility_cgpa, roles, category)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                company,
                year,
                ctc,
                cgpa,
                chunk.get("roles", ""),
                chunk.get("category", "Tech")
            ))
            
            added_companies.add(company)
        
        self.conn.commit()
        print(f"✅ Populated database with {len(added_companies)} companies")
    
    def _extract_number(self, text: str, patterns: List[str]) -> Optional[float]:
        """Extract a number using multiple patterns"""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def run(
        self,
        query: str,
        operation: str = "aggregate",
        raw_sql: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a SQL query.
        
        Args:
            query: Natural language query
            operation: Type of operation (aggregate, filter, rank)
            raw_sql: Optional raw SQL to execute directly
            
        Returns:
            Dict with:
                - result: Query result
                - sql: SQL query executed
                - rows: Number of rows returned
        """
        print(f"🗄️ SQL Query: '{query[:50]}...'")
        
        if raw_sql:
            sql = raw_sql
        else:
            sql = self._nl_to_sql(query, operation)
        
        if not sql:
            return {
                "result": None,
                "sql": None,
                "error": "Could not generate SQL query"
            }
        
        print(f"   SQL: {sql}")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            # Format result
            if len(rows) == 1 and len(columns) == 1:
                # Single value result
                result = rows[0][0]
            else:
                # Multiple rows - convert to list of dicts
                result = [dict(zip(columns, row)) for row in rows]
            
            return {
                "result": result,
                "sql": sql,
                "rows": len(rows),
                "columns": columns
            }
            
        except Exception as e:
            print(f"❌ SQL error: {e}")
            return {
                "result": None,
                "sql": sql,
                "error": str(e)
            }
    
    def _nl_to_sql(self, query: str, operation: str) -> Optional[str]:
        """Convert natural language to SQL"""
        
        # Try rule-based first
        sql = self._nl_to_sql_rules(query, operation)
        if sql:
            return sql
        
        # Fall back to LLM
        if self.llm_client:
            return self._nl_to_sql_llm(query)
        
        return None
    
    def _nl_to_sql_rules(self, query: str, operation: str) -> Optional[str]:
        """Rule-based natural language to SQL"""
        query_lower = query.lower()
        
        # Average CTC queries
        if "average" in query_lower and ("ctc" in query_lower or "salary" in query_lower or "package" in query_lower):
            if "2025" in query_lower:
                return "SELECT AVG(ctc_lpa) as avg_ctc FROM placements WHERE year = 2025 AND ctc_lpa IS NOT NULL"
            elif "2024" in query_lower:
                return "SELECT AVG(ctc_lpa) as avg_ctc FROM placements WHERE year = 2024 AND ctc_lpa IS NOT NULL"
            else:
                return "SELECT AVG(ctc_lpa) as avg_ctc FROM placements WHERE ctc_lpa IS NOT NULL"
        
        # Count queries
        if "how many" in query_lower and "companies" in query_lower:
            if "2025" in query_lower:
                return "SELECT COUNT(DISTINCT company) as company_count FROM placements WHERE year = 2025"
            else:
                return "SELECT COUNT(DISTINCT company) as company_count FROM placements"
        
        # Highest/Top CTC queries
        if ("highest" in query_lower or "top" in query_lower or "maximum" in query_lower) and ("ctc" in query_lower or "salary" in query_lower):
            limit = 10
            match = re.search(r'top\s*(\d+)', query_lower)
            if match:
                limit = int(match.group(1))
            return f"SELECT company, ctc_lpa FROM placements WHERE ctc_lpa IS NOT NULL ORDER BY ctc_lpa DESC LIMIT {limit}"
        
        # Lowest CTC queries
        if ("lowest" in query_lower or "minimum" in query_lower) and ("ctc" in query_lower or "salary" in query_lower):
            return "SELECT company, ctc_lpa FROM placements WHERE ctc_lpa IS NOT NULL ORDER BY ctc_lpa ASC LIMIT 10"
        
        # Companies with CTC above X
        match = re.search(r'ctc\s*(?:above|greater|more than|>)\s*(\d+)', query_lower)
        if match:
            threshold = float(match.group(1))
            return f"SELECT company, ctc_lpa FROM placements WHERE ctc_lpa > {threshold} ORDER BY ctc_lpa DESC"
        
        # Companies by year
        if "companies" in query_lower:
            for year in ["2023", "2024", "2025"]:
                if year in query_lower:
                    return f"SELECT company, ctc_lpa, roles FROM placements WHERE year = {year}"
        
        # All companies list
        if "all companies" in query_lower or "list" in query_lower and "companies" in query_lower:
            return "SELECT DISTINCT company FROM placements ORDER BY company"
        
        return None
    
    def _nl_to_sql_llm(self, query: str) -> Optional[str]:
        """Use LLM to convert natural language to SQL"""
        
        schema = """
Table: placements
Columns:
- id: INTEGER PRIMARY KEY
- company: TEXT (company name)
- year: INTEGER (placement year, e.g., 2025)
- ctc_lpa: REAL (CTC in LPA)
- roles: TEXT (job roles offered)
- eligibility_cgpa: REAL (minimum CGPA required)
- branches: TEXT (eligible branches)
- selection_rounds: INTEGER (number of interview rounds)
- students_selected: INTEGER
- students_applied: INTEGER
- visit_date: TEXT
- category: TEXT (Tech/Non-Tech)
"""

        prompt = f"""Convert this natural language query to SQL.

Database Schema:
{schema}

User Query: "{query}"

Rules:
1. Use SQLite syntax
2. Handle NULL values with IS NOT NULL when aggregating
3. Use DISTINCT when counting unique values
4. Limit results to 20 unless specified

Return ONLY the SQL query, no explanation."""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            sql = response.choices[0].message.content.strip()
            
            # Clean up the response
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            # Basic validation
            if sql.upper().startswith("SELECT"):
                return sql
                
        except Exception as e:
            print(f"⚠️ LLM SQL generation failed: {e}")
        
        return None
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a quick summary of the database"""
        cursor = self.conn.cursor()
        
        try:
            stats = {}
            
            cursor.execute("SELECT COUNT(DISTINCT company) FROM placements")
            stats["total_companies"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(ctc_lpa) FROM placements WHERE ctc_lpa IS NOT NULL")
            stats["avg_ctc"] = round(cursor.fetchone()[0] or 0, 2)
            
            cursor.execute("SELECT MAX(ctc_lpa) FROM placements")
            stats["max_ctc"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(ctc_lpa) FROM placements WHERE ctc_lpa IS NOT NULL")
            stats["min_ctc"] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == "__main__":
    tool = SQLQueryTool()
    
    # Test with some sample data
    conn = sqlite3.connect(tool.db_path)
    cursor = conn.cursor()
    
    # Insert sample data
    sample_data = [
        ("Amazon", 2025, 28.0, "SDE", 7.5),
        ("Google", 2025, 35.0, "SWE", 8.0),
        ("Microsoft", 2025, 25.0, "SDE", 7.0),
        ("Intel", 2025, 20.0, "Hardware Engineer", 7.5),
    ]
    
    for company, year, ctc, roles, cgpa in sample_data:
        cursor.execute("""
            INSERT OR REPLACE INTO placements 
            (company, year, ctc_lpa, roles, eligibility_cgpa)
            VALUES (?, ?, ?, ?, ?)
        """, (company, year, ctc, roles, cgpa))
    
    conn.commit()
    conn.close()
    
    # Test queries
    test_queries = [
        "What is the average CTC?",
        "How many companies visited?",
        "Show top 5 highest CTC companies",
        "Companies with CTC above 25 LPA"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = tool.run(query)
        print(f"Result: {result['result']}")
