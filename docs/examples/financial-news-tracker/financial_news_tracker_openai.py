#!/usr/bin/env python3
"""
Financial News Tracker - A tool to fetch and analyze financial news using OpenAI's API.
This tool provides real-time financial market insights, news summaries, and market analysis.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """Model for representing a single financial news item."""

    headline: str = Field(description="The news headline")
    summary: str = Field(description="Brief summary of the news")
    impact: str = Field(
        description="Potential market impact: HIGH, MEDIUM, LOW, or NEUTRAL"
    )
    sectors_affected: List[str] = Field(
        description="List of sectors/companies affected"
    )
    source: str = Field(description="News source")


class MarketAnalysis(BaseModel):
    """Model for financial market analysis."""

    market_sentiment: str = Field(
        description="Overall market sentiment: BULLISH, BEARISH, or NEUTRAL"
    )
    key_drivers: List[str] = Field(description="Key factors driving the market")
    risks: List[str] = Field(description="Current market risks")
    opportunities: List[str] = Field(description="Potential market opportunities")


class FinancialNewsResult(BaseModel):
    """Model for the complete financial news result."""

    query_topic: str = Field(description="The topic/query that was searched")
    time_period: str = Field(description="Time period covered by the news")
    summary: str = Field(description="Executive summary of the financial news")
    news_items: List[NewsItem] = Field(description="List of relevant news items")
    market_analysis: MarketAnalysis = Field(description="Overall market analysis")
    recommendations: List[str] = Field(
        description="Investment recommendations or insights"
    )


class FinancialNewsTracker:
    """A class to interact with OpenAI API for financial news tracking."""

    API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o"

    # Available OpenAI models
    AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FinancialNewsTracker with API key.

        Args:
            api_key: OpenAI API key. If None, will try to read from environment.
        """
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "API key not found. Please provide via argument or environment variable OPENAI_API_KEY."
            )

    def _get_api_key(self) -> str:
        """
        Try to get API key from environment or from a file.

        Returns:
            The API key if found, empty string otherwise.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            return api_key

        for key_file in ["openai_api_key", ".openai_api_key"]:
            key_path = Path(key_file)
            if key_path.exists():
                try:
                    return key_path.read_text().strip()
                except Exception:
                    pass

        return ""

    def get_financial_news(
        self, query: str, time_range: str = "24h", model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """
        Fetch financial news based on the query.

        Args:
            query: The financial topic or query (e.g., "tech stocks", "S&P 500", "cryptocurrency")
            time_range: Time range for news (e.g., "24h", "1w", "1m")
            model: The OpenAI model to use

        Returns:
            The parsed response containing financial news and analysis.
        """
        if not query or not query.strip():
            return {
                "error": "Query is empty. Please provide a financial topic to search."
            }

        system_prompt = """You are a professional financial analyst with expertise in market research and news analysis. 
        Your task is to provide comprehensive financial news updates and market analysis. 
        Focus on accuracy, relevance, and actionable insights. Always provide balanced analysis.
        
        Please respond with a JSON object that follows this structure:
        {
            "query_topic": "the topic searched",
            "time_period": "time period description",
            "summary": "executive summary of financial news",
            "news_items": [
                {
                    "headline": "news headline",
                    "summary": "brief summary",
                    "impact": "HIGH/MEDIUM/LOW/NEUTRAL",
                    "sectors_affected": ["sector1", "sector2"],
                    "source": "news source"
                }
            ],
            "market_analysis": {
                "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
                "key_drivers": ["driver1", "driver2"],
                "risks": ["risk1", "risk2"],
                "opportunities": ["opportunity1", "opportunity2"]
            },
            "recommendations": ["recommendation1", "recommendation2"]
        }"""

        time_context = self._get_time_context(time_range)

        user_prompt = f"""Provide a comprehensive financial news update and analysis for: {query}

Time period: {time_context}

Please include:
1. Recent relevant news items with their potential market impact
2. Overall market sentiment and analysis
3. Key market drivers and risks
4. Sectors or companies most affected
5. Investment insights or recommendations

Focus on the most significant and recent developments. Respond with valid JSON only."""

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
        }

        try:
            response = requests.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if (
                "choices" in result
                and result["choices"]
                and "message" in result["choices"][0]
            ):
                content = result["choices"][0]["message"]["content"]

                try:
                    # Try to parse as JSON first
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from the response
                    parsed = self._parse_response(content)
                    return parsed

            return {"error": "Unexpected API response format", "raw_response": result}

        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def _get_time_context(self, time_range: str) -> str:
        """
        Convert time range shorthand to descriptive context.

        Args:
            time_range: Time range string (e.g., "24h", "1w")

        Returns:
            Descriptive time context string.
        """
        now = datetime.now()

        if time_range == "24h":
            return "Last 24 hours"
        elif time_range == "1w":
            return "Last 7 days"
        elif time_range == "1m":
            return "Last 30 days"
        elif time_range == "3m":
            return "Last 3 months"
        elif time_range == "1y":
            return "Last year"
        else:
            return f"Recent period ({time_range})"

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """
        Parse the response content to extract structured information.

        Args:
            content: The response content from the API

        Returns:
            A dictionary with parsed information.
        """
        try:
            # Try to extract JSON if present
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_content)
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_content)
            else:
                # Fallback to returning raw content
                return {"raw_response": content}
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": content}


def display_results(results: Dict[str, Any], format_json: bool = False):
    """
    Display the financial news results in a human-readable format.

    Args:
        results: The financial news results dictionary
        format_json: Whether to display the results as formatted JSON
    """
    if "error" in results:
        print(f"Error: {results['error']}")
        if "raw_response" in results:
            print("\nRaw response:")
            print(results["raw_response"])
        return

    if format_json:
        print(json.dumps(results, indent=2))
        return

    if "query_topic" in results:
        print(f"\n📊 FINANCIAL NEWS REPORT: {results['query_topic']}")
        print(f"📅 Period: {results.get('time_period', 'Recent')}")

        if "summary" in results:
            print(f"\n📝 EXECUTIVE SUMMARY:")
            print(f"{results['summary']}\n")

        if "market_analysis" in results:
            analysis = results["market_analysis"]
            sentiment = analysis.get("market_sentiment", "UNKNOWN")
            sentiment_emoji = (
                "🐂"
                if sentiment == "BULLISH"
                else "🐻" if sentiment == "BEARISH" else "⚖️"
            )

            print(f"📈 MARKET ANALYSIS:")
            print(f"  Sentiment: {sentiment_emoji} {sentiment}")

            if "key_drivers" in analysis and analysis["key_drivers"]:
                print(f"\n  Key Drivers:")
                for driver in analysis["key_drivers"]:
                    print(f"    • {driver}")

            if "risks" in analysis and analysis["risks"]:
                print(f"\n  ⚠️  Risks:")
                for risk in analysis["risks"]:
                    print(f"    • {risk}")

            if "opportunities" in analysis and analysis["opportunities"]:
                print(f"\n  💡 Opportunities:")
                for opportunity in analysis["opportunities"]:
                    print(f"    • {opportunity}")

        if "news_items" in results and results["news_items"]:
            print(f"\n📰 KEY NEWS ITEMS:")
            for i, item in enumerate(results["news_items"], 1):
                impact = item.get("impact", "UNKNOWN")
                impact_emoji = (
                    "🔴"
                    if impact == "HIGH"
                    else (
                        "🟡"
                        if impact == "MEDIUM"
                        else "🟢" if impact == "LOW" else "⚪"
                    )
                )

                print(f"\n{i}. {item.get('headline', 'No headline')}")
                print(f"   Impact: {impact_emoji} {impact}")
                print(f"   Summary: {item.get('summary', 'No summary')}")

                if "sectors_affected" in item and item["sectors_affected"]:
                    print(f"   Sectors: {', '.join(item['sectors_affected'])}")

                if "source" in item:
                    print(f"   Source: {item['source']}")

        if "recommendations" in results and results["recommendations"]:
            print(f"\n💼 INSIGHTS & RECOMMENDATIONS:")
            for rec in results["recommendations"]:
                print(f"  • {rec}")

    elif "raw_response" in results:
        print("\n📊 FINANCIAL NEWS ANALYSIS:")
        print(results["raw_response"])


def main():
    """Main entry point for the financial news tracker CLI."""
    parser = argparse.ArgumentParser(
        description="Financial News Tracker - Fetch and analyze financial news using OpenAI API"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Financial topic to search (e.g., 'tech stocks', 'S&P 500', 'cryptocurrency', 'AAPL')",
    )

    parser.add_argument(
        "-t",
        "--time-range",
        type=str,
        default="24h",
        choices=["24h", "1w", "1m", "3m", "1y"],
        help="Time range for news (default: 24h)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=FinancialNewsTracker.DEFAULT_MODEL,
        choices=FinancialNewsTracker.AVAILABLE_MODELS,
        help=f"OpenAI model to use (default: {FinancialNewsTracker.DEFAULT_MODEL})",
    )

    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="OpenAI API key (if not provided, will look for environment variable OPENAI_API_KEY)",
    )

    parser.add_argument(
        "-j", "--json", action="store_true", help="Output results as JSON"
    )

    args = parser.parse_args()

    try:
        tracker = FinancialNewsTracker(api_key=args.api_key)

        print(f"Fetching financial news for '{args.query}'...", file=sys.stderr)

        results = tracker.get_financial_news(
            query=args.query, time_range=args.time_range, model=args.model
        )

        display_results(results, format_json=args.json)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
