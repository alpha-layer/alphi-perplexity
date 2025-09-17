# Financial News Tracker - OpenAI Version

This is a modified version of the Financial News Tracker that uses OpenAI's API instead of Perplexity's Sonar API.

## Key Differences from Perplexity Version

### API Changes
- **API Endpoint**: Changed from `https://api.perplexity.ai/chat/completions` to `https://api.openai.com/v1/chat/completions`
- **API Key**: Uses `OPENAI_API_KEY` environment variable instead of `PPLX_API_KEY`
- **Models**: Uses OpenAI models (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`) instead of Perplexity models

### Removed Features
- **Citations**: OpenAI doesn't provide source citations like Perplexity's Sonar API
- **Structured Output**: Removed the `--structured-output` flag as it's not available in OpenAI's API
- **Real-time Search**: OpenAI doesn't have built-in real-time search capabilities like Perplexity

### Enhanced Features
- **JSON Response**: The system prompt now explicitly requests JSON responses for better parsing
- **Temperature Control**: Added temperature parameter (0.3) for more consistent responses
- **Token Limits**: Added max_tokens parameter (4000) to control response length

## Installation

### 1. Install required dependencies

```bash
# Install from requirements file
pip install -r requirements_openai.txt

# Or install manually
pip install requests pydantic
```

### 2. Make the script executable

```bash
chmod +x financial_news_tracker_openai.py
```

## API Key Setup

The tool requires an OpenAI API key. You can provide it in one of these ways:

### 1. As an environment variable (recommended)

```bash
export OPENAI_API_KEY=your-openai-api-key-here
```

### 2. As a command-line argument

```bash
./financial_news_tracker_openai.py "tech stocks" --api-key your-openai-api-key-here
```

### 3. In a file

Create a file named `openai_api_key` or `.openai_api_key` in the same directory:

```bash
echo "your-openai-api-key-here" > .openai_api_key
chmod 600 .openai_api_key
```

## Usage Examples

### Basic usage - Get news for a specific topic

```bash
./financial_news_tracker_openai.py "S&P 500"
```

### Get cryptocurrency news from the past week

```bash
./financial_news_tracker_openai.py "cryptocurrency" --time-range 1w
```

### Track specific company news

```bash
./financial_news_tracker_openai.py "AAPL Apple stock"
```

### Use a different model

```bash
./financial_news_tracker_openai.py "Federal Reserve interest rates" --model gpt-4o-mini
```

### Output as JSON for programmatic use

```bash
./financial_news_tracker_openai.py "inflation rates" --json
```

## Available Models

- `gpt-4o` (default) - Most capable model
- `gpt-4o-mini` - Faster and cheaper option
- `gpt-4-turbo` - Good balance of capability and speed
- `gpt-3.5-turbo` - Fastest and most cost-effective

## Important Notes

### Limitations
- **No Real-time Data**: Unlike Perplexity's Sonar API, OpenAI doesn't have access to real-time web data. The responses are based on the model's training data, which has a cutoff date.
- **No Source Citations**: OpenAI doesn't provide source citations for the information it generates.
- **Cost**: OpenAI API calls are typically more expensive than Perplexity's API.

### Best Practices
1. **Be Specific**: Include company tickers, sector names, or specific events
2. **Combine Topics**: Mix company names with relevant themes (e.g., "TSLA electric vehicles")
3. **Use Appropriate Models**: Choose the model based on your needs:
   - `gpt-4o` for most comprehensive analysis
   - `gpt-4o-mini` for faster, cheaper responses
   - `gpt-3.5-turbo` for simple queries

## Error Handling

The tool includes comprehensive error handling for:
- Invalid API keys
- Network connectivity issues
- API rate limits
- Invalid queries
- JSON parsing errors

## Cost Considerations

OpenAI API pricing varies by model:
- `gpt-4o`: Most expensive but most capable
- `gpt-4o-mini`: Good balance of cost and capability
- `gpt-3.5-turbo`: Most cost-effective for simple queries

Monitor your usage in the OpenAI dashboard to avoid unexpected charges.

## Migration from Perplexity Version

If you're migrating from the Perplexity version:

1. **Change API Key**: Set `OPENAI_API_KEY` instead of `PPLX_API_KEY`
2. **Update Model Names**: Use OpenAI model names instead of Perplexity models
3. **Remove Structured Output**: The `--structured-output` flag is no longer available
4. **Expect Different Results**: Responses may differ due to different training data and capabilities

## Example Output

The output format remains the same as the Perplexity version, but the content will be based on OpenAI's training data rather than real-time web search results.

```
📊 FINANCIAL NEWS REPORT: tech stocks
📅 Period: Last 24 hours

📝 EXECUTIVE SUMMARY:
Based on recent market trends and developments in the technology sector...

📈 MARKET ANALYSIS:
  Sentiment: 🐂 BULLISH

  Key Drivers:
    • Strong earnings reports from major tech companies
    • Continued growth in cloud computing and AI sectors
    • Positive regulatory environment

  ⚠️ Risks:
    • Market volatility concerns
    • Regulatory scrutiny
    • Competition in key markets

  💡 Opportunities:
    • AI and machine learning investments
    • Cloud infrastructure growth
    • Cybersecurity demand

📰 KEY NEWS ITEMS:

1. Major Tech Companies Report Strong Q4 Results
   Impact: 🔴 HIGH
   Summary: Leading technology companies have reported better-than-expected earnings...
   Sectors: Technology, Software, Cloud Computing
   Source: Market Analysis

💼 INSIGHTS & RECOMMENDATIONS:
  • Consider diversifying within the tech sector
  • Focus on companies with strong AI capabilities
  • Monitor regulatory developments
```
