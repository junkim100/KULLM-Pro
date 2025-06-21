# Long Content Processing Guide

This guide explains how to configure and use KULLM-Pro for processing extremely long system prompts and input data that requires code switching.

## 📋 Overview

KULLM-Pro supports automatic chunking and processing of long content that exceeds OpenAI API limits. This is particularly useful when:

- Your system prompt is very long (like the sophisticated linguistic agent prompt)
- Your input data contains extremely long mathematical reasoning traces
- You need to maintain context and coherence across chunks

## ⚙️ Configuration

### Basic Long Content Configuration

```yaml
# OpenAI Configuration for Long Content
openai:
  model: "o4-mini-2025-04-16"
  max_tokens: 100000
  
  # Long Content Handling
  max_input_length: 150000      # 150K characters max per request
  chunk_size: 100000            # 100K character chunks
  overlap_size: 10000           # 10K character overlap
  enable_chunking: true         # Enable automatic chunking
  
  # Reduced settings for stability
  batch_size: 25                # Smaller batches for long content
  max_concurrent_requests: 3    # Conservative concurrency
  timeout: 900                  # 15 minutes timeout
```

### Advanced Configuration

For extremely long content, use the provided configuration:

```bash
# Use the long content optimized configuration
python code_switch.py run "GAIR/LIMO" \
  --split="train" \
  --n=100 \
  --config_file="examples/config_long_content.yaml"
```

## 🔧 How Chunking Works

### Automatic Chunking

When content exceeds `max_input_length`, KULLM-Pro automatically:

1. **Splits content** into chunks of `chunk_size` characters
2. **Adds overlap** of `overlap_size` characters between chunks
3. **Finds smart boundaries** at sentence/paragraph breaks
4. **Processes each chunk** with context preservation
5. **Combines results** maintaining coherence

### Smart Boundary Detection

The chunking algorithm looks for natural break points:
- Sentence endings (`.`, `!`, `?`)
- Paragraph breaks (`\n\n`)
- Section boundaries
- Mathematical equation breaks

### Context Preservation

Each chunk includes:
- The full system prompt
- Chunk number and total chunks
- Overlap from previous chunk
- Context markers for continuity

## 📊 Example Usage

### Long System Prompt

Your system prompt (`system_prompt.txt`) can be extremely long:

```text
## **1. Identity and Goal**

You are a sophisticated linguistic agent, an expert in comparative linguistics...
[8000+ characters of detailed instructions]
...
```

### Long Input Data

Process datasets with very long reasoning traces:

```python
# Example of extremely long mathematical reasoning
long_reasoning = """
To solve this complex mathematical problem, we need to consider multiple approaches...
[50,000+ characters of detailed mathematical reasoning]
...
"""
```

### Processing Command

```bash
# Process with automatic chunking
python code_switch.py run "GAIR/LIMO" \
  --split="train" \
  --n=500 \
  --config_file="examples/config_long_content.yaml" \
  --system_prompt="your_long_system_prompt.txt"
```

## 🎯 Best Practices

### 1. System Prompt Design

- **Structure clearly**: Use headers and sections
- **Include chunk instructions**: Add guidance for partial processing
- **Maintain consistency**: Use consistent terminology throughout

### 2. Chunk Size Optimization

```yaml
# For very long content (100K+ chars)
chunk_size: 100000
overlap_size: 10000

# For moderately long content (50K+ chars)
chunk_size: 50000
overlap_size: 5000

# For shorter content (20K+ chars)
chunk_size: 20000
overlap_size: 2000
```

### 3. Quality Assurance

- **Monitor chunk boundaries**: Check logs for boundary quality
- **Validate continuity**: Ensure logical flow across chunks
- **Review combined results**: Check final output coherence

### 4. Performance Optimization

```yaml
# Memory management
clear_cache_frequency: 10
max_memory_usage_gb: 24

# Processing stability
parallel_chunk_processing: false
chunk_processing_delay: 1.0
```

## 📈 Monitoring and Debugging

### Logging Configuration

```yaml
logging:
  level: "INFO"
  file: "logs/long_content_processing.log"
  log_chunk_details: true
  log_token_usage: true
  log_processing_times: true
```

### Key Metrics to Monitor

- **Chunk count**: Number of chunks created
- **Success rate**: Percentage of successful chunks
- **Token usage**: Total tokens across all chunks
- **Processing time**: Time per chunk and total time
- **Memory usage**: Peak memory consumption

### Common Issues and Solutions

#### Issue: Chunks too small/large
```yaml
# Solution: Adjust chunk size
chunk_size: 75000  # Increase/decrease as needed
```

#### Issue: Poor boundary detection
```yaml
# Solution: Enable smart chunking
smart_chunking: true
preserve_context: true
```

#### Issue: Context loss between chunks
```yaml
# Solution: Increase overlap
overlap_size: 15000  # Larger overlap for better context
```

#### Issue: API timeouts
```yaml
# Solution: Increase timeout and reduce concurrency
timeout: 1200  # 20 minutes
max_concurrent_requests: 2
```

## 🔍 Example Output

### Chunk Processing Log
```
2025-06-21 15:30:00 - INFO - Content is long (103158 chars), using chunked processing
2025-06-21 15:30:00 - INFO - Split content into 3 chunks (original: 103158 chars)
2025-06-21 15:30:00 - INFO - Processing chunk 1/3
2025-06-21 15:30:15 - INFO - Processing chunk 2/3
2025-06-21 15:30:30 - INFO - Processing chunk 3/3
2025-06-21 15:30:45 - INFO - Combined 3 chunks successfully
```

### Combined Result Structure
```json
{
  "success": true,
  "response": "Combined response from all chunks...",
  "usage": {
    "total_tokens": 15000,
    "prompt_tokens": 12000,
    "completion_tokens": 3000
  },
  "chunks_processed": 3
}
```

## 🚀 Advanced Features

### Batch API with Chunking

Long content is automatically handled in batch processing:

```bash
# Batch processing with chunking
python code_switch.py run "GAIR/LIMO" \
  --split="train" \
  --n=1000 \
  --config_file="examples/config_long_content.yaml" \
  --use_batch_api=true
```

### Custom Chunking Strategies

Implement custom chunking for specific content types:

```python
# Custom chunking for mathematical content
def custom_math_chunking(content):
    # Split at equation boundaries
    # Preserve mathematical context
    # Return optimized chunks
    pass
```

## 📋 Troubleshooting Checklist

- [ ] System prompt length is reasonable (< 50K chars recommended)
- [ ] Chunk size is appropriate for content type
- [ ] Overlap size provides sufficient context
- [ ] Timeout settings accommodate processing time
- [ ] Memory limits are sufficient for chunk processing
- [ ] API rate limits are configured conservatively
- [ ] Logging is enabled for debugging
- [ ] Error handling is configured appropriately

## 📞 Support

For issues with long content processing:

1. Check the processing logs for detailed error information
2. Verify your configuration matches the content characteristics
3. Test with smaller content first to isolate issues
4. Monitor system resources during processing
5. Adjust chunk parameters based on results

---

**Note**: Processing extremely long content may take significantly longer and consume more API tokens. Monitor costs and processing times accordingly.
