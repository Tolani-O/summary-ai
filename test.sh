#!/bin/bash

input_path="sample_text/understanding-machine-learning-theory-algorithms.pdf"
mode="single"
log_output_path="sample_text/log.txt"
summary_output_path="sample_text/summary.txt"
recursion_limit=10
validation_threshold=85 # 85% validation threshold
agent_url="http://localhost:8000/summarize"


# Test endpoint
curl -X POST "$agent_url" \
  -H "Content-Type: application/json" \
  -d '{
        "input_path": "'"${input_path}"'",
        "mode": "'"${mode}"'",
        "log_output_path": "'"${log_output_path}"'",
        "summary_output_path": "'"${summary_output_path}"'",
        "recursion_limit": "'"${recursion_limit}"'",
        "validation_threshold": "'"${validation_threshold}"'"
      }'
