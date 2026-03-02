# Production AI Systems Architecture Guide
## 5 Projects for Building Robust, Enterprise-Ready AI Applications

---

## Overview

This guide covers five interconnected projects that build progressively sophisticated AI systems. Together, they demonstrate the full lifecycle of production AI development: from basic RAG pipelines through local optimization, observability, model customization, and real-time systems.

**Audience:** Engineers, product managers, and architects building AI-first products  
**Tone:** Technical, opinionated, concrete recommendations

---

## Project 1: Production RAG Application ("Ask My Docs")

### What It Is
A domain-specific document Q&A system that goes beyond basic chatbots. Implements hybrid retrieval, reranking, citation enforcement, and automated evaluation—the most common enterprise AI pattern in 2026.

### Architecture Overview

```
User Query
    ↓
┌─────────────────────────────────────────────────┐
│ Hybrid Retrieval Layer                          │
├──────────────────────┬──────────────────────────┤
│  Vector Search       │  Sparse Retrieval (BM25) │
│  (Dense embeddings)  │  (Keyword matching)      │
├──────────────────────┴──────────────────────────┤
│ Reciprocal Rank Fusion (RRF) - Merge Results    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Cross-Encoder Reranking                         │
│ (Cohere Rerank API or local cross-encoder)      │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ LLM with Citation Enforcement                   │
│ (Structured output: [answer, citations[]])      │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ FastAPI Backend + Streamlit/Gradio UI           │
│ Display: answer + source document highlighting  │
└─────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Tools |
|-------|-------|
| **Orchestration** | LangChain / LangGraph |
| **Vector Database** | Pinecone, Weaviate, or ChromaDB |
| **Sparse Retrieval** | BM25Okapi (rank_bm25), Elasticsearch |
| **Embeddings** | OpenAI text-embedding-3-large, Cohere embed-v3, HuggingFace BAAI/bge |
| **Reranking** | Cohere Rerank API, cross-encoder (ms-marco-MiniLM) |
| **LLM** | GPT-4o, Claude 3.5, or Mistral |
| **Evaluation** | RAGAS (faithfulness, answer relevance, context precision) |
| **Backend API** | FastAPI |
| **Frontend** | Streamlit or Gradio |
| **Document Parsing** | LangChain loaders, PyMuPDF, Unstructured.io |
| **CI/CD** | GitHub Actions |

### Implementation Tasks

1. **Setup & Ingestion**
   - Choose domain (legal docs, medical papers, company policy)
   - Implement pipeline: load PDFs → chunk with overlap → generate embeddings → store in vector DB
   - Implement BM25 index in parallel for sparse retrieval

2. **Hybrid Retrieval**
   - Build retriever running vector search and BM25 simultaneously
   - Implement Reciprocal Rank Fusion (RRF) to merge results
   - Test ranking quality on validation set

3. **Cross-Encoder Reranking**
   - Pass top-N hybrid results through reranker
   - Return top-K reranked results to LLM
   - Measure reranking impact on retrieval precision

4. **Citation Enforcement**
   - Prompt LLM to cite source chunks
   - Use Pydantic structured output to enforce format: `{answer: str, citations: [str]}`
   - Validate citations reference actual retrieved documents

5. **CI-Gated Evaluation**
   - Write RAGAS test suite measuring faithfulness, answer relevance, context recall
   - Set threshold scores (e.g., faithfulness ≥ 0.85)
   - Hook into GitHub Actions CI—fail pipeline on metric regression

6. **API & UI Deployment**
   - Build FastAPI endpoint `/ask` accepting query, returning answer + citations
   - Build Streamlit/Gradio UI with source document highlighting
   - Add logging and error handling

### Key Success Metrics

- **Retrieval Precision@K:** % of top-K results containing answer-relevant content
- **Faithfulness:** Are answers grounded in retrieved documents? (RAGAS metric)
- **Citation Accuracy:** % of cited chunks actually supporting the answer
- **Latency:** p50/p95 query response time
- **User Satisfaction:** Relevance scores on manual evaluation set

### Deliverables

- README with architecture diagram, baseline metrics, retrieval comparison analysis
- Runnable FastAPI server + Streamlit UI
- RAGAS evaluation harness with thresholds
- GitHub Actions CI pipeline with regression gates

---

## Project 2: Local SLM App with Ollama

### What It Is
Run small language models entirely offline on local hardware. Benchmark 3 models head-to-head on identical tasks. Demonstrates understanding of privacy, latency, and cost trade-offs critical to enterprise teams.

### Architecture Overview

```
User Input (UI / CLI)
    ↓
┌──────────────────────────────────────┐
│ FastAPI Backend                      │
├──────────────────────────────────────┤
│ • Orchestrates inference calls       │
│ • Enforces structured output         │
│ • Tracks metrics per request         │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Ollama Runtime                       │
├──────────────────────────────────────┤
│ Model A (GGUF Q4)  │ Model B (GGUF Q5) │
│ Model C (GGUF Q8)  │                   │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Metrics Collection                   │
├──────────────────────────────────────┤
│ • Tokens/sec, TTFT, total latency    │
│ • Peak/avg RAM usage                 │
│ • Output quality (ROUGE, LLM judge)  │
└──────────────────────────────────────┘
    ↓
Streamlit Dashboard / Benchmark Report
```

### Tech Stack

| Layer | Tools |
|-------|-------|
| **Model Runtime** | Ollama |
| **Models to Test** | Mistral 7B, Llama 3.2 (3B/8B), Phi-3 Mini, Gemma 2B |
| **Backend API** | FastAPI |
| **Structured Output** | Instructor (Pydantic-based LLM output validation) |
| **Quantization** | GGUF (Q4, Q5, Q8) via llama.cpp |
| **Benchmarking** | Python time, psutil, custom benchmark harness |
| **Frontend** | Streamlit or CLI |
| **Experiment Tracking** | Optional: Weights & Biases |

### Implementation Tasks

1. **Environment Setup**
   - Install Ollama locally
   - Pull 3 models: `ollama pull mistral`, `ollama pull llama3.2`, `ollama pull phi3`
   - Verify all models run correctly on target hardware

2. **Application Core**
   - Choose use case: offline summarization, local code assistant, private medical Q&A
   - Wrap Ollama calls in FastAPI service
   - Use Instructor library to enforce Pydantic schema outputs

3. **Benchmark Harness**
   - Create test dataset: 20–50 diverse prompts
   - For each model, measure:
     - Tokens/second throughput
     - Time-to-first-token (TTFT)
     - Total latency
     - Peak RAM usage
   - Score output quality (human evaluation or ROUGE/BLEU)

4. **Quantization Analysis**
   - Run same model at Q4, Q5, Q8 levels
   - Document quality degradation vs. speed improvement
   - Create trade-off curves: quality vs. model size vs. speed

5. **Comparison Report**
   - Build table: Model × Metrics (speed, quality, size, RAM)
   - Write conclusions: which model wins under different constraints
   - Include example outputs from each model

6. **Packaging**
   - Add Makefile or shell script to reproduce benchmark
   - Document hardware specs used
   - Include model download instructions

### Key Success Metrics

- **Throughput:** Tokens per second (higher is better)
- **Time-to-First-Token:** Latency to start responding (lower is better)
- **Output Quality:** Task-specific metric (F1, ROUGE, manual score)
- **Memory Efficiency:** Peak RAM vs. model size
- **Cost Per Request:** Operational cost per query on given hardware

### Deliverables

- Benchmark harness (Python script, 500+ lines)
- Comparison table with concrete numbers
- Streamlit dashboard showing per-model metrics
- README with hardware specs, methodology, and conclusions
- Makefile for reproducibility

---

## Project 3: Monitoring & Observability for AI Systems

### What It Is
Add production-grade tracing, latency tracking, cost monitoring, and quality metrics to Project 1 (RAG system). "70% of real production AI work"—almost nobody includes observability in portfolios.

### Architecture Overview

```
┌──────────────────────────────────────┐
│ Production RAG Pipeline              │
│ (from Project 1)                     │
└──────────────────────────────────────┘
         ↓↓↓ instrumented ↓↓↓
         
┌──────────────────────────────────────┐
│ Langfuse / LangSmith Tracing         │
├──────────────────────────────────────┤
│ Per-request capture:                 │
│ • Retrieval latency                  │
│ • LLM latency                        │
│ • Total latency                      │
│ • Token counts & cost estimate       │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│ Metrics Pipeline                     │
├──────────────────────────────────────┤
│ Latency: p50, p95, p99 percentiles   │
│ Cost: $ per request, daily spend     │
│ Quality: RAGAS scores (nightly batch)│
│ Errors: failure rate, error types    │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│ Dashboards & Alerting                │
├──────────────────────────────────────┤
│ Grafana / Langfuse dashboards        │
│ Slack alerts on degradation          │
│ CI regression gates (GitHub Actions) │
└──────────────────────────────────────┘
```

### Tech Stack

| Layer | Tools |
|-------|-------|
| **Tracing & Observability** | Langfuse (open-source), LangSmith |
| **Metrics Backend** | Prometheus + Grafana (or Langfuse dashboards) |
| **Latency Tracking** | Python time.perf_counter(), numpy/statistics |
| **Cost Tracking** | OpenAI token usage callbacks, LangChain callbacks |
| **Quality Metrics** | RAGAS (ongoing), LLM-as-judge |
| **Alerting** | Langfuse alerts, custom Slack webhooks |
| **Log Storage** | SQLite (local) or PostgreSQL |
| **CI/CD Integration** | GitHub Actions |

### Implementation Tasks

1. **Instrumentation**
   - Add LangChain callbacks or Langfuse SDK to RAG pipeline
   - Capture per-request: retrieval latency, LLM latency, total latency, token count
   - Calculate cost estimate: `(prompt_tokens × input_price) + (completion_tokens × output_price)`

2. **Latency Analysis**
   - Collect ≥100 production requests
   - Compute p50, p95, p99 latency per stage
   - Break down: ASR (if applicable) → retrieval → reranking → LLM → formatting

3. **Cost Monitoring**
   - Track cost per request in real time
   - Aggregate daily spend and alert if exceeds budget
   - Identify high-cost queries (e.g., long contexts) for optimization

4. **Quality Dashboard**
   - Run RAGAS nightly on sampled production queries
   - Track faithfulness, answer relevance, context precision over time
   - Alert on metric drops below threshold

5. **Regression Gating in CI**
   - Define thresholds: p95 latency < 3s, faithfulness ≥ 0.82
   - Add GitHub Actions job running evaluation on every PR
   - Fail PR if metric regresses

6. **Alerting & Runbooks**
   - Set up Slack/email alerts for latency spikes, quality drops, cost overruns
   - Write runbook: "When X metric drops, do Y"
   - Include escalation paths and on-call procedures

### Key Success Metrics

- **Availability:** % of requests completing successfully
- **Latency:** p50, p95, p99 response times
- **Cost/Request:** $ per API call
- **Quality:** RAGAS faithfulness, answer relevance
- **Regression Detection:** Time to identify metric degradation

### Deliverables

- Langfuse/LangSmith integration with RAG pipeline
- Grafana dashboard with key metrics
- RAGAS evaluation harness (nightly batch)
- GitHub Actions CI pipeline with regression gates
- Slack integration with alert rules
- Runbook documentation

---

## Project 4: Fine-Tuning with LoRA & DPO

### What It Is
Fine-tune an open-source LLM for a specific task (JSON extraction, tool-calling) using parameter-efficient LoRA/QLoRA. Apply DPO for alignment/preference tuning. Show concrete before-and-after numbers.

### Architecture Overview

```
Base Model (e.g., Mistral 7B)
    ↓
┌──────────────────────────────────────┐
│ Supervised Fine-Tuning (SFT)         │
├──────────────────────────────────────┤
│ • 4-bit QLoRA quantization           │
│ • LoRA adapters on q_proj, v_proj    │
│ • Train on 500–2K task examples      │
│ • Track loss, metrics in wandb       │
└──────────────────────────────────────┘
    ↓ SFT Model
┌──────────────────────────────────────┐
│ Direct Preference Optimization (DPO) │
├──────────────────────────────────────┤
│ • Train on preference pairs          │
│   (prompt, chosen, rejected)         │
│ • Optimize for alignment             │
│ • No reward model needed             │
└──────────────────────────────────────┘
    ↓ DPO Model
┌──────────────────────────────────────┐
│ Export & Deployment                  │
├──────────────────────────────────────┤
│ • Merge LoRA into base model         │
│ • Export to GGUF (Ollama)            │
│ • OR deploy via vLLM / HF Inference  │
└──────────────────────────────────────┘
```

### Tech Stack

| Layer | Tools |
|-------|-------|
| **Base Models** | Mistral 7B, Llama 3.1 8B, Phi-3 |
| **Fine-Tuning Framework** | Hugging Face transformers + TRL |
| **Parameter Efficiency** | PEFT (LoRA / QLoRA) |
| **Quantization** | BitsAndBytes (4-bit QLoRA) |
| **Preference Tuning** | TRL DPOTrainer |
| **Training Data (SFT)** | Custom domain data + HF ultrachat_200k |
| **Training Data (DPO)** | argilla/distilabel-intel-orca-dpo-pairs |
| **Experiment Tracking** | Weights & Biases (wandb) |
| **Hardware** | 1× A100/RTX 4090 (or Google Colab) |
| **Serving** | Ollama (GGUF) or vLLM |

### Implementation Tasks

1. **Task & Dataset Definition**
   - Pick specific task: JSON extraction OR function/tool-calling
   - Collect/create 500–2K training examples in instruction-response format
   - Split: 80% train, 10% val, 10% test

2. **Baseline Evaluation**
   - Run base model (e.g., Mistral 7B) on test set
   - Record metrics: exact match %, F1 score, format compliance %
   - Capture qualitative samples for comparison

3. **Supervised Fine-Tuning (SFT)**
   - Load base model with 4-bit quantization (BitsAndBytes)
   - Apply LoRA adapters (target q_proj, v_proj)
   - Train with `trl.SFTTrainer`
   - Log loss curves and eval metrics to wandb
   - Save best checkpoint based on validation metric

4. **DPO Preference Tuning**
   - Create preference dataset: (prompt, chosen_response, rejected_response)
   - Use `trl.DPOTrainer` on top of SFT model
   - Tune beta, lr, and num_train_epochs
   - Compare DPO model vs. SFT-only on test set

5. **Before-and-After Comparison**
   - Run identical test set on: Base → SFT → DPO models
   - Create comparison table with metrics and examples
   - Highlight qualitative improvements

6. **Export & Documentation**
   - Merge LoRA adapter weights into base model
   - Export to GGUF format for Ollama
   - Document VRAM requirements, training time, convergence curves
   - Include inference code examples

### Key Success Metrics

- **Exact Match %:** Correct output format and values
- **F1 Score:** Precision-recall tradeoff on extracted entities
- **Format Compliance %:** Valid JSON / function calls
- **Inference Latency:** Tokens/sec on target hardware
- **Training Efficiency:** Tokens/sec during training, VRAM usage

### Deliverables

- Training scripts (SFT + DPO) with hyperparameters
- Dataset preparation and validation code
- Before-and-after comparison table (base vs SFT vs DPO)
- Trained LoRA adapters + merged GGUF model
- Weights & Biases report with loss curves
- README with training methodology and results

---

## Project 5: Real-Time Multimodal Application (Voice Assistant)

### What It Is
Build a voice assistant with decomposed end-to-end latency budget. Implement graceful degradation, timeout handling, and WebSocket streaming. Demonstrates real-time systems design and resilience.

### Architecture Overview

```
User (speaks into microphone)
    ↓
┌──────────────────────────────────────┐
│ ASR (Deepgram WebSocket)             │
│ Budget: ≤ 200ms                      │
│ Streams interim + final transcripts  │
└──────────────────────────────────────┘
    ↓ "User said: ..."
┌──────────────────────────────────────┐
│ LLM with Streaming (GPT-4o/Claude)   │
│ Budget: ≤ 600ms TTFT                 │
│ Stream tokens as they arrive         │
└──────────────────────────────────────┘
    ↓ "The answer is..."
┌──────────────────────────────────────┐
│ TTS (ElevenLabs WebSocket)           │
│ Budget: ≤ 400ms first chunk          │
│ Stream audio in real time            │
└──────────────────────────────────────┘
    ↓ 🔊
User hears response (Total: ~1.2s)

Fallbacks & Degradation:
• If LLM timeout → play filler audio "I'm still thinking..."
• If TTS fails 3× → use cheaper OpenAI TTS
• Circuit breaker → switch to faster model on cascade failures
```

### Tech Stack

| Layer | Tools |
|-------|-------|
| **Orchestration** | Pipecat AI (open-source real-time multimodal) |
| **Speech-to-Text** | Deepgram Streaming API |
| **Text-to-Speech** | ElevenLabs (WebSocket), OpenAI TTS (fallback) |
| **LLM** | GPT-4o-mini, Claude 3 Haiku, or Mistral (streaming) |
| **Real-Time Transport** | WebSockets, WebRTC (Daily.co, Pipecat) |
| **Backend** | FastAPI + asyncio |
| **Frontend** | React or vanilla JS (WebSocket audio) |
| **Latency Measurement** | Python time.perf_counter(), per-stage timing |
| **Error Handling** | Circuit breaker, exponential backoff |

### Implementation Tasks

1. **Latency Budget Design**
   - Define target end-to-end: < 1.5s speech-in to audio-response-start
   - Break down budget:
     - ASR: ≤ 200ms
     - LLM TTFT: ≤ 600ms
     - TTS first chunk: ≤ 400ms
     - Network overhead: ≤ 100ms
   - Test on target network conditions

2. **ASR Integration**
   - Integrate Deepgram WebSocket for real-time STT
   - Stream audio chunks from frontend
   - Receive interim and final transcripts
   - Measure and log ASR latency per utterance

3. **LLM Streaming**
   - Connect transcript to LLM with `stream=True`
   - Stream tokens as they arrive (don't buffer)
   - Measure Time-to-First-Token
   - Add system prompt for concise responses

4. **TTS Streaming**
   - Pipe LLM token stream → ElevenLabs WebSocket TTS
   - Send audio chunks to frontend in real time
   - Don't buffer full response—stream immediately
   - Measure TTS latency: first token → first audio chunk

5. **Graceful Degradation**
   - Add timeout at each stage (ASR, LLM, TTS)
   - Fallback responses: "I'm still thinking..." filler audio on LLM timeout
   - Circuit breaker: if service fails 3× in a row, switch to faster fallback
   - Retry logic with exponential backoff

6. **Latency Dashboard**
   - Log per-stage latencies for every turn
   - Compute p50/p95 per stage
   - Visualize stacked bar chart: where is time spent?
   - Export metrics to Grafana or custom dashboard

7. **Demo & Documentation**
   - Record demo video
   - Write README with architecture diagram and latency budget table
   - Document degradation scenarios and circuit breaker triggers
   - Include deployment instructions

### Key Success Metrics

- **End-to-End Latency:** p50, p95 from speech end to audio start
- **Time-to-First-Token:** Latency from prompt to first LLM token
- **Availability:** % of requests completing without fallback
- **Graceful Degradation:** How often circuit breaker activates
- **User Experience:** MOS (Mean Opinion Score) if user feedback collected

### Deliverables

- FastAPI backend with full streaming pipeline
- React/JS frontend with WebSocket audio streaming
- Latency measurement and dashboarding code
- Circuit breaker and fallback logic
- Demo video showing voice assistant in action
- README with architecture, latency budget, and deployment guide

---

## How These Projects Tell a Cohesive Story

| Project | What It Demonstrates | Audience Takeaway |
|---------|----------------------|-------------------|
| **1. Production RAG** | End-to-end AI feature shipping | You can build and deploy real AI applications |
| **2. Local SLM** | Cost & privacy optimization | You understand hardware constraints and trade-offs |
| **3. Monitoring & Observability** | Production reliability | You think about operability, not just building demos |
| **4. Fine-Tuning (LoRA/DPO)** | Model customization with ROI | You can optimize models for specific tasks with measured results |
| **5. Real-Time Multimodal** | Latency budgets & resilience | You understand streaming, degradation, and real-time constraints |

**The narrative:** Start with RAG (the most common enterprise AI pattern), prove you understand cost/privacy with local models, add observability (the hardest production problem), customize models (the ROI lever), and cap with real-time resilience (the hardest scaling problem).

---

## Key Resources

### Foundational Patterns & Design
- **Eugene Yan's LLM Patterns:** https://eugeneyan.com/writing/llm-patterns/
- **LangChain RAG Docs:** https://docs.langchain.com/oss/python/langchain/rag

### Project 1: Production RAG
- **Cohere Rerank Guide:** https://docs.cohere.com/docs/rerank-guide
- **RAGAS Evaluation:** https://docs.ragas.io/
- **LangChain Documentation:** https://python.langchain.com/

### Project 2: Local SLM
- **Ollama GitHub:** https://github.com/ollama/ollama
- **HuggingFace PEFT:** https://huggingface.co/docs/peft
- **Instructor (Structured Output):** https://github.com/jxnl/instructor

### Project 3: Monitoring & Observability
- **Langfuse Observability:** https://langfuse.com/docs
- **LangSmith Tracing:** https://docs.smith.langchain.com/
- **Prometheus + Grafana:** https://prometheus.io/, https://grafana.com/

### Project 4: Fine-Tuning
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers
- **TRL (DPO Trainer):** https://huggingface.co/docs/trl
- **Weights & Biases:** https://docs.wandb.ai/

### Project 5: Real-Time Multimodal
- **Pipecat AI:** https://github.com/pipecat-ai/pipecat
- **Deepgram ASR:** https://developers.deepgram.com/
- **ElevenLabs WebSockets:** https://elevenlabs.io/docs/api-reference/websockets

---

## Quick Start Path

**Week 1–2:** Build Project 1 (RAG) end-to-end with Streamlit UI  
**Week 3:** Build Project 2 (Local SLM) with 3-model benchmark  
**Week 4:** Add Project 3 (Observability) instrumentation to Project 1  
**Week 5–6:** Implement Project 4 (LoRA/DPO) on custom domain task  
**Week 7:** Prototype Project 5 (Voice) on Pipecat or custom FastAPI  

**Portfolio outcome:** 5 full-stack projects showing enterprise AI patterns, optimization, production readiness, and real-time systems design.

---

## Summary

These five projects progress from foundational (RAG retrieval) through optimization (local models), observability (production metrics), customization (fine-tuning), and scaling (real-time systems). Together they tell the story of an engineer who can ship AI at scale: with reliability, cost-consciousness, measurable quality, and architectural sophistication.

**Target:** Senior IC or staff engineer level in AI/ML infrastructure.
