# RAG System Evaluation Report Guide

## 📋 Overview

This document explains the evaluation metrics and how to interpret the results from `evaluate.py`. The evaluation system tests your RAG chatbot with 50 carefully designed questions to measure its performance across multiple dimensions.


## 🎯 What Gets Evaluated

### Question Categories (50 Total)

#### 1. **Simple Factual Questions** (20 questions)
- **Purpose:** Test basic information retrieval
- **Examples:**
  - "What is ICAO?"
  - "What does ATC stand for?"
  - "What is the definition of flight planning?"
- **Target:** >90% accuracy expected

#### 2. **Applied Questions** (20 questions)
- **Purpose:** Test procedural and operational knowledge
- **Examples:**
  - "How do you calculate the required fuel for a flight?"
  - "What steps are involved in filing a flight plan?"
  - "How do you select an alternate aerodrome?"
- **Target:** >75% accuracy expected

#### 3. **Higher-Order Reasoning Questions** (10 questions)
- **Purpose:** Test multi-step reasoning and decision-making
- **Examples:**
  - "What trade-offs should be considered when choosing between a direct route and airways?"
  - "How would weather conditions affect the choice of alternate aerodrome?"
- **Target:** >60% accuracy expected (most challenging)

---

## 📊 Key Metrics Explained

### 1. Retrieval Hit Rate
**What it measures:** Did the system find relevant information?

**How it's calculated:**
- Extracts key terms from the generated answer
- Checks if those terms appear in the retrieved chunks
- Considers it a "hit" if ≥20% of answer terms are found in chunks

**Interpretation:**
- **>85%** = Excellent retrieval system ✅
- **70-85%** = Good, but room for improvement 📈
- **<70%** = Poor retrieval, needs optimization ❌

---

### 2. Faithfulness Rate
**What it measures:** Is the answer grounded in the source documents?

**How it's calculated:**
- Splits answer into sentences
- Checks each sentence against retrieved chunk content
- Sentence is "grounded" if ≥30% of its words appear in chunks
- Answer is "faithful" if ≥60% of sentences are grounded

**Interpretation:**
- **>90%** = Excellent, LLM staying true to source ✅
- **75-90%** = Good, mostly grounded 📈
- **<75%** = LLM adding too much external knowledge ❌


---

### 3. Hallucination Rate
**What it measures:** Is the LLM making things up?

**How it's calculated:**
- Checks if answer is unfaithful (see above)
- Looks for uncertain language: "I think", "probably", "might be", "typically"
- Flags answers with content not supported by chunks

**Interpretation:**
- **<10%** = Excellent reliability ✅
- **10-20%** = Acceptable, some caution needed 📈
- **>20%** = High risk, major concern ❌

---

### 4. No Answer Rate
**What it measures:** How often does the system say "information not available"?

**Interpretation:**
- **<15%** = Good document coverage ✅
- **15-30%** = Moderate coverage gaps 📈
- **>30%** = Poor document coverage ❌

---

## 🏆 Best Answers Section

Shows the **top 5 best-performing answers** where the system excelled:
- ✓ Retrieved relevant chunks
- ✓ Generated faithful answer
- ✓ No hallucinations detected

---

## ⚠️ Worst Answers Section

Shows the **top 5 worst-performing answers** where the system struggled:
- ✗ Failed retrieval or
- ✗ Unfaithful answer or
- ✗ Hallucinated content

---

## 📁 Output Files

### `evaluation_report.txt`
- Human-readable report
- Summary metrics
- Best/worst answers
- Category breakdown

### `evaluation_report.json`
- Machine-readable format
- Detailed results for each question
- Useful for programmatic analysis
- Can be imported into dashboards

---

