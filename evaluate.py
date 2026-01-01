"""
===========================================
RAG SYSTEM EVALUATION SCRIPT
===========================================

This script evaluates the RAG system by:
1. Testing with 50 categorized questions
2. Measuring retrieval hit-rate, faithfulness, and hallucinations
3. Generating a comprehensive evaluation report
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import re

# Import our RAG components
from vector import DocumentStore
from tech import generate_llm_response

# ========================================
# TEST QUESTIONS (50 Total)
# ========================================

# Based on "Flight planning and monitoring.pdf"
QUESTIONS = {
    "simple_factual": [
        # Direct lookups, definitions, basic facts (20 questions)
        "What is ICAO?",
        "What does ATC stand for?",
        "What is the definition of flight planning?",
        "What is a flight plan?",
        "What does VFR mean?",
        "What does IFR mean?",
        "What is NOTAM?",
        "What is the purpose of a flight plan?",
        "What is the minimum fuel reserve requirement?",
        "What is an alternate aerodrome?",
        "What is RVSM?",
        "What does ETOPS stand for?",
        "What is a SID?",
        "What is a STAR?",
        "What is the definition of cruising level?",
        "What is a waypoint?",
        "What is the definition of route?",
        "What is AIP?",
        "What is a navigation log?",
        "What is the purpose of pre-flight planning?"
    ],
    
    "applied": [
        # Scenario-based, operational, procedural (20 questions)
        "How do you calculate the required fuel for a flight?",
        "What steps are involved in filing a flight plan?",
        "When should a pilot file a flight plan?",
        "How do you determine the optimal cruising altitude?",
        "What factors affect flight planning?",
        "What should be checked during pre-flight planning?",
        "How do you select an alternate aerodrome?",
        "What are the requirements for international flight planning?",
        "How do you calculate the estimated time en route?",
        "What information is required in a flight plan?",
        "How do you determine the route of flight?",
        "What weather information is needed for flight planning?",
        "How do you calculate the top of descent point?",
        "What are the procedures for flight plan amendments?",
        "How do you monitor a flight in progress?",
        "What actions should be taken if the flight deviates from the plan?",
        "How do you calculate fuel burn rate?",
        "What are the considerations for route selection?",
        "How do you determine the appropriate airspeed for cruise?",
        "What are the procedures for flight plan closure?"
    ],
    
    "higher_order": [
        # Multi-step reasoning, trade-offs, conditional logic (10 questions)
        "What trade-offs should be considered when choosing between a direct route and airways?",
        "How would weather conditions affect the choice of alternate aerodrome?",
        "If the planned cruising altitude is unavailable, how should the pilot adjust the flight plan?",
        "What factors would lead a pilot to decide to divert to an alternate airport?",
        "How do fuel requirements change for different flight conditions?",
        "What is the relationship between aircraft weight and optimal cruise altitude?",
        "How should a pilot balance fuel efficiency with flight time?",
        "What considerations are needed when planning a flight over mountainous terrain?",
        "How do wind conditions affect route planning and fuel calculations?",
        "What decision-making process should be used when flight conditions deteriorate?"
    ]
}

# ========================================
# EVALUATION METRICS
# ========================================

class EvaluationMetrics:
    """Stores and calculates evaluation metrics."""
    
    def __init__(self):
        self.results = []
        self.total_questions = 0
        self.retrieval_hits = 0
        self.faithful_answers = 0
        self.hallucinations = 0
        self.no_answer_count = 0
    
    def add_result(self, question: str, category: str, answer: str, 
                   chunks: List[Dict], hit: bool, faithful: bool, hallucinated: bool):
        """Add a single evaluation result."""
        self.results.append({
            'question': question,
            'category': category,
            'answer': answer,
            'chunks': chunks,
            'retrieval_hit': hit,
            'faithful': faithful,
            'hallucinated': hallucinated
        })
        
        self.total_questions += 1
        if hit:
            self.retrieval_hits += 1
        if faithful:
            self.faithful_answers += 1
        if hallucinated:
            self.hallucinations += 1
        if "not available" in answer.lower():
            self.no_answer_count += 1
    
    def calculate_rates(self) -> Dict[str, float]:
        """Calculate percentage rates."""
        if self.total_questions == 0:
            return {}
        
        return {
            'retrieval_hit_rate': (self.retrieval_hits / self.total_questions) * 100,
            'faithfulness_rate': (self.faithful_answers / self.total_questions) * 100,
            'hallucination_rate': (self.hallucinations / self.total_questions) * 100,
            'no_answer_rate': (self.no_answer_count / self.total_questions) * 100
        }

# ========================================
# EVALUATION FUNCTIONS
# ========================================

def check_retrieval_hit(answer: str, chunks: List[Dict]) -> bool:
    """
    Check if retrieved chunks contain information relevant to the answer.
    
    Returns True if answer keywords appear in chunks.
    """
    if "not available" in answer.lower():
        return False
    
    # Extract key terms from answer (words longer than 3 chars)
    answer_words = set(
        word.lower() 
        for word in re.findall(r'\b\w+\b', answer) 
        if len(word) > 3
    )
    
    # Check if answer words appear in chunks
    chunk_text = " ".join(chunk['text'].lower() for chunk in chunks)
    chunk_words = set(re.findall(r'\b\w+\b', chunk_text))
    
    # If at least 20% of answer words are in chunks, consider it a hit
    overlap = len(answer_words & chunk_words)
    if len(answer_words) == 0:
        return False
    
    return (overlap / len(answer_words)) >= 0.2


def check_faithfulness(answer: str, chunks: List[Dict]) -> bool:
    """
    Check if the answer is grounded in retrieved chunks.
    
    Returns True if answer content can be found in chunks.
    """
    if "not available" in answer.lower():
        return True  # Fallback is acceptable
    
    # Get all chunk text
    chunk_text = " ".join(chunk['text'].lower() for chunk in chunks)
    
    # Split answer into sentences
    sentences = re.split(r'[.!?]+', answer.lower())
    
    grounded_count = 0
    total_sentences = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
        
        total_sentences += 1
        
        # Extract key terms from sentence
        sentence_words = set(
            word for word in re.findall(r'\b\w+\b', sentence)
            if len(word) > 3
        )
        
        # Check overlap with chunks
        chunk_words = set(re.findall(r'\b\w+\b', chunk_text))
        overlap = len(sentence_words & chunk_words)
        
        # If at least 30% of sentence words are in chunks, consider grounded
        if len(sentence_words) > 0 and (overlap / len(sentence_words)) >= 0.3:
            grounded_count += 1
    
    if total_sentences == 0:
        return False
    
    # Answer is faithful if at least 60% of sentences are grounded
    return (grounded_count / total_sentences) >= 0.6


def check_hallucination(answer: str, chunks: List[Dict]) -> bool:
    """
    Check if answer contains unsupported claims (hallucinations).
    
    Returns True if answer contains content NOT in chunks.
    """
    if "not available" in answer.lower():
        return False  # Fallback is not a hallucination
    
    # If answer is not faithful, it's likely hallucinated
    faithful = check_faithfulness(answer, chunks)
    
    # Also check for specific hallucination indicators
    hallucination_phrases = [
        "i think", "probably", "might be", "could be",
        "in general", "typically", "usually", "often",
        "most likely", "it seems"
    ]
    
    answer_lower = answer.lower()
    has_uncertain_language = any(phrase in answer_lower for phrase in hallucination_phrases)
    
    return (not faithful) or has_uncertain_language


def run_evaluation(doc_store: DocumentStore) -> EvaluationMetrics:
    """
    Run complete evaluation on all 50 questions.
    
    Args:
        doc_store: Initialized DocumentStore with loaded documents
        
    Returns:
        EvaluationMetrics object with all results
    """
    metrics = EvaluationMetrics()
    
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)
    
    # Evaluate each category
    for category, questions in QUESTIONS.items():
        print(f"\n📝 Testing {category.replace('_', ' ').title()} Questions...")
        print(f"   Total: {len(questions)}")
        
        for i, question in enumerate(questions, 1):
            print(f"\n   [{i}/{len(questions)}] {question[:60]}...")
            
            # Get retrieved chunks
            retrieval_result = doc_store.retrieve(question, top_k=3)
            chunks = retrieval_result.get('chunks', [])
            
            # Generate answer
            answer = generate_llm_response(chunks, question)
            
            # Evaluate
            hit = check_retrieval_hit(answer, chunks)
            faithful = check_faithfulness(answer, chunks)
            hallucinated = check_hallucination(answer, chunks)
            
            # Add to metrics
            metrics.add_result(
                question=question,
                category=category,
                answer=answer,
                chunks=chunks,
                hit=hit,
                faithful=faithful,
                hallucinated=hallucinated
            )
            
            # Quick status
            status = "✓" if hit and faithful and not hallucinated else "✗"
            print(f"   {status} Hit: {hit} | Faithful: {faithful} | Hallucinated: {hallucinated}")
    
    return metrics


def generate_report(metrics: EvaluationMetrics, output_file: str = "evaluation_report.txt"):
    """
    Generate comprehensive evaluation report.
    
    Args:
        metrics: Evaluation results
        output_file: Path to save report
    """
    rates = metrics.calculate_rates()
    
    # Find best and worst answers
    best_answers = sorted(
        metrics.results,
        key=lambda x: (x['retrieval_hit'], x['faithful'], not x['hallucinated']),
        reverse=True
    )[:5]
    
    worst_answers = sorted(
        metrics.results,
        key=lambda x: (x['retrieval_hit'], x['faithful'], not x['hallucinated'])
    )[:5]
    
    # Generate report content
    report = []
    report.append("="*80)
    report.append("RAG SYSTEM EVALUATION REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Questions Tested: {metrics.total_questions}")
    report.append("")
    
    # Overall Metrics
    report.append("="*80)
    report.append("OVERALL METRICS")
    report.append("="*80)
    report.append(f"Retrieval Hit Rate:    {rates['retrieval_hit_rate']:.2f}%")
    report.append(f"  → {metrics.retrieval_hits}/{metrics.total_questions} questions had relevant chunks retrieved")
    report.append("")
    report.append(f"Faithfulness Rate:     {rates['faithfulness_rate']:.2f}%")
    report.append(f"  → {metrics.faithful_answers}/{metrics.total_questions} answers were grounded in retrieved text")
    report.append("")
    report.append(f"Hallucination Rate:    {rates['hallucination_rate']:.2f}%")
    report.append(f"  → {metrics.hallucinations}/{metrics.total_questions} answers contained unsupported claims")
    report.append("")
    report.append(f"No Answer Rate:        {rates['no_answer_rate']:.2f}%")
    report.append(f"  → {metrics.no_answer_count}/{metrics.total_questions} questions could not be answered")
    report.append("")
    
    # Category Breakdown
    report.append("="*80)
    report.append("CATEGORY BREAKDOWN")
    report.append("="*80)
    
    for category in QUESTIONS.keys():
        category_results = [r for r in metrics.results if r['category'] == category]
        total = len(category_results)
        hits = sum(1 for r in category_results if r['retrieval_hit'])
        faithful = sum(1 for r in category_results if r['faithful'])
        hallucinated = sum(1 for r in category_results if r['hallucinated'])
        
        report.append(f"\n{category.replace('_', ' ').title()}:")
        report.append(f"  Total Questions: {total}")
        report.append(f"  Hit Rate:        {(hits/total*100):.1f}%")
        report.append(f"  Faithfulness:    {(faithful/total*100):.1f}%")
        report.append(f"  Hallucinations:  {(hallucinated/total*100):.1f}%")
    
    report.append("")
    
    # Best Answers
    report.append("="*80)
    report.append("TOP 5 BEST ANSWERS")
    report.append("="*80)
    
    for i, result in enumerate(best_answers, 1):
        report.append(f"\n{i}. Question: {result['question']}")
        report.append(f"   Category: {result['category'].replace('_', ' ').title()}")
        report.append(f"   Answer: {result['answer'][:200]}...")
        report.append(f"   ✓ Retrieval Hit: {result['retrieval_hit']}")
        report.append(f"   ✓ Faithful: {result['faithful']}")
        report.append(f"   ✓ No Hallucination: {not result['hallucinated']}")
        report.append(f"   Explanation: This answer demonstrates excellent retrieval and grounding.")
    
    report.append("")
    
    # Worst Answers
    report.append("="*80)
    report.append("TOP 5 WORST ANSWERS")
    report.append("="*80)
    
    for i, result in enumerate(worst_answers, 1):
        report.append(f"\n{i}. Question: {result['question']}")
        report.append(f"   Category: {result['category'].replace('_', ' ').title()}")
        report.append(f"   Answer: {result['answer'][:200]}...")
        report.append(f"   ✗ Retrieval Hit: {result['retrieval_hit']}")
        report.append(f"   ✗ Faithful: {result['faithful']}")
        report.append(f"   ✗ Hallucinated: {result['hallucinated']}")
        
        # Diagnose issue
        if not result['retrieval_hit']:
            report.append(f"   Issue: Retrieval failed - chunks did not contain relevant information")
        elif result['hallucinated']:
            report.append(f"   Issue: Answer contains unsupported claims not found in source")
        elif not result['faithful']:
            report.append(f"   Issue: Answer not properly grounded in retrieved chunks")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    
    output_path = Path(output_file)
    output_path.write_text(report_text, encoding='utf-8')
    
    print(f"\n✓ Report saved to: {output_path.absolute()}")
    print(f"\n{report_text}")
    
    # Also save detailed JSON results
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'metrics': rates,
            'results': metrics.results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Detailed results saved to: {json_path.absolute()}")


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    """Main evaluation workflow."""
    print("\n" + "="*60)
    print("RAG SYSTEM EVALUATION")
    print("="*60)
    
    # Initialize document store
    print("\n1. Loading document store...")
    doc_store = DocumentStore()
    
    if not doc_store.is_loaded:
        print("\n✗ ERROR: No documents loaded!")
        print("   Please ingest documents first using:")
        print("   python tech.py")
        print("   Then POST to /ingest endpoint")
        return
    
    print(f"✓ Loaded {len(doc_store.chunks)} chunks")
    
    # Run evaluation
    print("\n2. Running evaluation...")
    metrics = run_evaluation(doc_store)
    
    # Generate report
    print("\n3. Generating report...")
    generate_report(metrics)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
