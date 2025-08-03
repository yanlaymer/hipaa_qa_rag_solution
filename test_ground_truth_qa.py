#!/usr/bin/env python3
"""
Test script to validate HIPAA QA system with ground truth questions.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hipaa_qa.config import Settings
from hipaa_qa.database import DatabaseManager, ChunkRepository
from hipaa_qa.services import EmbeddingService, QAService
from hipaa_qa.schemas import QuestionRequest


# Ground truth questions from test_questions_ground_truth.md
GROUND_TRUTH_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the overall purpose of HIPAA Part 160?",
        "expected_key_points": [
            "Implements Social Security Act sections 1171-1180",
            "Administrative simplification provisions", 
            "General administrative requirements for HIPAA"
        ],
        "ground_truth": "The requirements of this subchapter implement sections 1171-1180 of the Social Security Act"
    },
    {
        "id": 2,
        "question": "Which part covers data privacy measures?",
        "expected_key_points": [
            "Part 164, Subpart E is the Privacy Rule",
            "Covers uses and disclosures of protected health information",
            "Includes sections 164.500-164.534"
        ],
        "ground_truth": "Part 164, Subpart E covers \"Privacy of Individually Identifiable Health Information\""
    },
    {
        "id": 3,
        "question": "What does \"minimum necessary\" mean in HIPAA terminology?",
        "expected_key_points": [
            "Limit PHI to minimum necessary for intended purpose",
            "Applies to uses, disclosures, and requests",
            "Requirement for reasonable efforts"
        ],
        "ground_truth": "make reasonable efforts to limit protected health information to the minimum necessary"
    },
    {
        "id": 4,
        "question": "Which entities are specifically regulated under HIPAA?",
        "expected_key_points": [
            "Health plans",
            "Health care clearinghouses", 
            "Health care providers (who transmit electronically)",
            "Business associates"
        ],
        "ground_truth": "health plan, health care clearinghouse, health care provider who transmits"
    },
    {
        "id": 5,
        "question": "What are the potential civil penalties for noncompliance?",
        "expected_key_points": [
            "Did not know: $100-$50,000 per violation",
            "Reasonable cause: $1,000-$50,000 per violation",
            "Willful neglect: $10,000-$50,000 per violation",
            "Up to $1.5M per year"
        ],
        "ground_truth": "Penalties range from $100 to $50,000 per violation"
    },
    {
        "id": 6,
        "question": "Does HIPAA mention encryption best practices?",
        "expected_key_points": [
            "Encryption is an \"addressable\" implementation specification",
            "Required for access control and transmission security",
            "Not mandatory but must be implemented if reasonable and appropriate"
        ],
        "ground_truth": "Implement a mechanism to encrypt and decrypt electronic protected health information"
    },
    {
        "id": 7,
        "question": "Can I disclose personal health information to family members?",
        "expected_key_points": [
            "Permitted under § 164.510",
            "Must be relevant to involvement in care",
            "Opportunity to agree/object required",
            "Special rules for deceased individuals"
        ],
        "ground_truth": "May disclose to family member, other relative, or close personal friend"
    },
    {
        "id": 8,
        "question": "If a covered entity outsources data processing, which sections apply?",
        "expected_key_points": [
            "Business associate relationship created",
            "Written business associate agreement required",
            "Business associate subject to HIPAA requirements",
            "Covered entity remains liable for oversight"
        ],
        "ground_truth": "Business associate provisions apply"
    },
    {
        "id": 9,
        "question": "Cite the specific regulation texts regarding permitted disclosures to law enforcement.",
        "expected_key_points": [
            "§ 164.512(f) is the main law enforcement section",
            "Six specific categories of permitted disclosures",
            "Each category has specific conditions and limitations",
            "Must meet applicable conditions"
        ],
        "ground_truth": "§ 164.512(f): Disclosures for law enforcement purposes are permitted"
    }
]


async def run_qa_tests():
    """Run QA tests with ground truth questions."""
    print("🚀 Starting HIPAA QA System Ground Truth Tests")
    print("=" * 60)
    
    # Initialize services
    settings = Settings()
    # Override DB_HOST to connect to Docker database
    settings.db_host = "hipaa-qa-system-db-1"
    settings.db_user = "postgres"
    settings.db_password = "postgres"
    
    db_manager = DatabaseManager(settings)
    embedding_service = EmbeddingService(settings)
    repository = ChunkRepository(db_manager)
    qa_service = QAService(repository, embedding_service, settings)
    
    results = []
    total_questions = len(GROUND_TRUTH_QUESTIONS)
    
    print(f"📊 Testing {total_questions} ground truth questions...\n")
    
    for i, test_case in enumerate(GROUND_TRUTH_QUESTIONS, 1):
        print(f"🔍 Question {i}/{total_questions}: {test_case['question'][:60]}...")
        
        start_time = time.time()
        try:
            # Ask the question
            response = await qa_service.answer_question(
                question=test_case["question"],
                max_chunks=5,
                similarity_threshold=0.6
            )
            
            processing_time = time.time() - start_time
            
            # Evaluate the response
            answer = response.answer.lower()
            ground_truth = test_case["ground_truth"].lower()
            
            # Simple relevance scoring
            key_words_found = sum(1 for key in test_case["expected_key_points"] 
                                if any(word.lower() in answer for word in key.split()))
            relevance_score = key_words_found / len(test_case["expected_key_points"])
            
            # Check if ground truth content is mentioned
            contains_ground_truth = any(word in answer for word in ground_truth.split() if len(word) > 3)
            
            result = {
                "question_id": test_case["id"],
                "question": test_case["question"],
                "answer": response.answer,
                "chunks_retrieved": response.chunks_retrieved,
                "confidence_score": response.confidence_score,
                "processing_time_ms": response.processing_time_ms,
                "relevance_score": relevance_score,
                "contains_ground_truth": contains_ground_truth,
                "sources": [{"section": src.section_id, "citation": src.cfr_citation} 
                           for src in response.sources],
                "evaluation": {
                    "relevant": relevance_score > 0.3,
                    "accurate": contains_ground_truth,
                    "comprehensive": response.chunks_retrieved >= 3
                }
            }
            
            results.append(result)
            
            # Print summary
            status = "✅" if result["evaluation"]["accurate"] else "⚠️"
            print(f"   {status} Confidence: {response.confidence_score:.2f}, "
                  f"Chunks: {response.chunks_retrieved}, "
                  f"Time: {processing_time:.2f}s")
            print(f"   📝 Answer: {response.answer[:100]}...")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "question_id": test_case["id"],
                "question": test_case["question"],
                "error": str(e),
                "evaluation": {"relevant": False, "accurate": False, "comprehensive": False}
            })
            print()
    
    # Generate summary report
    await generate_test_report(results)
    
    # Cleanup
    await db_manager.close()
    
    return results


async def generate_test_report(results: List[Dict]):
    """Generate a comprehensive test report."""
    print("\n" + "=" * 60)
    print("📊 HIPAA QA SYSTEM TEST RESULTS")
    print("=" * 60)
    
    total_questions = len(results)
    successful_answers = sum(1 for r in results if not r.get("error"))
    accurate_answers = sum(1 for r in results if r.get("evaluation", {}).get("accurate", False))
    relevant_answers = sum(1 for r in results if r.get("evaluation", {}).get("relevant", False))
    
    print(f"📈 Overall Performance:")
    print(f"   • Total Questions: {total_questions}")
    print(f"   • Successful Responses: {successful_answers}/{total_questions} ({successful_answers/total_questions*100:.1f}%)")
    print(f"   • Accurate Answers: {accurate_answers}/{total_questions} ({accurate_answers/total_questions*100:.1f}%)")
    print(f"   • Relevant Answers: {relevant_answers}/{total_questions} ({relevant_answers/total_questions*100:.1f}%)")
    
    if successful_answers > 0:
        avg_confidence = sum(r.get("confidence_score", 0) for r in results if not r.get("error")) / successful_answers
        avg_chunks = sum(r.get("chunks_retrieved", 0) for r in results if not r.get("error")) / successful_answers
        avg_time = sum(r.get("processing_time_ms", 0) for r in results if not r.get("error")) / successful_answers
        
        print(f"\n📊 Performance Metrics:")
        print(f"   • Average Confidence: {avg_confidence:.2f}")
        print(f"   • Average Chunks Retrieved: {avg_chunks:.1f}")
        print(f"   • Average Response Time: {avg_time:.0f}ms")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅" if result.get("evaluation", {}).get("accurate") else "❌"
        print(f"   {status} Q{result['question_id']}: {result['question'][:50]}...")
        if result.get("error"):
            print(f"      ❌ Error: {result['error']}")
        else:
            eval_data = result.get("evaluation", {})
            print(f"      📊 Accurate: {'Yes' if eval_data.get('accurate') else 'No'}, "
                  f"Relevant: {'Yes' if eval_data.get('relevant') else 'No'}, "
                  f"Confidence: {result.get('confidence_score', 0):.2f}")
    
    # Save detailed results to file
    timestamp = int(time.time())
    results_file = f"qa_test_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("=" * 60)


async def main():
    """Main function."""
    try:
        await run_qa_tests()
        print("\n🎉 HIPAA QA System testing completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)