import os
import json
import csv
import time
import argparse
from typing import List, Dict, Any
from pydantic import TypeAdapter
from src.models import EmailScenario
from src.generator import EmailGenerator
from src.evaluator import evaluate_email_generation
from src.config import MODEL_A, MODEL_B

# Define Metric Definitions & Logic to save in the final report
METRICS_DEFINITIONS = {
    "Fact Recall Metric": {
        "description": "Measures the completeness of information retrieval in the generated email compared to key facts.",
        "logic": "Uses an LLM-as-a-Judge system to verify if each individual fact bullet point is semantically present in the generated email body. Score = (facts recalled / total facts), ranging from 0.0 to 1.0."
    },
    "Tone Adherence Metric": {
        "description": "Measures style and voice alignment with the requested tone.",
        "logic": "Uses an LLM-as-a-Judge system to evaluate tone alignment (formal, casual, urgent, empathetic) on a 1-5 rubric. Score is normalized to a 0.0 to 1.0 scale."
    },
    "Structure and Professionalism Metric": {
        "description": "Measures structure correctness, layout flow, and professional greeting/sign-off while checking for extra conversation fluff.",
        "logic": "Combines a programmatic search for standard greetings and sign-offs (40% weight) with an LLM-as-a-Judge evaluation of layout quality and checks for conversational meta-talk (60% weight)."
    }
}

def load_scenarios(file_path: str) -> List[EmailScenario]:
    """Loads and validates scenarios from the JSON file using Pydantic."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not found.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Use Pydantic's TypeAdapter to validate the array of items
    ta = TypeAdapter(List[EmailScenario])
    return ta.validate_python(raw_data)

def run_evaluation(scenarios: List[EmailScenario], dry_run: bool = False):
    """Orchestrates generation and evaluation of email scenarios across models."""
    generator = EmailGenerator()
    models = [MODEL_A, MODEL_B]
    
    # Slice to a single scenario if dry-run mode is enabled
    if dry_run:
        print("\n=== RUNNING IN DRY-RUN MODE (First Scenario Only) ===")
        scenarios = scenarios[:1]
        
    results_by_model: Dict[str, List[Dict[str, Any]]] = {model: [] for model in models}
    
    for model in models:
        print(f"\nEvaluating Model: {model}...")
        print("-" * 50)
        
        for idx, scenario in enumerate(scenarios):
            print(f"Scenario {scenario.id}/{len(scenarios)}: Intent: '{scenario.intent[:40]}...' | Tone: '{scenario.tone}'")
            
            start_time = time.time()
            try:
                # 1. Generate Email using Instructor & Layered Prompting
                gen_response = generator.generate_email(scenario, model)
                latency = round(time.time() - start_time, 2)
                
                # 2. Evaluate using DeepEval Custom Metrics
                eval_results = evaluate_email_generation(
                    intent=scenario.intent,
                    key_facts=scenario.key_facts,
                    tone=scenario.tone,
                    generated_email=gen_response.email_body,
                    reference_email=scenario.reference_email
                )
                
                # Store results
                results_by_model[model].append({
                    "scenario_id": scenario.id,
                    "intent": scenario.intent,
                    "tone": scenario.tone,
                    "thinking": gen_response.thinking,
                    "subject": gen_response.email_subject,
                    "generated_body": gen_response.email_body,
                    "reference_email": scenario.reference_email,
                    "latency_seconds": latency,
                    "metrics": eval_results,
                    "status": "success"
                })
                print(f"  -> Generated in {latency}s | Fact Recall: {eval_results['fact_recall']['score']:.2f} | Tone: {eval_results['tone_adherence']['score']:.2f} | Avg: {eval_results['average_score']:.2f}")
                
            except Exception as e:
                latency = round(time.time() - start_time, 2)
                print(f"  -> ERROR: Failed to generate/evaluate scenario {scenario.id}: {str(e)}")
                results_by_model[model].append({
                    "scenario_id": scenario.id,
                    "intent": scenario.intent,
                    "tone": scenario.tone,
                    "status": "failed",
                    "error": str(e),
                    "latency_seconds": latency,
                    "metrics": {
                        "fact_recall": {"score": 0.0, "reason": f"Failed: {str(e)}"},
                        "tone_adherence": {"score": 0.0, "reason": f"Failed: {str(e)}"},
                        "structure_polish": {"score": 0.0, "reason": f"Failed: {str(e)}"},
                        "average_score": 0.0
                    }
                })
            time.sleep(1.0) # Rate limiting buffer between OpenRouter requests
            
    # Calculate Averages and Compile Report
    compile_reports(results_by_model, dry_run)

def compile_reports(results: Dict[str, List[Dict[str, Any]]], dry_run: bool):
    """Calculates model averages, outputs console summary, and writes JSON/CSV reports."""
    summary_report: Dict[str, Any] = {
        "metrics_definitions": METRICS_DEFINITIONS,
        "run_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dry_run": dry_run
        },
        "model_performances": {}
    }
    
    csv_rows = []
    
    print("\n" + "="*50)
    print(" EVALUATION SUMMARY RESULTS ")
    print("="*50)

    for model, runs in results.items():
        total_scenarios = len(runs)
        successful_runs = [r for r in runs if r["status"] == "success"]
        
        avg_fact = sum(r["metrics"]["fact_recall"]["score"] for r in runs) / total_scenarios if total_scenarios > 0 else 0.0
        avg_tone = sum(r["metrics"]["tone_adherence"]["score"] for r in runs) / total_scenarios if total_scenarios > 0 else 0.0
        avg_struct = sum(r["metrics"]["structure_polish"]["score"] for r in runs) / total_scenarios if total_scenarios > 0 else 0.0
        avg_overall = (avg_fact + avg_tone + avg_struct) / 3.0
        avg_latency = sum(r["latency_seconds"] for r in runs) / total_scenarios if total_scenarios > 0 else 0.0
        
        summary_report["model_performances"][model] = {
            "aggregates": {
                "average_fact_recall": round(avg_fact, 4),
                "average_tone_adherence": round(avg_tone, 4),
                "average_structure_polish": round(avg_struct, 4),
                "overall_average_score": round(avg_overall, 4),
                "average_latency_seconds": round(avg_latency, 2)
            },
            "runs": runs
        }
        
        print(f"Model: {model}")
        print(f"  Average Fact Recall  : {avg_fact:.4f}")
        print(f"  Average Tone Match   : {avg_tone:.4f}")
        print(f"  Average Structure    : {avg_struct:.4f}")
        print(f"  Overall Average Score: {avg_overall:.4f}")
        print(f"  Average Latency      : {avg_latency:.2f}s")
        print("-" * 50)
        
        # Build rows for CSV reporting
        for r in runs:
            csv_rows.append({
                "Model": model,
                "Scenario ID": r["scenario_id"],
                "Intent": r["intent"],
                "Tone": r["tone"],
                "Fact Recall Score": r["metrics"]["fact_recall"]["score"],
                "Tone Adherence Score": r["metrics"]["tone_adherence"]["score"],
                "Structure Polish Score": r["metrics"]["structure_polish"]["score"],
                "Average Score": r["metrics"]["average_score"],
                "Latency (Seconds)": r["latency_seconds"],
                "Status": r["status"]
            })
            
    # Ensure Output directory exists
    os.makedirs("Output", exist_ok=True)

    # Write JSON report
    report_json_path = os.path.join("Output", "evaluation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed evaluation logs to: {report_json_path}")
    
    # Write CSV summary
    report_csv_path = os.path.join("Output", "evaluation_summary.csv")
    csv_headers = [
        "Model", "Scenario ID", "Intent", "Tone", 
        "Fact Recall Score", "Tone Adherence Score", "Structure Polish Score", 
        "Average Score", "Latency (Seconds)", "Status"
    ]
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved comparative summary stats to: {report_csv_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email Generation Assistant Evaluation CLI Runner.")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Run only the first scenario to verify OpenRouter connection and evaluator metrics."
    )
    args = parser.parse_args()
    
    try:
        scenarios_data = load_scenarios(os.path.join("Input", "scenarios.json"))
        run_evaluation(scenarios_data, args.dry_run)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
