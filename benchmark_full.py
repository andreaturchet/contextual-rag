"""
Benchmark Visualization Script
------------------------------
Generates graphics from benchmark results comparing Baseline vs Contextual Retrieval.

Usage:
    python benchmark_full.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_benchmark_results():
    """Load benchmark results from JSON files."""
    base_path = Path(__file__).parent / "data"
    baseline_path = base_path / "baseline_results.json"
    contextual_path = base_path / "contextual_results.json"

    if not baseline_path.exists():
        print(f"Error: {baseline_path} not found!")
        print("Run 'python benchmark.py --baseline' first to generate results.")
        return None

    if not contextual_path.exists():
        print(f"Error: {contextual_path} not found!")
        print("Run 'python benchmark.py --contextual' first to generate results.")
        return None

    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(contextual_path) as f:
        contextual = json.load(f)

    # Limit to first 5 questions for chart compatibility
    if len(baseline.get("results", [])) > 5:
        print(f"  Note: Limiting charts to first 5 questions (found {len(baseline['results'])})")
        baseline["results"] = baseline["results"][:5]
        contextual["results"] = contextual["results"][:5]

    return {
        "baseline": baseline,
        "contextual": contextual,
        "timestamp": "latest"
    }


def create_output_dir():
    """Create output directory for charts."""
    output_dir = Path(__file__).parent / "data" / "charts"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_question_labels(results):
    """Generate short labels from questions dynamically."""
    labels = []
    for r in results:
        q = r["question"].lower()
        if "business segments" in q or "main segments" in q:
            labels.append("Business\nSegments")
        elif "electric vehicles" in q or "ev products" in q or "products for ev" in q:
            labels.append("EV\nProducts")
        elif "sustainability" in q:
            labels.append("Sustainability")
        elif "internship" in q or "skills" in q:
            labels.append("Internship\nSkills")
        elif "ai strategy" in q:
            labels.append("AI\nStrategy")
        elif "founded" in q or "when was" in q:
            labels.append("Founded")
        elif "headquarter" in q or "where is" in q:
            labels.append("HQ\nLocation")
        elif "what is infineon" in q:
            labels.append("About\nInfineon")
        elif "ceo" in q:
            labels.append("CEO")
        elif "radar" in q or "adas" in q or "autonomous" in q:
            labels.append("Radar/\nADAS")
        else:
            # Fallback: first 12 chars
            labels.append(r["question"][:12] + "...")
    return labels


def plot_overall_comparison(data, output_dir):
    """Create bar chart comparing overall metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline = data["baseline"]
    contextual = data["contextual"]

    # Colors
    colors = ['#3498db', '#2ecc71']
    methods = ['Baseline', 'Contextual']

    # Helper to get metric with fallback for old format
    def get_metric(d, new_key, old_key):
        return d.get(new_key, d.get(old_key, 0))

    # Subplot 1: Ground Truth Accuracy (NEW) or Keyword Score as fallback
    ax1 = axes[0, 0]
    if "avg_accuracy" in baseline or "avg_accuracy" in contextual:
        scores = [baseline.get("avg_accuracy", 0) * 100,
                  contextual.get("avg_accuracy", 0) * 100]
        title = 'Ground Truth Accuracy'
    else:
        scores = [get_metric(baseline, "avg_keyword_score", "avg_score") * 100,
                  get_metric(contextual, "avg_keyword_score", "avg_score") * 100]
        title = 'Keyword Score'
    bars1 = ax1.bar(methods, scores, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, score in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    improvement = scores[1] - scores[0]
    ax1.annotate(f'{improvement:+.1f}%', xy=(0.5, max(scores) + 8),
                fontsize=14, color='green' if improvement > 0 else 'red', fontweight='bold', ha='center')

    # Subplot 2: Faithfulness
    ax2 = axes[0, 1]
    faithfulness = [baseline.get("avg_faithfulness", 0) * 100,
                    contextual.get("avg_faithfulness", 0) * 100]
    bars2 = ax2.bar(methods, faithfulness, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Score (%)', fontsize=12)
    ax2.set_title('Faithfulness', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar, score in zip(bars2, faithfulness):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    faith_improvement = faithfulness[1] - faithfulness[0]
    ax2.annotate(f'{faith_improvement:+.1f}%', xy=(0.5, max(faithfulness) + 8),
                fontsize=14, color='green' if faith_improvement > 0 else 'red', fontweight='bold', ha='center')

    # Subplot 3: Retrieval Precision
    ax3 = axes[1, 0]
    precision = [baseline.get("avg_precision", 0) * 100,
                 contextual.get("avg_precision", 0) * 100]
    bars3 = ax3.bar(methods, precision, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Score (%)', fontsize=12)
    ax3.set_title('Retrieval Precision', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 100)
    for bar, score in zip(bars3, precision):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    prec_improvement = precision[1] - precision[0]
    ax3.annotate(f'{prec_improvement:+.1f}%', xy=(0.5, max(precision) + 8),
                fontsize=14, color='green' if prec_improvement > 0 else 'red', fontweight='bold', ha='center')

    # Subplot 4: Average Latency
    ax4 = axes[1, 1]
    latencies = [baseline["avg_latency"], contextual["avg_latency"]]
    bars4 = ax4.bar(methods, latencies, color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Latency (seconds)', fontsize=12)
    ax4.set_title('Average Query Latency', fontsize=14, fontweight='bold')
    for bar, lat in zip(bars4, latencies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{lat:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    overhead = latencies[1] - latencies[0]
    ax4.annotate(f'{overhead:+.1f}s', xy=(0.5, max(latencies) + 5),
                fontsize=11, color='orange', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "overall_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: overall_comparison.png")


def plot_per_question_scores(data, output_dir):
    """Create grouped bar chart for per-question scores."""
    baseline_results = data["baseline"]["results"]
    contextual_results = data["contextual"]["results"]

    # Generate dynamic labels
    questions = get_question_labels(baseline_results)

    # Helper to get score with fallback
    def get_score(r):
        return r.get("keyword_score", r.get("score", 0))

    baseline_scores = [get_score(r) * 100 for r in baseline_results]
    contextual_scores = [get_score(r) * 100 for r in contextual_results]

    x = np.arange(len(questions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline',
                   color='#3498db', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, contextual_scores, width, label='Contextual',
                   color='#2ecc71', edgecolor='black', linewidth=1)

    ax.set_ylabel('Keyword Score (%)', fontsize=12)
    ax.set_title('Per-Question Keyword Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(questions, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9)

    # Add improvement indicators
    for i, (b, c) in enumerate(zip(baseline_scores, contextual_scores)):
        diff = c - b
        if diff > 0:
            ax.annotate(f'+{diff:.0f}%', xy=(i, max(b, c) + 8),
                       fontsize=9, color='green', fontweight='bold', ha='center')
        elif diff < 0:
            ax.annotate(f'{diff:.0f}%', xy=(i, max(b, c) + 8),
                       fontsize=9, color='red', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "per_question_scores.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: per_question_scores.png")


def plot_latency_comparison(data, output_dir):
    """Create latency comparison chart."""
    baseline_results = data["baseline"]["results"]
    contextual_results = data["contextual"]["results"]

    questions = [f"Q{i+1}" for i in range(len(baseline_results))]
    baseline_latencies = [r["latency"] for r in baseline_results]
    contextual_latencies = [r["latency"] for r in contextual_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(questions))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_latencies, width, label='Baseline',
                   color='#3498db', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, contextual_latencies, width, label='Contextual',
                   color='#2ecc71', edgecolor='black', linewidth=1)

    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_xlabel('Question', fontsize=12)
    ax.set_title('Query Latency per Question', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(questions, fontsize=11)
    ax.legend(fontsize=11)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: latency_comparison.png")


def plot_radar_chart(data, output_dir):
    """Create radar chart for multi-dimensional comparison."""
    baseline_results = data["baseline"]["results"]
    contextual_results = data["contextual"]["results"]

    # Generate dynamic labels
    categories = get_question_labels(baseline_results)

    # Helper to get score with fallback
    def get_score(r):
        return r.get("keyword_score", r.get("score", 0))

    baseline_scores = [get_score(r) for r in baseline_results]
    contextual_scores = [get_score(r) for r in contextual_results]

    # Number of categories
    N = len(categories)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Complete the loop
    baseline_scores += baseline_scores[:1]
    contextual_scores += contextual_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot data
    ax.plot(angles, baseline_scores, 'o-', linewidth=2, label='Baseline', color='#3498db')
    ax.fill(angles, baseline_scores, alpha=0.25, color='#3498db')

    ax.plot(angles, contextual_scores, 'o-', linewidth=2, label='Contextual', color='#2ecc71')
    ax.fill(angles, contextual_scores, alpha=0.25, color='#2ecc71')

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)

    # Set y-axis
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9)

    ax.set_title('Answer Quality by Topic\n(Radar Comparison)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: radar_comparison.png")


def plot_improvement_waterfall(data, output_dir):
    """Create waterfall chart showing improvement breakdown."""
    baseline_results = data["baseline"]["results"]
    contextual_results = data["contextual"]["results"]

    # Generate dynamic labels (remove newlines for this chart)
    questions_short = [q.replace('\n', ' ') for q in get_question_labels(baseline_results)]

    # Helper to get score with fallback
    def get_score(r):
        return r.get("keyword_score", r.get("score", 0))

    improvements = [(get_score(c) - get_score(b)) * 100
                   for b, c in zip(baseline_results, contextual_results)]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in improvements]

    bars = ax.bar(questions_short, improvements, color=colors, edgecolor='black', linewidth=1.2)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Score Improvement (%)', fontsize=12)
    ax.set_title('Contextual Retrieval Improvement per Question', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=10)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 1 if height >= 0 else -1
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
               f'{imp:+.0f}%', ha='center', va=va, fontsize=11, fontweight='bold')

    # Add average improvement line
    avg_improvement = np.mean(improvements)
    ax.axhline(y=avg_improvement, color='blue', linestyle='--', linewidth=2,
               label=f'Average: {avg_improvement:+.1f}%')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_waterfall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: improvement_waterfall.png")


def plot_summary_dashboard(data, output_dir):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))

    baseline = data["baseline"]
    contextual = data["contextual"]

    # Helper to get metric with fallback
    def get_metric(d, new_key, old_key):
        return d.get(new_key, d.get(old_key, 0))

    def get_score(r):
        return r.get("keyword_score", r.get("score", 0))

    # Title
    fig.suptitle('RAG Benchmark: Baseline vs Contextual Retrieval',
                 fontsize=18, fontweight='bold', y=0.98)

    # Subplot 1: Overall scores
    ax1 = fig.add_subplot(2, 3, 1)
    scores = [get_metric(baseline, "avg_keyword_score", "avg_score") * 100,
              get_metric(contextual, "avg_keyword_score", "avg_score") * 100]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(['Baseline', 'Contextual'], scores, color=colors, edgecolor='black')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Keyword Score', fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score:.1f}%', ha='center', fontweight='bold')

    # Subplot 2: Faithfulness comparison
    ax2 = fig.add_subplot(2, 3, 2)
    faithfulness = [baseline.get("avg_faithfulness", 0) * 100,
                    contextual.get("avg_faithfulness", 0) * 100]
    bars = ax2.bar(['Baseline', 'Contextual'], faithfulness, color=colors, edgecolor='black')
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Faithfulness', fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar, score in zip(bars, faithfulness):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score:.1f}%', ha='center', fontweight='bold')

    # Subplot 3: Key metrics box
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis('off')

    kw_improvement = (get_metric(contextual, "avg_keyword_score", "avg_score") -
                      get_metric(baseline, "avg_keyword_score", "avg_score")) * 100
    faith_improvement = (contextual.get("avg_faithfulness", 0) -
                         baseline.get("avg_faithfulness", 0)) * 100
    prec_improvement = (contextual.get("avg_precision", 0) -
                        baseline.get("avg_precision", 0)) * 100
    latency_overhead = contextual["avg_latency"] - baseline["avg_latency"]

    metrics_text = f"""
    KEY METRICS
    ─────────────────────
    
    Keyword Score:
    {kw_improvement:+.1f}%
    
    Faithfulness:
    {faith_improvement:+.1f}%
    
    Retrieval Precision:
    {prec_improvement:+.1f}%
    
    Latency Overhead:
    {latency_overhead:+.1f}s
    
    ─────────────────────
    """

    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 4: Per-question comparison (horizontal bar)
    ax4 = fig.add_subplot(2, 1, 2)

    # Generate dynamic labels
    questions_short = [f"Q{i+1}: {q.replace(chr(10), ' ')}" for i, q in enumerate(get_question_labels(baseline["results"]))]

    baseline_scores = [get_score(r) * 100 for r in baseline["results"]]
    contextual_scores = [get_score(r) * 100 for r in contextual["results"]]

    y = np.arange(len(questions_short))
    height = 0.35

    ax4.barh(y - height/2, baseline_scores, height, label='Baseline', color='#3498db', edgecolor='black')
    ax4.barh(y + height/2, contextual_scores, height, label='Contextual', color='#2ecc71', edgecolor='black')

    ax4.set_xlabel('Score (%)')
    ax4.set_title('Per-Question Score Comparison', fontweight='bold')
    ax4.set_yticks(y)
    ax4.set_yticklabels(questions_short)
    ax4.legend(loc='lower right')
    ax4.set_xlim(0, 110)
    ax4.axvline(x=50, color='red', linestyle='--', alpha=0.5)

    # Add improvement annotations
    for i, (b, c) in enumerate(zip(baseline_scores, contextual_scores)):
        diff = c - b
        color = 'green' if diff > 0 else 'red' if diff < 0 else 'gray'
        ax4.text(max(b, c) + 3, i, f'{diff:+.0f}%', va='center', fontweight='bold', color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: summary_dashboard.png")


def plot_faithfulness_comparison(data, output_dir):
    """Create faithfulness comparison chart per question."""
    baseline_results = data["baseline"]["results"]
    contextual_results = data["contextual"]["results"]

    # Generate dynamic labels
    questions_short = get_question_labels(baseline_results)

    # Get faithfulness scores (handle missing data)
    def get_faithfulness(r):
        if "response" in r and "faithfulness" in r["response"]:
            return r["response"]["faithfulness"] * 100
        return 0

    baseline_faith = [get_faithfulness(r) for r in baseline_results]
    contextual_faith = [get_faithfulness(r) for r in contextual_results]

    x = np.arange(len(questions_short))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, baseline_faith, width, label='Baseline',
                   color='#3498db', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, contextual_faith, width, label='Contextual',
                   color='#2ecc71', edgecolor='black', linewidth=1)

    ax.set_ylabel('Faithfulness Score (%)', fontsize=12)
    ax.set_title('Per-Question Faithfulness Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(questions_short, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% threshold')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9)

    # Add improvement indicators
    for i, (b, c) in enumerate(zip(baseline_faith, contextual_faith)):
        diff = c - b
        if diff > 0:
            ax.annotate(f'+{diff:.0f}%', xy=(i, max(b, c) + 8),
                       fontsize=9, color='green', fontweight='bold', ha='center')
        elif diff < 0:
            ax.annotate(f'{diff:.0f}%', xy=(i, max(b, c) + 8),
                       fontsize=9, color='red', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "faithfulness_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: faithfulness_comparison.png")


def plot_keywords_analysis(data, output_dir):
    """Create keyword matching analysis chart."""
    baseline_results = data["baseline"]["results"]
    contextual_results = data["contextual"]["results"]

    # Generate dynamic labels
    questions_short = get_question_labels(baseline_results)

    baseline_found = [len(r["found_keywords"]) for r in baseline_results]
    baseline_missing = [len(r["missing_keywords"]) for r in baseline_results]
    contextual_found = [len(r["found_keywords"]) for r in contextual_results]
    contextual_missing = [len(r["missing_keywords"]) for r in contextual_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(questions_short))
    width = 0.35

    # Baseline
    ax1 = axes[0]
    ax1.bar(x, baseline_found, width, label='Found', color='#2ecc71', edgecolor='black')
    ax1.bar(x, baseline_missing, width, bottom=baseline_found, label='Missing', color='#e74c3c', edgecolor='black')
    ax1.set_ylabel('Number of Keywords')
    ax1.set_title('Baseline: Keyword Matching', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(questions_short, fontsize=9)
    ax1.legend()

    # Contextual
    ax2 = axes[1]
    ax2.bar(x, contextual_found, width, label='Found', color='#2ecc71', edgecolor='black')
    ax2.bar(x, contextual_missing, width, bottom=contextual_found, label='Missing', color='#e74c3c', edgecolor='black')
    ax2.set_ylabel('Number of Keywords')
    ax2.set_title('Contextual: Keyword Matching', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(questions_short, fontsize=9)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "keywords_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: keywords_analysis.png")


def main():
    """Main function to generate all benchmark visualizations."""
    print("\n" + "=" * 60)
    print("  BENCHMARK VISUALIZATION")
    print("=" * 60)

    # Load data
    print("\nLoading benchmark results...")
    data = load_benchmark_results()

    if data is None:
        return

    print(f"  Loaded results from: {data.get('timestamp', 'unknown')}")

    # Create output directory
    output_dir = create_output_dir()
    print(f"  Output directory: {output_dir}")

    # Generate charts
    print("\nGenerating visualizations...")

    try:
        plot_overall_comparison(data, output_dir)
        plot_per_question_scores(data, output_dir)
        plot_faithfulness_comparison(data, output_dir)
        plot_latency_comparison(data, output_dir)
        plot_radar_chart(data, output_dir)
        plot_improvement_waterfall(data, output_dir)
        plot_keywords_analysis(data, output_dir)
        plot_summary_dashboard(data, output_dir)

        print("\n" + "=" * 60)
        print("  VISUALIZATION COMPLETE")
        print("=" * 60)
        print(f"\nAll charts saved to: {output_dir}")
        print("\nGenerated files:")
        for f in sorted(output_dir.glob("*.png")):
            print(f"  • {f.name}")

        # Print summary
        baseline = data["baseline"]
        contextual = data["contextual"]

        # Helper to get metric with fallback
        def get_metric(d, new_key, old_key):
            return d.get(new_key, d.get(old_key, 0))

        kw_improvement = (get_metric(contextual, "avg_keyword_score", "avg_score") -
                          get_metric(baseline, "avg_keyword_score", "avg_score")) * 100
        faith_improvement = (contextual.get("avg_faithfulness", 0) -
                             baseline.get("avg_faithfulness", 0)) * 100
        prec_improvement = (contextual.get("avg_precision", 0) -
                            baseline.get("avg_precision", 0)) * 100

        print("\n" + "-" * 60)
        print("SUMMARY:")

        # Show accuracy if available
        if "avg_accuracy" in baseline or "avg_accuracy" in contextual:
            acc_improvement = (contextual.get("avg_accuracy", 0) - baseline.get("avg_accuracy", 0)) * 100
            print(f"  Ground Truth Acc:   Baseline {baseline.get('avg_accuracy', 0)*100:.1f}% → Contextual {contextual.get('avg_accuracy', 0)*100:.1f}% ({acc_improvement:+.1f}%)")

        print(f"  Keyword Score:      Baseline {get_metric(baseline, 'avg_keyword_score', 'avg_score')*100:.1f}% → Contextual {get_metric(contextual, 'avg_keyword_score', 'avg_score')*100:.1f}% ({kw_improvement:+.1f}%)")
        print(f"  Faithfulness:       Baseline {baseline.get('avg_faithfulness', 0)*100:.1f}% → Contextual {contextual.get('avg_faithfulness', 0)*100:.1f}% ({faith_improvement:+.1f}%)")
        print(f"  Retrieval Precision: Baseline {baseline.get('avg_precision', 0)*100:.1f}% → Contextual {contextual.get('avg_precision', 0)*100:.1f}% ({prec_improvement:+.1f}%)")
        print("-" * 60)

    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
