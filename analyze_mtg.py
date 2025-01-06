import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import argparse


@dataclass
class MTGCard:
    ref: str
    name: str
    card_type: str
    full_card_type: str
    color: List[str]
    use_percent: float


@dataclass
class YearStats:
    year: int
    top_card_type: str
    top_card_colors: List[str]
    top_card_use_percent: float
    target_type_in_top3: int
    target_color_in_top3: int
    num_target_type: int
    num_target_color: int
    cards_above_threshold: int


def analyze_year(
    year: str,
    cards: List[MTGCard],
    target_type: str,
    target_color: str,
    usage_threshold: float,
) -> YearStats:
    """Analyze card statistics for a given year."""
    # Sort cards by use_percent to ensure proper ranking
    cards.sort(key=lambda x: x.use_percent, reverse=True)

    top_card = cards[0]

    return YearStats(
        year=int(year),
        top_card_type=top_card.card_type,
        top_card_colors=top_card.color,
        top_card_use_percent=top_card.use_percent,
        target_type_in_top3=sum(1 for c in cards[:3] if c.card_type == target_type),
        target_color_in_top3=sum(1 for c in cards[:3] if target_color in c.color),
        num_target_type=sum(1 for c in cards if c.card_type == target_type),
        num_target_color=sum(1 for c in cards if target_color in c.color),
        cards_above_threshold=sum(1 for c in cards if c.use_percent >= usage_threshold),
    )


def calculate_probabilities(
    yearly_stats: List[YearStats],
    target_type: str,
    target_color: str,
    usage_threshold: float,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Calculate both simple and weighted probabilities from yearly statistics."""
    if weights is None:
        # Equal weights for simple probability
        weights = [1 / len(yearly_stats)] * len(yearly_stats)

    weighted_p_type = sum(
        weight * (1 if stat.top_card_type == target_type else 0)
        for weight, stat in zip(weights, yearly_stats)
    )

    weighted_p_color = sum(
        weight * (1 if target_color in stat.top_card_colors else 0)
        for weight, stat in zip(weights, yearly_stats)
    )

    weighted_p_threshold = sum(
        weight * (1 if stat.top_card_use_percent >= usage_threshold else 0)
        for weight, stat in zip(weights, yearly_stats)
    )

    # Calculate simple probabilities
    simple_p_type = sum(
        1 for stat in yearly_stats if stat.top_card_type == target_type
    ) / len(yearly_stats)
    simple_p_color = sum(
        1 for stat in yearly_stats if target_color in stat.top_card_colors
    ) / len(yearly_stats)
    simple_p_threshold = sum(
        1 for stat in yearly_stats if stat.top_card_use_percent >= usage_threshold
    ) / len(yearly_stats)

    return {
        "weighted_type": weighted_p_type,
        "weighted_color": weighted_p_color,
        "weighted_threshold": weighted_p_threshold,
        "simple_type": simple_p_type,
        "simple_color": simple_p_color,
        "simple_threshold": simple_p_threshold,
    }


def plot_trends(
    yearly_stats: List[YearStats],
    target_type: str,
    target_color: str,
    usage_threshold: float,
):
    """Create visualizations of the trends."""
    years = [stat.year for stat in yearly_stats]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Top card usage percentage
    ax1.plot(years, [stat.top_card_use_percent for stat in yearly_stats], marker="o")
    ax1.set_title("Top Card Usage Percentage by Year")
    ax1.set_ylabel("Usage %")
    ax1.axhline(y=usage_threshold, color="r", linestyle="--", alpha=0.5)

    # Plot 2: Target type representation
    ax2.plot(
        years,
        [stat.num_target_type for stat in yearly_stats],
        marker="o",
        label=f"Total {target_type}s",
    )
    ax2.plot(
        years,
        [stat.target_type_in_top3 for stat in yearly_stats],
        marker="s",
        label=f"{target_type}s in Top 3",
    )
    ax2.set_title(f"{target_type} Cards Representation")
    ax2.legend()

    # Plot 3: Target color representation
    ax3.plot(
        years,
        [stat.num_target_color for stat in yearly_stats],
        marker="o",
        label=f"Total {target_color}",
    )
    ax3.plot(
        years,
        [stat.target_color_in_top3 for stat in yearly_stats],
        marker="s",
        label=f"{target_color} in Top 3",
    )
    ax3.set_title(f"{target_color} Cards Representation")
    ax3.legend()

    # Plot 4: Percentage representation
    ax4.plot(
        years,
        [stat.num_target_type / 20 * 100 for stat in yearly_stats],
        marker="o",
        label=f"{target_type} %",
    )
    ax4.plot(
        years,
        [stat.num_target_color / 20 * 100 for stat in yearly_stats],
        marker="s",
        label=f"{target_color} %",
    )
    ax4.set_title("Percentage in Top 20")
    ax4.set_ylabel("%")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def print_analysis(
    yearly_stats: List[YearStats],
    probabilities: Dict[str, float],
    target_type: str,
    target_color: str,
    usage_threshold: float,
):
    """Print detailed analysis results."""
    print("\nYearly Statistics:")
    print("-----------------")
    for stat in yearly_stats:
        print(f"\n{stat.year}:")
        print(f"  Top card type: {stat.top_card_type}")
        print(f"  Top card colors: {', '.join(stat.top_card_colors)}")
        print(f"  Top card usage: {stat.top_card_use_percent:.1f}%")
        print(
            f"  {target_type}s: {stat.num_target_type} total, {stat.target_type_in_top3} in top 3"
        )
        print(
            f"  {target_color} cards: {stat.num_target_color} total, {stat.target_color_in_top3} in top 3"
        )
        print(f"  Cards ≥{usage_threshold}%: {stat.cards_above_threshold}")

    print("\nProbability Estimates for 2025:")
    print("------------------------------")
    print(f"Most used card being {target_type}:")
    print(f"  Simple: {probabilities['simple_type']:.1%}")
    print(f"  Weighted: {probabilities['weighted_type']:.1%}")

    print(f"\nMost used card being {target_color}:")
    print(f"  Simple: {probabilities['simple_color']:.1%}")
    print(f"  Weighted: {probabilities['weighted_color']:.1%}")

    print(f"\nMost used card ≥{usage_threshold}% usage:")
    print(f"  Simple: {probabilities['simple_threshold']:.1%}")
    print(f"  Weighted: {probabilities['weighted_threshold']:.1%}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Magic: The Gathering card usage statistics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="mtg_top.json",
        help="Path to the input JSON file (default: mtg_top.json)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="Instant",
        help="Target card type to analyze (default: Instant)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="Black",
        help="Target card color to analyze (default: Black)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=45.0,
        help="Usage percentage threshold (default: 45.0)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of recent years to analyze (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mtg_top_json = Path(args.input)
    with open(mtg_top_json, "r") as f:
        data = json.load(f)

    all_years = sorted(data.keys(), reverse=True)
    recent_years = all_years[: args.years]

    yearly_stats = []
    for year in recent_years:
        cards = [MTGCard(**card_data) for card_data in data[year]]
        yearly_stats.append(
            analyze_year(year, cards, args.type, args.color, args.threshold)
        )

    yearly_stats.sort(key=lambda x: x.year)

    weights = [
        (args.years - i) / sum(range(1, args.years + 1)) for i in range(args.years)
    ]
    probabilities = calculate_probabilities(
        yearly_stats, args.type, args.color, args.threshold, weights
    )

    print_analysis(yearly_stats, probabilities, args.type, args.color, args.threshold)

    plot_trends(yearly_stats, args.type, args.color, args.threshold)


if __name__ == "__main__":
    main()
