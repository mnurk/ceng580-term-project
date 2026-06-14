import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SUMMARY_PATH = Path("results/ablation_summary.csv")
PLOTS_DIR = Path("results/plots")

ABLATION_ORDER = [
    "full_reward",
    "no_error_penalty",
    "no_cost_penalty",
    "no_failure_penalty",
    "no_step_penalty",
    "low_success_reward",
    "high_success_reward",
]

ABLATION_LABELS = {
    "full_reward": "Full",
    "no_error_penalty": "No Error",
    "no_cost_penalty": "No Cost",
    "no_failure_penalty": "No Failure",
    "no_step_penalty": "No Step",
    "low_success_reward": "Low Success",
    "high_success_reward": "High Success",
}

METHOD_ORDER = ["heuristic", "q_learning", "sarsa"]
METHOD_LABELS = {
    "heuristic": "Heuristic",
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
}
METHOD_COLORS = {
    "heuristic": "#059669",
    "q_learning": "#2563EB",
    "sarsa": "#7C3AED",
}

PLOTS = [
    {
        "metric": "success_rate",
        "title": "V3 Reward Ablation: Success Rate",
        "filename": "ablation_success_rate.png",
        "y_label": "Success rate",
        "fixed_range": (0.0, 0.65),
    },
    {
        "metric": "average_reward",
        "title": "V3 Reward Ablation: Average Reward",
        "filename": "ablation_average_reward.png",
        "y_label": "Average reward",
        "fixed_range": None,
    },
]


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def get_font(size, bold=False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass

    return ImageFont.load_default()


def load_summary(path):
    with path.open("r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    indexed = {}
    for row in rows:
        indexed[(row["ablation"], row["method"])] = row

    return indexed


def metric_values(summary, metric):
    values = []
    ci_values = []

    for ablation in ABLATION_ORDER:
        for method in METHOD_ORDER:
            row = summary[(ablation, method)]
            values.append(float(row[f"{metric}_mean"]))
            ci_values.append(float(row[f"{metric}_ci95"]))

    return values, ci_values


def choose_range(values, ci_values, fixed_range):
    if fixed_range is not None:
        return fixed_range

    lower = min(value - ci for value, ci in zip(values, ci_values))
    upper = max(value + ci for value, ci in zip(values, ci_values))
    padding = max((upper - lower) * 0.12, 0.5)

    return lower - padding, upper + padding


def draw_rotated_label(image, position, text, font, fill):
    label_width = max(1, int(font.getlength(text)) + 8)
    label_height = font.size + 8
    label = Image.new("RGBA", (label_width, label_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label)
    draw.text((4, 2), text, font=font, fill=fill)
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, position, rotated)


def draw_ablation_chart(summary, plot_config):
    width = 1900
    height = 980
    margin_left = 150
    margin_right = 60
    margin_top = 90
    margin_bottom = 220
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    metric = plot_config["metric"]
    values, ci_values = metric_values(summary, metric)
    y_min, y_max = choose_range(values, ci_values, plot_config["fixed_range"])

    if y_min > 0:
        y_min = 0

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(30, bold=True)
    axis_font = get_font(18)
    tick_font = get_font(16)
    legend_font = get_font(17)
    value_font = get_font(12)

    def y_to_pixel(value):
        if y_max == y_min:
            return margin_top + chart_height
        ratio = (value - y_min) / (y_max - y_min)
        return margin_top + chart_height - ratio * chart_height

    draw.text((margin_left, 30), plot_config["title"], font=title_font, fill="#111827")

    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = y_to_pixel(value)
        draw.line((margin_left, y, width - margin_right, y), fill="#E5E7EB", width=1)
        draw.text((26, y - 10), f"{value:.2f}", font=tick_font, fill="#374151")

    zero_y = y_to_pixel(0)
    if margin_top <= zero_y <= margin_top + chart_height:
        draw.line((margin_left, zero_y, width - margin_right, zero_y), fill="#111827", width=2)

    draw.line((margin_left, margin_top, margin_left, margin_top + chart_height), fill="#111827", width=2)
    draw.line(
        (margin_left, margin_top + chart_height, width - margin_right, margin_top + chart_height),
        fill="#111827",
        width=2,
    )

    draw_rotated_label(
        image,
        (84, margin_top + chart_height // 2 - 80),
        plot_config["y_label"],
        axis_font,
        "#111827",
    )

    group_gap = 46
    group_width = (chart_width - group_gap * (len(ABLATION_ORDER) - 1)) / len(ABLATION_ORDER)
    bar_gap = 8
    bar_width = (group_width - bar_gap * (len(METHOD_ORDER) - 1)) / len(METHOD_ORDER)

    for ablation_index, ablation in enumerate(ABLATION_ORDER):
        group_x = margin_left + ablation_index * (group_width + group_gap)
        group_center = group_x + group_width / 2

        for method_index, method in enumerate(METHOD_ORDER):
            row = summary[(ablation, method)]
            mean_value = float(row[f"{metric}_mean"])
            ci_value = float(row[f"{metric}_ci95"])
            x0 = group_x + method_index * (bar_width + bar_gap)
            x1 = x0 + bar_width
            y0 = y_to_pixel(max(mean_value, 0))
            y1 = y_to_pixel(min(mean_value, 0))

            color = hex_to_rgb(METHOD_COLORS[method])
            draw.rectangle((x0, y0, x1, y1), fill=color)

            ci_top = y_to_pixel(mean_value + ci_value)
            ci_bottom = y_to_pixel(mean_value - ci_value)
            error_x = (x0 + x1) / 2
            draw.line((error_x, ci_top, error_x, ci_bottom), fill="#111827", width=2)
            draw.line((error_x - 6, ci_top, error_x + 6, ci_top), fill="#111827", width=2)
            draw.line((error_x - 6, ci_bottom, error_x + 6, ci_bottom), fill="#111827", width=2)

            value_text = f"{mean_value:.2f}"
            text_width = value_font.getlength(value_text)
            text_y = y0 - 18 if mean_value >= 0 else y1 + 6
            draw.text(
                (error_x - text_width / 2, text_y),
                value_text,
                font=value_font,
                fill="#111827",
            )

        label = ABLATION_LABELS[ablation]
        label_width = axis_font.getlength(label)
        draw.text(
            (group_center - label_width / 2, margin_top + chart_height + 28),
            label,
            font=axis_font,
            fill="#111827",
        )

    legend_x = margin_left
    legend_y = height - 80
    for method in METHOD_ORDER:
        color = hex_to_rgb(METHOD_COLORS[method])
        draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 24), fill=color)
        draw.text(
            (legend_x + 34, legend_y + 1),
            METHOD_LABELS[method],
            font=legend_font,
            fill="#111827",
        )
        legend_x += 210

    return image


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Run experiments.run_ablation_study first."
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary(SUMMARY_PATH)

    written_files = []
    for plot_config in PLOTS:
        image = draw_ablation_chart(summary, plot_config)
        output_path = PLOTS_DIR / plot_config["filename"]
        image.save(output_path)
        written_files.append(output_path)

    print("Ablation plots written:")
    for path in written_files:
        print(path)


if __name__ == "__main__":
    main()
