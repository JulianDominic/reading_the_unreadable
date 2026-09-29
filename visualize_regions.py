"""
Visualize detected regions on the newspaper image.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualize_detections(
    image_path: str,
    bbox_path: str,
    output_path: str = "output_visualization.png"
):
    """Draw bounding boxes on the image."""
    # Load image
    img = Image.open(image_path)

    # Load bounding boxes
    bbox_df = pd.read_parquet(bbox_path)

    # Create figure
    fig, ax = plt.subplots(1, figsize=(20, 30))
    ax.imshow(img)

    # Color map for classes
    color_map = {
        'title': 'red',
        'text': 'blue',
        'figure': 'green',
        'table': 'orange'
    }

    # Draw each bounding box
    for _, row in bbox_df.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        width = x2 - x1
        height = y2 - y1

        color = color_map.get(row['class'], 'yellow')

        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add reading order label
        ax.text(
            x1, y1 - 5,
            f"{row['reading_order']} ({row['class']})",
            color=color,
            fontsize=8,
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, pad=2)
        )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    visualize_detections(
        image_path="input_images/testing_page.png",
        bbox_path="data/periodical_bboxes/post_process_fill/custom_newspaper_1056.parquet",
        output_path="detected_regions.png"
    )


if __name__ == "__main__":
    main()
