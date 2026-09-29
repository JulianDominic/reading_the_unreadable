"""
Run the newspaper OCR pipeline synchronously (no batch jobs).

This script processes newspaper images directly with synchronous API calls,
suitable for single pages or small batches.

Usage:
    python run_sync_pipeline.py
"""

import os
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from function_modules.pipeline import NewspaperPipeline
from function_modules.send_to_lm_functions import (
    crop_and_encode_boxes,
    process_image_with_api,
    convert_returned_streaming_to_dataframe,
    combine_article_segments,
    decompose_filenames
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_boxes_sync(
    bbox_df: pd.DataFrame,
    image_folder: str,
    prompt_dict: dict,
    model: str,
    output_path: str,
    cache_path: str = None
) -> pd.DataFrame:
    """
    Process bounding boxes synchronously with direct API calls.

    Args:
        bbox_df: DataFrame with bounding box information
        image_folder: Path to folder containing images
        prompt_dict: Dictionary mapping classes to prompts
        model: Model name to use
        output_path: Where to save results
        cache_path: Path to cache raw OCR results

    Returns:
        DataFrame with OCR results
    """
    # Check if we have cached raw results
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached OCR results from {cache_path}")
        combined_df = pd.read_parquet(cache_path)
    else:
        # Encode all images
        logger.info("Encoding image regions...")
        encoded_images = crop_and_encode_boxes(
            df=bbox_df,
            images_folder=image_folder,
            max_ratio=1.0,
            overlap_fraction=0.2,
            deskew=True,
            crop_image=True
        )

        logger.info(f"Processing {len(encoded_images)} image regions...")

        # Process each encoded image
        results = []
        for image_id, image_data in tqdm(encoded_images.items(), desc="OCR"):
            # Get appropriate prompt
            image_class = image_data.get("class", "text")
            prompt = prompt_dict.get(image_class, prompt_dict["text"])

            # Call API
            try:
                response = process_image_with_api(
                    image_base64=image_data["image"],
                    prompt=prompt,
                    model=model
                )

                # Convert to dataframe row
                df_row = convert_returned_streaming_to_dataframe(
                    response=response,
                    custom_id=image_id
                )
                results.append(df_row)

            except Exception as e:
                logger.error(f"Error processing {image_id}: {e}")
                continue

        # Combine all results
        logger.info("Combining results...")
        combined_df = pd.concat(results, ignore_index=True)

        # Cache the raw results
        if cache_path:
            logger.info(f"Caching raw OCR results to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            combined_df.to_parquet(cache_path, index=False)

    # Decompose filenames to extract metadata
    combined_df = decompose_filenames(combined_df)

    # Combine segments
    final_df = combine_article_segments(combined_df)

    # Save results
    logger.info(f"Saving results to {output_path}")
    final_df.to_parquet(output_path, index=False)

    return final_df


def main():
    """Run the synchronous pipeline."""

    # Configuration
    image_folder = "input_images"
    periodical_name = "custom_newspaper"

    # Check environment
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_base or not api_key:
        logger.error("Missing OPENAI_API_BASE or OPENAI_API_KEY!")
        return

    # Set litellm API base
    import litellm
    litellm.api_base = api_base
    litellm.api_key = api_key

    logger.info(f"Using API: {api_base}")

    # Check for images
    if not os.path.exists(image_folder):
        logger.error(f"Folder '{image_folder}' not found!")
        return

    png_files = list(Path(image_folder).glob("*.png"))
    if not png_files:
        logger.error(f"No PNG files in '{image_folder}'!")
        return

    logger.info(f"Found {len(png_files)} images")

    # Initialize pipeline
    pipeline = NewspaperPipeline()

    # Stage 1: Predict bounding boxes
    logger.info("\n[1/4] Detecting layout...")
    bbox_path = pipeline.predict_bounding_boxes(
        periodical=periodical_name,
        image_folder=image_folder
    )
    logger.info(f"✓ Boxes: {bbox_path}")

    # Stage 2: Post-process boxes
    logger.info("\n[2/4] Post-processing boxes...")
    processed_bbox_path = pipeline.postprocess_bounding_boxes(
        periodical=periodical_name
    )
    logger.info(f"✓ Processed: {processed_bbox_path}")

    # Load processed boxes
    bbox_df = pd.read_parquet(processed_bbox_path)

    # Configure prompts
    prompt_dict = {
        "text": "Transcribe the text from this newspaper image. Include linebreaks. Use plain text only, no markdown. Do not add commentary.",
        "figure": "Describe the graphic from this newspaper image. Do not add commentary.",
        "table": "Extract the table from this image as tab-separated values (TSV). Do not add commentary."
    }

    # Stage 3: Process with API
    logger.info("\n[3/4] Running OCR...")
    output_path = f"data/periodical_dataframes/raw/{periodical_name}_raw.parquet"
    cache_path = f"data/cache/{periodical_name}_ocr_cache.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result_df = process_boxes_sync(
        bbox_df=bbox_df,
        image_folder=image_folder,
        prompt_dict=prompt_dict,
        model="openai/chandra-ocr-2-vllm",
        # model="openai/qwen3-6-35b-a3b-vllm",
        output_path=output_path,
        cache_path=cache_path
    )

    logger.info(f"✓ OCR complete: {len(result_df)} text regions")

    # Stage 4: Post-process
    logger.info("\n[4/4] Post-processing text...")
    post_processed_path = f"data/periodical_dataframes/post_processed/{periodical_name}.parquet"
    os.makedirs(os.path.dirname(post_processed_path), exist_ok=True)

    # Load raw results
    raw_df = pd.read_parquet(output_path)

    # Clean up content
    raw_df["content"] = raw_df["content"].str.strip("`").str.replace("tsv", "", n=1)

    # Fix misclassifications (title -> text for long content)
    raw_df.loc[
        (raw_df["completion_tokens"] > 50) & (raw_df["class"] == "title"),
        "class",
    ] = "text"

    # Clean line breaks except for tables
    from function_modules.analysis_functions import remove_line_breaks
    raw_df.loc[raw_df["class"] != "table", "content"] = remove_line_breaks(
        raw_df.loc[raw_df["class"] != "table", "content"]
    )

    # Save post-processed dataframe
    raw_df.to_parquet(post_processed_path, index=False)
    logger.info(f"✓ Post-processed: {post_processed_path}")

    logger.info("\n" + "="*60)
    logger.info("COMPLETE!")
    logger.info("="*60)
    logger.info(f"Raw results: {output_path}")
    logger.info(f"Processed results: {post_processed_path}")
    logger.info("")
    logger.info("Load results with:")
    logger.info("  import pandas as pd")
    logger.info(f"  df = pd.read_parquet('{post_processed_path}')")
    logger.info("="*60)


if __name__ == "__main__":
    main()
