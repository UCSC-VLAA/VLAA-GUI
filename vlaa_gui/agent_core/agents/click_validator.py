"""
Click Validation Agent for validating and correcting click coordinates.

This module provides functionality to:
1. Annotate screenshots with click point markers
2. Validate click locations using a vision-language model
3. Iteratively correct click coordinates until valid
"""

import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

from vlaa_gui.agent_core.core.mllm import LMMAgent
from vlaa_gui.agent_core.utils.common_utils import (
    call_llm_safe,
    annotate_screenshot_with_click,
)
from vlaa_gui.agent_core.memory.procedural_memory import PROCEDURAL_MEMORY

logger = logging.getLogger("desktopenv.agent")


def _crop_centered_and_resize(
    screenshot_bytes: bytes,
    center_px: Tuple[int, int],
    crop_ratio: float,
) -> Tuple[bytes, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Crop a window centered on `center_px` with (w,h)=crop_ratio*(W,H) and resize back to (W,H).

    Returns:
      zoomed_bytes, (left_px, top_px), (crop_w_px, crop_h_px), (img_w, img_h)
    """
    img = Image.open(io.BytesIO(screenshot_bytes))
    img_w, img_h = img.size

    crop_w_px = max(1, int(round(img_w * crop_ratio)))
    crop_h_px = max(1, int(round(img_h * crop_ratio)))
    crop_w_px = min(crop_w_px, img_w)
    crop_h_px = min(crop_h_px, img_h)

    cx, cy = center_px
    cx = max(0, min(cx, img_w - 1))
    cy = max(0, min(cy, img_h - 1))

    left_px = int(round(cx - crop_w_px / 2))
    top_px = int(round(cy - crop_h_px / 2))
    left_px = max(0, min(left_px, img_w - crop_w_px))
    top_px = max(0, min(top_px, img_h - crop_h_px))

    cropped = img.crop((left_px, top_px, left_px + crop_w_px, top_px + crop_h_px))
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:  # pragma: no cover
        resample = Image.BICUBIC
    zoomed = cropped.resize((img_w, img_h), resample=resample)

    output = io.BytesIO()
    zoomed.save(output, format=img.format or "PNG")
    return (
        output.getvalue(),
        (left_px, top_px),
        (crop_w_px, crop_h_px),
        (img_w, img_h),
    )


def _get_next_screenshot_path(directory: Path, suffix: str = ".png") -> Path:
    """Return the next available integer-named path in `directory` (e.g., 1.png, 2.png, ...)."""
    max_id = 0
    try:
        for p in directory.glob(f"*{suffix}"):
            stem = p.stem
            if stem.isdigit():
                max_id = max(max_id, int(stem))
    except Exception:
        # If directory listing fails for any reason, fall back to 1.png
        max_id = 0
    return directory / f"{max_id + 1}{suffix}"


def _save_annotated_screenshot(
    annotated_screenshot: bytes, out_dir: str = "screenshots"
) -> Optional[str]:
    """Save annotated screenshot bytes to `out_dir` using an integer filename. Returns saved path or None."""
    try:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = _get_next_screenshot_path(directory, suffix=".png")
        out_path.write_bytes(annotated_screenshot)
        return str(out_path)
    except Exception as e:
        logger.warning(f"Failed to save annotated screenshot to '{out_dir}': {e}")
        return None


class ClickValidatorAgent:
    """
    An agent that validates click locations and can iteratively correct them
    using visual feedback from a grounding model.
    """

    def __init__(
        self,
        engine_params: Dict,
        max_retries: int = 3,
        enable_zoom_grounding: bool = False,
        zoom_grounding_crop_ratio: float = 1.0,
    ):
        """
        Initialize the click validator agent.

        Args:
            engine_params: Configuration for the LLM engine
            max_retries: Maximum number of attempts to correct a click location
        """
        self.engine_params = engine_params
        self.max_retries = max_retries
        self.validator_agent = LMMAgent(
            engine_params=engine_params,
            system_prompt=PROCEDURAL_MEMORY.CLICK_VALIDATOR_PROMPT,
        )
        self.enable_zoom_grounding = enable_zoom_grounding
        self.zoom_grounding_crop_ratio = zoom_grounding_crop_ratio

    def _parse_validation_response(self, response: str) -> Dict:
        """Parse the JSON response from the validator agent."""
        import json
        import re

        default = {
            "valid": False,
            "reason": "Unable to parse validation response",
            "suggestion": "",
        }

        if not response:
            return default

        cleaned = response.strip()
        fence_match = re.fullmatch(
            r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL
        )
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return default

        if not isinstance(data, dict):
            return default

        # Support multiple schemas:
        # - New: {"results": true/false, "suggestion": {...}}
        # - Old: {"valid": ..., "reason": "...", "suggestion": "..."}
        # - Transitional: {"result": {"valid": ...}, "thoughts": {...}}
        results_obj = data.get("results")
        if isinstance(results_obj, dict):
            valid = bool(results_obj.get("valid", False))
        elif isinstance(results_obj, bool):
            valid = bool(results_obj)
        elif isinstance(results_obj, str):
            valid = results_obj.strip().lower() in {"true", "valid", "yes", "y", "1"}
        else:
            nested_result = data.get("result")
            if isinstance(nested_result, dict):
                valid = bool(nested_result.get("valid", False))
            else:
                valid = bool(data.get("valid", False))

        suggestion_obj = data.get("suggestion")
        if suggestion_obj is None and "thoughts" in data:
            suggestion_obj = {
                "reason": data.get("reason", ""),
                "thoughts": data.get("thoughts"),
            }

        if isinstance(suggestion_obj, (dict, list)):
            suggestion_text = json.dumps(suggestion_obj, ensure_ascii=False)
        elif suggestion_obj is None:
            suggestion_text = str(data.get("suggestion", "") or "")
        else:
            suggestion_text = str(suggestion_obj)

        reason = str(data.get("reason", "") or "").strip()
        if not reason and isinstance(suggestion_obj, dict):
            for key in ("summary", "diagnosis", "reason"):
                value = suggestion_obj.get(key)
                if value:
                    reason = str(value).strip()
                    break
        if not reason and suggestion_text:
            reason = suggestion_text.splitlines()[0].strip()

        parsed = {"valid": valid, "reason": reason, "suggestion": suggestion_text}
        if "thoughts" in data:
            parsed["thoughts"] = data.get("thoughts")
        return parsed

    def _parse_coordinate_response(
        self, response: str, image_size: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Parse and bounds-check a coordinate response like "x,y" or "x y"."""
        import re

        if not response:
            return None

        # Try comma-separated first (e.g., "18,953"), then space-separated (e.g., "18 953")
        match = re.search(r"(-?\d+)\s*[,\s]\s*(-?\d+)", response)
        if not match:
            return None

        x = int(match.group(1))
        y = int(match.group(2))

        width, height = image_size
        if not (0 <= x < width and 0 <= y < height):
            return None

        return (x, y)

    def validate_click(
        self,
        screenshot_bytes: bytes,
        click_coords: Tuple[int, int],
        element_description: str,
        debug: bool = False,
    ) -> Dict:
        """
        Validate whether a click at the given coordinates correctly targets the intended element.

        Args:
            screenshot_bytes: The screenshot image as bytes
            click_coords: (x, y) coordinates of the proposed click
            element_description: Description of the element that should be clicked

        Returns:
            Dictionary with validation result:
            {
                "valid": bool,
                "reason": str,
                "suggestion": str,
                "annotated_screenshot": bytes
            }
        """
        # Annotate the screenshot with the click location
        annotated_screenshot = annotate_screenshot_with_click(
            screenshot_bytes, click_coords
        )

        # Optionally save the annotated screenshot for debugging
        if debug:
            saved_path = _save_annotated_screenshot(
                annotated_screenshot, out_dir="debug/screenshots"
            )
            if saved_path:
                logger.debug(f"Saved annotated screenshot: {saved_path}")

        # Reset and query the validator
        self.validator_agent.reset()
        self.validator_agent.add_message(
            text_content=PROCEDURAL_MEMORY.construct_click_validator_prompt(
                element_description
            ),
            image_content=annotated_screenshot,
            role="user",
        )

        response = call_llm_safe(self.validator_agent, temperature=0.0)
        result = self._parse_validation_response(response)
        result["annotated_screenshot"] = annotated_screenshot
        if debug and saved_path:
            result["annotated_screenshot_path"] = saved_path

        logger.info(
            f"Click validation result: valid={result['valid']}, reason={result['reason']}"
        )

        return result

    def validate_and_correct_click(
        self,
        screenshot_bytes: bytes,
        initial_coords: Tuple[int, int],
        element_description: str,
        grounding_model: LMMAgent,
        resize_coordinates_fn=None,
        resize_screenshot_fn=None,
        resize_target_size: Optional[Tuple[int, int]] = None,
        debug: bool = False,
    ) -> Tuple[Tuple[int, int], Dict]:
        """
        Validate a click and iteratively correct it if invalid.

        This method will:
        1. Validate the initial click coordinates
        2. If invalid, ask the grounding model to re-locate the element with the annotated screenshot
        3. Repeat until valid or max_retries is reached

        Args:
            screenshot_bytes: The screenshot image as bytes
            initial_coords: Initial (x, y) coordinates to validate
            element_description: Description of the target element
            grounding_model: The grounding model to use for re-locating elements
            resize_coordinates_fn: Optional function to resize coordinates from grounding model space
            resize_screenshot_fn: Optional function to resize screenshots for grounding
            resize_target_size: Optional target size for grounding screenshots
            debug: Whether to enable debug mode (e.g., saving annotated screenshots)

        Returns:
            Tuple of (final_coords, validation_history)
            - final_coords: The corrected (x, y) coordinates
            - validation_history: Dict with validation attempts and results
        """
        import re

        current_coords = initial_coords
        history = {
            "attempts": [],
            "final_valid": False,
            "total_retries": 0,
        }

        image_size = None
        try:
            image_size = Image.open(io.BytesIO(screenshot_bytes)).size
        except Exception:
            image_size = None

        correction_image_size = (
            resize_target_size if resize_screenshot_fn and resize_target_size else None
        )
        if correction_image_size is None:
            correction_image_size = image_size

        for attempt in range(self.max_retries + 1):
            # Validate current coordinates
            validation_result = self.validate_click(
                screenshot_bytes, current_coords, element_description, debug=debug
            )

            history["attempts"].append(
                {
                    "coords": current_coords,
                    "valid": validation_result["valid"],
                    "reason": validation_result["reason"],
                    "suggestion": validation_result["suggestion"],
                }
            )

            if validation_result["valid"]:
                history["final_valid"] = True
                logger.info(
                    f"Click validated successfully at {current_coords} after {attempt} retries"
                )
                return current_coords, history

            if attempt >= self.max_retries:
                logger.warning(
                    f"Max retries ({self.max_retries}) reached. Using last coordinates: {current_coords}"
                )
                break

            # Use annotated screenshot to ask grounding model for corrected coordinates
            logger.info(
                f"Click validation failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                f"Reason: {validation_result['reason']}. Re-grounding..."
            )

            # Reset grounding model and query with annotated screenshot
            grounding_screenshot = validation_result["annotated_screenshot"]
            if resize_screenshot_fn and resize_target_size:
                grounding_screenshot = resize_screenshot_fn(
                    grounding_screenshot, resize_target_size
                )
            grounding_model.reset()
            grounding_model.add_message(
                text_content=PROCEDURAL_MEMORY.construct_click_corrector_prompt(
                    validation_result, element_description
                ),
                image_content=grounding_screenshot,
                put_text_last=True,
            )

            response = call_llm_safe(grounding_model)
            logger.debug(f"Grounding model correction response: {response}")

            # Parse new coordinates
            if correction_image_size is not None:
                parsed = self._parse_coordinate_response(
                    response, correction_image_size
                )
            else:
                match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", response or "")
                parsed = (int(match.group(1)), int(match.group(2))) if match else None

            if parsed is not None:
                new_coords = [parsed[0], parsed[1]]

                # Apply coordinate resize if provided
                if resize_coordinates_fn:
                    new_coords = resize_coordinates_fn(new_coords)

                current_coords = tuple(new_coords)
                history["total_retries"] = attempt + 1
            else:
                logger.warning(f"Could not parse coordinates from response: {response}")
                # Keep current coordinates and try again

        return current_coords, history

    def validate_and_correct_click_with_zoom(
        self,
        validation_screenshot_bytes: bytes,
        grounding_screenshot_bytes: bytes,
        initial_model_coords: Tuple[int, int],
        element_description: str,
        grounding_model: LMMAgent,
        model_coordinate_space: Tuple[int, int],
        resize_coordinates_fn,
        crop_ratio: float = 0.5,
        debug: bool = False,
    ) -> Tuple[Tuple[int, int], Dict]:
        """
        Integrated click verifier + zoom-in refinement loop.

        Flow:
          1) Validate the coarse coordinate (in OS coords).
          2) If invalid, crop a zoomed region centered on the (incorrect) click, and draw the red circle.
          3) Re-ground on the zoomed crop until the verifier says it's correct or max_retries is reached.
        """
        import re

        coord_w, coord_h = model_coordinate_space
        if coord_w <= 0 or coord_h <= 0:
            raise ValueError(
                f"Invalid model coordinate space: {model_coordinate_space}"
            )

        try:
            img = Image.open(io.BytesIO(grounding_screenshot_bytes))
            img_w, img_h = img.size
        except Exception:
            # Fallback to non-zoom correction loop if we can't open the grounding image.
            initial_os = tuple(
                resize_coordinates_fn(
                    [initial_model_coords[0], initial_model_coords[1]]
                )
            )
            return self.validate_and_correct_click(
                screenshot_bytes=validation_screenshot_bytes,
                initial_coords=initial_os,
                element_description=element_description,
                grounding_model=grounding_model,
                resize_coordinates_fn=resize_coordinates_fn,
                resize_screenshot_fn=None,
                resize_target_size=None,
                debug=debug,
            )

        def _clamp_model(x: int, y: int) -> Tuple[int, int]:
            return (
                max(0, min(int(x), coord_w - 1)),
                max(0, min(int(y), coord_h - 1)),
            )

        current_model_coords = _clamp_model(*initial_model_coords)

        history: Dict = {
            "attempts": [],
            "final_valid": False,
            "total_retries": 0,
            "mode": "zoom_on_failure",
        }

        for attempt in range(self.max_retries + 1):
            os_coords = tuple(
                resize_coordinates_fn(
                    [current_model_coords[0], current_model_coords[1]]
                )
            )

            validation_result = self.validate_click(
                validation_screenshot_bytes,
                os_coords,
                element_description,
                debug=debug,
            )

            history["attempts"].append(
                {
                    "model_coords": current_model_coords,
                    "coords": os_coords,
                    "valid": validation_result["valid"],
                    "reason": validation_result["reason"],
                    "suggestion": validation_result["suggestion"],
                }
            )

            if validation_result["valid"]:
                history["final_valid"] = True
                return os_coords, history

            if attempt >= self.max_retries:
                break

            # Crop around the incorrect click (in grounding-image pixel space), then annotate that incorrect point.
            click_px = (
                int(round(current_model_coords[0] * img_w / coord_w)),
                int(round(current_model_coords[1] * img_h / coord_h)),
            )
            click_px = (
                max(0, min(click_px[0], img_w - 1)),
                max(0, min(click_px[1], img_h - 1)),
            )

            if not self.enable_zoom_grounding:
                self.zoom_grounding_crop_ratio = 1.0  # Disable zooming if not enabled.

            try:
                zoomed_bytes, (left_px, top_px), (crop_w_px, crop_h_px), _ = (
                    _crop_centered_and_resize(
                        screenshot_bytes=grounding_screenshot_bytes,
                        center_px=click_px,
                        crop_ratio=self.zoom_grounding_crop_ratio,
                    )
                )
            except Exception:
                # If zoom/crop fails, fall back to a full-image correction query.
                grounding_model.reset()
                grounding_model.add_message(
                    text_content=PROCEDURAL_MEMORY.construct_click_corrector_prompt(
                        validation_result, element_description
                    ),
                    image_content=validation_result["annotated_screenshot"],
                    put_text_last=True,
                )
                response = call_llm_safe(grounding_model)
                parsed = self._parse_coordinate_response(
                    response, model_coordinate_space
                )
                if parsed is not None:
                    current_model_coords = _clamp_model(parsed[0], parsed[1])
                    history["total_retries"] = attempt + 1
                continue

            # Map the incorrect click point into zoomed-image pixel coordinates (zoomed image has size img_w x img_h).
            rel_x = click_px[0] - left_px
            rel_y = click_px[1] - top_px
            zoom_click_px = (
                int(round(rel_x * img_w / max(crop_w_px, 1))),
                int(round(rel_y * img_h / max(crop_h_px, 1))),
            )
            zoom_click_px = (
                max(0, min(zoom_click_px[0], img_w - 1)),
                max(0, min(zoom_click_px[1], img_h - 1)),
            )
            annotated_zoomed = annotate_screenshot_with_click(
                zoomed_bytes, zoom_click_px
            )

            if debug:
                _save_annotated_screenshot(
                    annotated_zoomed, out_dir="debug/screenshots_zoom"
                )

            grounding_model.reset()
            grounding_model.add_message(
                text_content=(
                    "This is a zoomed-in crop around the previous (incorrect) click.\n"
                    + PROCEDURAL_MEMORY.construct_click_corrector_prompt(
                        validation_result, element_description
                    )
                ),
                image_content=annotated_zoomed,
                put_text_last=True,
            )
            response = call_llm_safe(grounding_model)
            logger.debug(f"Grounding model correction response (zoomed): {response}")

            parsed_zoom = self._parse_coordinate_response(
                response, model_coordinate_space
            )
            if parsed_zoom is None:
                match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", response or "")
                parsed_zoom = (
                    (int(match.group(1)), int(match.group(2))) if match else None
                )

            if parsed_zoom is None:
                continue

            zoom_x, zoom_y = parsed_zoom
            scale_x = crop_w_px / img_w
            scale_y = crop_h_px / img_h
            left_coord = left_px / img_w * coord_w
            top_coord = top_px / img_h * coord_h

            new_model_x = int(round(left_coord + zoom_x * scale_x))
            new_model_y = int(round(top_coord + zoom_y * scale_y))
            current_model_coords = _clamp_model(new_model_x, new_model_y)
            history["total_retries"] = attempt + 1

        # Return last OS coords even if invalid.
        final_os = tuple(
            resize_coordinates_fn([current_model_coords[0], current_model_coords[1]])
        )
        return final_os, history
