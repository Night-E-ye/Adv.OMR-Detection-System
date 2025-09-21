import cv2
import numpy as np
import json
from typing import Dict, Tuple, List, Optional
import math


class SuperiorOMRProcessor:
    def __init__(self, template_config: Dict = None):
        self.template = template_config or self.get_default_template()
        self.debug_mode = True

    def get_default_template(self) -> Dict:
        return {
            "bubble_params": {
                "min_radius": 6,
                "max_radius": 20,
                "dp": 1,
                "param1": 30,
                "param2": 15,
                "min_distance": 25
            },
            "preprocessing": {
                "gaussian_blur": (3, 3),
                "threshold_value": 127,
                "morph_kernel": (2, 2)
            },
            "fill_detection": {
                "method": "multi_criteria",  # Use multiple criteria
                "intensity_threshold": 140,  # Darker pixels threshold
                "fill_ratio_threshold": 0.35,  # % of dark pixels needed
                "edge_penalty": True,  # Penalize edge artifacts
                "adaptive_threshold": True  # Use adaptive thresholding
            }
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blur_params = self.template["preprocessing"]["gaussian_blur"]
        blurred = cv2.GaussianBlur(gray, blur_params, 0)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        _, binary = cv2.threshold(enhanced, self.template["preprocessing"]["threshold_value"], 255,
                                  cv2.THRESH_BINARY_INV)

        kernel = np.ones(self.template["preprocessing"]["morph_kernel"], np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    def detect_bubbles_contour_method(self, binary_image: np.ndarray) -> List[Tuple[int, int, int]]:
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        min_area = 40  # Slightly smaller minimum
        max_area = 1000  # Slightly larger maximum

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * math.pi * area / (perimeter * perimeter)

                # More lenient circularity for printed circles
                if 0.2 < circularity < 1.5:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    bubbles.append((int(x), int(y), int(radius)))

        return bubbles

    def detect_bubbles(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        bubbles_contour = self.detect_bubbles_contour_method(image)

        # Remove duplicates
        unique_bubbles = []
        for bubble in bubbles_contour:
            x, y, r = bubble
            is_duplicate = False

            for existing in unique_bubbles:
                ex, ey, er = existing
                distance = math.sqrt((x - ex) ** 2 + (y - ey) ** 2)

                if distance < 18:  # Slightly smaller distance threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_bubbles.append(bubble)

        return unique_bubbles

    def advanced_fill_detection(self, original_image: np.ndarray, x: int, y: int, radius: int) -> Tuple[bool, float]:
        """
        Advanced fill detection using multiple criteria
        Returns: (is_filled, confidence_score)
        """
        if len(original_image.shape) == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_image.copy()

        # Create multiple masks for analysis
        inner_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        outer_mask = np.zeros(gray.shape[:2], dtype=np.uint8)

        # Inner circle (core of bubble)
        inner_radius = max(1, radius - 4)
        cv2.circle(inner_mask, (x, y), inner_radius, 255, -1)

        # Outer ring (edge detection)
        cv2.circle(outer_mask, (x, y), radius, 255, -1)
        cv2.circle(outer_mask, (x, y), inner_radius, 0, -1)  # Remove inner part

        # Extract regions
        inner_region = gray[inner_mask > 0]
        outer_region = gray[outer_mask > 0]

        if len(inner_region) == 0:
            return False, 0.0

        # Criterion 1: Mean intensity (darker = more filled)
        inner_mean = np.mean(inner_region)
        intensity_score = max(0, (255 - inner_mean) / 255)  # Normalize to 0-1

        # Criterion 2: Percentage of dark pixels
        fill_params = self.template["fill_detection"]
        dark_pixels = np.sum(inner_region < fill_params["intensity_threshold"])
        total_pixels = len(inner_region)
        dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

        # Criterion 3: Contrast between inner and outer regions
        contrast_score = 0
        if len(outer_region) > 0:
            outer_mean = np.mean(outer_region)
            contrast = abs(inner_mean - outer_mean)
            contrast_score = min(1.0, contrast / 100)  # Normalize

        # Criterion 4: Standard deviation (filled bubbles have lower std)
        std_score = max(0, (50 - np.std(inner_region)) / 50)  # Normalize

        # Combine criteria with weights
        weights = {
            'intensity': 0.4,
            'dark_ratio': 0.3,
            'contrast': 0.2,
            'std': 0.1
        }

        confidence = (
                weights['intensity'] * intensity_score +
                weights['dark_ratio'] * (1 if dark_ratio > fill_params["fill_ratio_threshold"] else 0) +
                weights['contrast'] * contrast_score +
                weights['std'] * std_score
        )

        # Decision threshold
        is_filled = confidence > 0.5

        if self.debug_mode and is_filled:
            print(f"Bubble at ({x},{y}): intensity={inner_mean:.1f}, dark_ratio={dark_ratio:.2f}, "
                  f"contrast={contrast_score:.2f}, confidence={confidence:.2f}")

        return is_filled, confidence

    def organize_bubbles_into_grid(self, bubbles: List[Tuple[int, int, int]]) -> Dict[
        int, Dict[str, Tuple[int, int, int]]]:
        """Organize bubbles into question rows with improved grouping"""
        if not bubbles:
            return {}

        # Sort all bubbles by Y coordinate first (top to bottom)
        sorted_by_y = sorted(bubbles, key=lambda b: b[1])

        # Group bubbles by Y coordinate (same row) with adaptive tolerance
        question_rows = []
        current_row = [sorted_by_y[0]]

        for bubble in sorted_by_y[1:]:
            # Calculate adaptive tolerance based on bubble size
            avg_radius = sum(b[2] for b in current_row) / len(current_row)
            y_tolerance = max(15, int(avg_radius * 0.8))  # Adaptive tolerance

            if abs(bubble[1] - current_row[0][1]) <= y_tolerance:
                current_row.append(bubble)
            else:
                if len(current_row) >= 4:  # Only keep rows with at least 4 bubbles
                    question_rows.append(current_row)
                current_row = [bubble]

        # Add the last row
        if len(current_row) >= 4:
            question_rows.append(current_row)

        # Organize each row: sort by X coordinate and assign to A,B,C,D
        organized_questions = {}

        for row_index, row_bubbles in enumerate(question_rows):
            # Sort by X coordinate (left to right)
            sorted_by_x = sorted(row_bubbles, key=lambda b: b[0])

            question_number = row_index + 1

            # Take exactly 4 bubbles for A,B,C,D
            question_bubbles = {}
            options = ["A", "B", "C", "D"]

            for i, option in enumerate(options):
                if i < len(sorted_by_x):
                    question_bubbles[option] = sorted_by_x[i]

            # Only add if we have all 4 options
            if len(question_bubbles) == 4:
                organized_questions[question_number] = question_bubbles

        if self.debug_mode:
            print(f"Organized {len(organized_questions)} questions with improved grouping")

        return organized_questions

    def extract_answers_with_superior_detection(self, original_image: np.ndarray,
                                                bubbles: List[Tuple[int, int, int]]) -> Dict[str, str]:
        """Extract answers with superior fill detection"""
        if not bubbles:
            return {}

        # Get properly organized questions
        organized_questions = self.organize_bubbles_into_grid(bubbles)

        answers = {}
        detection_details = {}

        for question_num, question_bubbles in organized_questions.items():
            option_scores = {}

            # Check each option with confidence scoring
            for option in ["A", "B", "C", "D"]:
                if option in question_bubbles:
                    x, y, r = question_bubbles[option]
                    is_filled, confidence = self.advanced_fill_detection(original_image, x, y, r)
                    option_scores[option] = (is_filled, confidence)

            # Find the best filled option (highest confidence among filled)
            filled_options = [(opt, conf) for opt, (filled, conf) in option_scores.items() if filled]

            if filled_options:
                # Take the option with highest confidence
                best_option = max(filled_options, key=lambda x: x[1])
                answers[str(question_num)] = best_option[0]
                detection_details[question_num] = option_scores

                if self.debug_mode and question_num <= 25:  # Debug first 25
                    print(f"Q{question_num}: Selected {best_option[0]} (conf: {best_option[1]:.2f})")

        return answers

    def calculate_score(self, detected_answers: Dict[str, str], answer_key: Dict[str, str]) -> int:
        correct_count = 0

        if self.debug_mode:
            print(f"\n--- DETAILED SCORING ---")
            print(f"Detected answers: {len(detected_answers)}")
            print(f"Answer key questions: {len(answer_key)}")

        for question, correct_answer in answer_key.items():
            if question in detected_answers:
                detected = detected_answers[question]
                is_correct = detected == correct_answer
                if is_correct:
                    correct_count += 1

                if self.debug_mode and int(question) <= 30:  # Show more for debugging
                    status = "✓" if is_correct else "✗"
                    print(f"Q{question}: Got={detected}, Expected={correct_answer} {status}")

        if self.debug_mode:
            accuracy = (correct_count / len(answer_key)) * 100 if answer_key else 0
            print(f"Final Score: {correct_count}/{len(answer_key)} ({accuracy:.1f}%)")

        return correct_count


def evaluate_omr(image: np.ndarray, answer_key: Dict[str, str], template: Dict = None) -> Tuple[Dict[str, str], int]:
    """Superior OMR evaluation with advanced fill detection"""
    processor = SuperiorOMRProcessor(template)

    print("=== Starting SUPERIOR OMR Evaluation ===")

    # Preprocess image
    preprocessed = processor.preprocess_image(image)

    # Detect bubbles
    bubbles = processor.detect_bubbles(preprocessed)
    print(f"Total bubbles detected: {len(bubbles)}")

    if len(bubbles) == 0:
        print("ERROR: No bubbles detected!")
        return {}, 0

    # Extract answers with superior detection
    detected_answers = processor.extract_answers_with_superior_detection(image, bubbles)

    # Calculate score
    total_score = processor.calculate_score(detected_answers, answer_key)

    print(f"=== SUPERIOR RESULT: {total_score}/{len(answer_key)} ===")

    return detected_answers, total_score


def load_template(template_file: str = "template.json") -> Optional[Dict]:
    try:
        with open(template_file, 'r') as f:
            return json.load(f)
    except:
        return None