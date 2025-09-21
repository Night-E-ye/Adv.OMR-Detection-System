import streamlit as st
import json
import pandas as pd
import cv2
import numpy as np
from omr_processor import evaluate_omr, SuperiorOMRProcessor

st.set_page_config(page_title="Parameter Tuning Tool", layout="wide")


# Load answer key
@st.cache_data
def load_answer_keys():
    try:
        with open("answer_key.json") as f:
            return json.load(f)
    except:
        sample_key = {}
        for i in range(1, 101):
            sample_key[str(i)] = ["A", "B", "C", "D"][i % 4]
        return {"A": sample_key, "B": sample_key}


def create_fill_analysis_image(image, organized_questions, processor):
    """Create detailed fill analysis visualization"""
    debug_image = image.copy()

    colors = {"A": (255, 0, 0), "B": (0, 255, 0), "C": (0, 0, 255), "D": (255, 0, 255)}

    fill_details = []

    for question_num, question_bubbles in organized_questions.items():
        if question_num > 30:  # Limit to first 30 for performance
            break

        for option, (x, y, r) in question_bubbles.items():
            is_filled, confidence = processor.advanced_fill_detection(image, x, y, r)

            color = colors[option]

            # Draw circle with thickness based on fill confidence
            thickness = max(1, int(confidence * 8)) if is_filled else 1
            cv2.circle(debug_image, (x, y), r, color, thickness)

            # Add confidence score
            if is_filled:
                cv2.putText(debug_image, f"{confidence:.2f}", (x - 15, y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Store details
            fill_details.append({
                "Question": question_num,
                "Option": option,
                "X": x, "Y": y, "Radius": r,
                "Filled": is_filled,
                "Confidence": confidence
            })

    return debug_image, fill_details


# Main app
st.title("🎛️ Parameter Tuning Tool")
st.markdown("**Fine-tune fill detection parameters for perfect accuracy**")

answer_keys = load_answer_keys()

# Parameter controls
st.sidebar.header("Fill Detection Parameters")

intensity_threshold = st.sidebar.slider("Intensity Threshold", 100, 200, 140,
                                        help="Lower = more sensitive to dark pixels")

fill_ratio_threshold = st.sidebar.slider("Fill Ratio Threshold", 0.1, 0.8, 0.35, 0.05,
                                         help="Minimum % of dark pixels to consider filled")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                                         help="Minimum confidence to consider a bubble filled")

# Bubble detection parameters
st.sidebar.subheader("Bubble Detection")
min_area = st.sidebar.slider("Min Bubble Area", 20, 100, 40)
max_area = st.sidebar.slider("Max Bubble Area", 500, 1500, 1000)
circularity_min = st.sidebar.slider("Min Circularity", 0.1, 0.5, 0.2, 0.1)
circularity_max = st.sidebar.slider("Max Circularity", 1.0, 2.0, 1.5, 0.1)

sheet_set = st.sidebar.selectbox("Sheet Set", ["A", "B"])
selected_key = answer_keys[sheet_set]

# Upload image
uploaded_file = st.file_uploader("Upload OMR Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to opencv
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        # Create custom template with tuned parameters
        custom_template = {
            "bubble_params": {
                "min_radius": 6, "max_radius": 20, "dp": 1,
                "param1": 30, "param2": 15, "min_distance": 25
            },
            "preprocessing": {
                "gaussian_blur": (3, 3),
                "threshold_value": 127,
                "morph_kernel": (2, 2)
            },
            "fill_detection": {
                "method": "multi_criteria",
                "intensity_threshold": intensity_threshold,
                "fill_ratio_threshold": fill_ratio_threshold,
                "edge_penalty": True,
                "adaptive_threshold": True
            },
            "bubble_detection": {
                "min_area": min_area,
                "max_area": max_area,
                "circularity_min": circularity_min,
                "circularity_max": circularity_max
            }
        }

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, channels="BGR", width=400)

        # Process with custom parameters
        processor = SuperiorOMRProcessor(custom_template)

        # Manual override for bubble detection parameters
        processor.detect_bubbles_contour_method = lambda binary_image: [
            (int(x), int(y), int(radius))
            for contour in cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for area in [cv2.contourArea(contour)]
            if min_area < area < max_area
            for perimeter in [cv2.arcLength(contour, True)]
            if perimeter > 0
            for circularity in [4 * np.pi * area / (perimeter * perimeter)]
            if circularity_min < circularity < circularity_max
            for (x, y), radius in [cv2.minEnclosingCircle(contour)]
        ]

        try:
            with st.spinner("Processing with custom parameters..."):
                # Get step-by-step results
                preprocessed = processor.preprocess_image(image)
                bubbles = processor.detect_bubbles(preprocessed)
                organized_questions = processor.organize_bubbles_into_grid(bubbles)
                detected_answers = processor.extract_answers_with_superior_detection(image, bubbles)
                score = processor.calculate_score(detected_answers, selected_key)

            # Results summary
            st.header("📊 Results Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bubbles Detected", len(bubbles))
            with col2:
                st.metric("Questions Organized", len(organized_questions))
            with col3:
                st.metric("Answers Detected", len(detected_answers))
            with col4:
                st.metric("Score", f"{score}/100")

            # Detailed fill analysis
            if organized_questions:
                st.header("🔍 Fill Analysis Visualization")

                analysis_image, fill_details = create_fill_analysis_image(image, organized_questions, processor)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Fill Confidence Visualization")
                    st.image(analysis_image, channels="BGR",
                             caption="Thickness = Confidence, Numbers = Confidence scores",
                             width=600)

                with col2:
                    st.subheader("Parameter Impact")

                    # Show impact of current parameters
                    filled_bubbles = [d for d in fill_details if d['Filled']]
                    avg_confidence = np.mean([d['Confidence'] for d in filled_bubbles]) if filled_bubbles else 0

                    st.write(f"**Filled bubbles detected:** {len(filled_bubbles)}")
                    st.write(f"**Average confidence:** {avg_confidence:.3f}")
                    st.write(f"**Total questions with answers:** {len(detected_answers)}")

                    # Show confidence distribution
                    if filled_bubbles:
                        confidence_scores = [d['Confidence'] for d in filled_bubbles]
                        st.write("**Confidence Distribution:**")
                        confidence_df = pd.DataFrame({
                            'Confidence Range': ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'],
                            'Count': [
                                sum(1 for c in confidence_scores if 0.5 <= c < 0.6),
                                sum(1 for c in confidence_scores if 0.6 <= c < 0.7),
                                sum(1 for c in confidence_scores if 0.7 <= c < 0.8),
                                sum(1 for c in confidence_scores if 0.8 <= c < 0.9),
                                sum(1 for c in confidence_scores if 0.9 <= c <= 1.0),
                            ]
                        })
                        st.dataframe(confidence_df, hide_index=True)

                # Detailed results for specific questions
                st.header("🎯 Question-by-Question Analysis")

                # Focus on problematic questions (21, 22, etc.)
                focus_questions = [21, 22, 23, 24, 25]
                focus_data = []

                for q in focus_questions:
                    if q in organized_questions:
                        q_bubbles = organized_questions[q]
                        detected = detected_answers.get(str(q), "None")
                        correct = selected_key.get(str(q), "?")

                        # Get fill details for this question
                        q_fill_details = [d for d in fill_details if d['Question'] == q]

                        fill_info = {}
                        for detail in q_fill_details:
                            fill_info[
                                detail['Option']] = f"{'✓' if detail['Filled'] else '✗'} ({detail['Confidence']:.2f})"

                        focus_data.append({
                            "Question": q,
                            "A": fill_info.get('A', 'N/A'),
                            "B": fill_info.get('B', 'N/A'),
                            "C": fill_info.get('C', 'N/A'),
                            "D": fill_info.get('D', 'N/A'),
                            "Detected": detected,
                            "Correct": correct,
                            "Match": "✅" if detected == correct else "❌"
                        })

                if focus_data:
                    st.dataframe(pd.DataFrame(focus_data), hide_index=True)

                    st.markdown("**Legend:** ✓ = Filled, ✗ = Not filled, (number) = Confidence score")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            import traceback

            st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.markdown("**Tuning Tips:**")
st.sidebar.markdown("• Lower intensity threshold for darker fills")
st.sidebar.markdown("• Adjust fill ratio for partial fills")
st.sidebar.markdown("• Increase confidence threshold to reduce false positives")
st.sidebar.markdown("• Monitor question 21-25 for improvements")