import streamlit as st
import json
import pandas as pd
import cv2
import numpy as np
from omr_processor import evaluate_omr, FixedOMRProcessor

# Page config
st.set_page_config(page_title="Bubble Mapping Debug", layout="wide")


# Load answer key
@st.cache_data
def load_answer_keys():
    try:
        with open("answer_key.json") as f:
            return json.load(f)
    except:
        # Sample key for testing
        sample_key = {}
        for i in range(1, 101):
            sample_key[str(i)] = ["A", "B", "C", "D"][i % 4]
        return {"A": sample_key, "B": sample_key}


def draw_bubble_mapping_debug(image, organized_questions, detected_answers):
    """Draw bubbles with their A,B,C,D labels and fill status"""
    debug_image = image.copy()

    colors = {
        "A": (255, 0, 0),  # Blue
        "B": (0, 255, 0),  # Green
        "C": (0, 0, 255),  # Red
        "D": (255, 0, 255)  # Magenta
    }

    for question_num, question_bubbles in organized_questions.items():
        for option, (x, y, r) in question_bubbles.items():
            color = colors[option]

            # Check if this bubble is detected as filled
            is_selected = (str(question_num) in detected_answers and
                           detected_answers[str(question_num)] == option)

            # Draw circle - thicker if selected
            thickness = 4 if is_selected else 2
            cv2.circle(debug_image, (x, y), r, color, thickness)

            # Add label
            label = f"{option}"
            if is_selected:
                label += "✓"

            cv2.putText(debug_image, label, (x - 15, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Add question number for first few questions
            if question_num <= 5:
                cv2.putText(debug_image, f"Q{question_num}", (x + r + 5, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return debug_image


# Main app
st.title("🎯 Bubble Mapping Debug Tool")
st.markdown("**Check if A, B, C, D are being mapped correctly**")

# Load answer keys
answer_keys = load_answer_keys()
sheet_set = st.sidebar.selectbox("Sheet Set", ["A", "B"])
selected_key = answer_keys[sheet_set]

# Upload image
uploaded_file = st.file_uploader("Upload OMR Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to opencv
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        st.header("Original Image")
        st.image(image, channels="BGR", width=400)

        # Process with fixed processor
        processor = FixedOMRProcessor()

        # Get preprocessing result
        preprocessed = processor.preprocess_image(image)

        # Detect bubbles
        bubbles = processor.detect_bubbles(preprocessed)
        st.write(f"**Total bubbles detected:** {len(bubbles)}")

        if bubbles:
            # Organize bubbles into grid
            organized_questions = processor.organize_bubbles_into_grid(bubbles)
            st.write(f"**Questions organized:** {len(organized_questions)}")

            # Extract answers
            detected_answers = processor.extract_answers_with_fixed_mapping(image, bubbles)
            st.write(f"**Answers detected:** {len(detected_answers)}")

            # Show mapping visualization
            if organized_questions:
                debug_img = draw_bubble_mapping_debug(image, organized_questions, detected_answers)

                st.header("🔍 Bubble Mapping Visualization")
                st.image(debug_img, channels="BGR",
                         caption="Colors: A=Blue, B=Green, C=Red, D=Magenta. Thick border = Selected", width=800)

                # Show detailed mapping for first few questions
                st.header("📋 Detailed Question Mapping")

                mapping_data = []
                for q_num in sorted(organized_questions.keys())[:10]:  # Show first 10
                    q_bubbles = organized_questions[q_num]
                    detected = detected_answers.get(str(q_num), "None")
                    correct = selected_key.get(str(q_num), "?")

                    mapping_data.append({
                        "Question": q_num,
                        "A_pos": f"({q_bubbles['A'][0]}, {q_bubbles['A'][1]})",
                        "B_pos": f"({q_bubbles['B'][0]}, {q_bubbles['B'][1]})",
                        "C_pos": f"({q_bubbles['C'][0]}, {q_bubbles['C'][1]})",
                        "D_pos": f"({q_bubbles['D'][0]}, {q_bubbles['D'][1]})",
                        "Detected": detected,
                        "Correct": correct,
                        "Match": "✅" if detected == correct else "❌"
                    })

                st.dataframe(pd.DataFrame(mapping_data), hide_index=True)

                # Calculate and show score
                score = processor.calculate_score(detected_answers, selected_key)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{score}/100")
                with col2:
                    percentage = (score / 100) * 100
                    st.metric("Percentage", f"{percentage:.1f}%")
                with col3:
                    st.metric("Detected Answers", len(detected_answers))

                # Show sample comparisons
                st.header("🔍 Answer Comparison (First 20 Questions)")
                comparison_data = []
                for i in range(1, min(21, len(selected_key) + 1)):
                    q_str = str(i)
                    detected = detected_answers.get(q_str, "None")
                    correct = selected_key.get(q_str, "?")

                    comparison_data.append({
                        "Q": i,
                        "Detected": detected,
                        "Correct": correct,
                        "Status": "✅" if detected == correct else "❌"
                    })

                df_comparison = pd.DataFrame(comparison_data)


                # Color code the dataframe
                def highlight_rows(row):
                    if row['Status'] == '✅':
                        return ['background-color: lightgreen'] * len(row)
                    elif row['Status'] == '❌':
                        return ['background-color: lightcoral'] * len(row)
                    else:
                        return [''] * len(row)


                styled_df = df_comparison.style.apply(highlight_rows, axis=1)
                st.dataframe(styled_df, hide_index=True)

            else:
                st.error("Could not organize bubbles into questions. Check bubble detection parameters.")

        else:
            st.error("No bubbles detected! Check preprocessing parameters.")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("**Debug Guide:**")
st.sidebar.markdown("1. Upload your problematic image")
st.sidebar.markdown("2. Check bubble mapping visualization")
st.sidebar.markdown("3. Verify A,B,C,D positions are correct")
st.sidebar.markdown("4. Compare detected vs correct answers")
st.sidebar.markdown("5. Look for systematic mapping errors")