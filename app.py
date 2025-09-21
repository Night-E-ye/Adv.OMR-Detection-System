import streamlit as st
import json
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
import io
from omr_processor import evaluate_omr, load_template

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="OMR Sheet Evaluator",
    page_icon="📊",
    layout="wide"
)


# --------------------------
# Load answer key
# --------------------------
@st.cache_data
def load_answer_keys():
    try:
        with open("answer_key.json") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("answer_key.json not found!")
        return {"A": {}, "B": {}}


answer_keys = load_answer_keys()


# --------------------------
# Load template
# --------------------------
@st.cache_data
def load_template_cached():
    try:
        return load_template("template.json")
    except Exception as e:
        st.warning(f"Template loading issue: {e}")
        return None


template = load_template_cached()

# --------------------------
# Subject Configuration (20 questions per subject)
# --------------------------
SUBJECTS = {
    1: "Data Analytics",
    2: "Machine Learning",
    3: "Statistics",
    4: "Python Programming",
    5: "SQL & Databases"
}

QUESTIONS_PER_SUBJECT = 20


def calculate_subject_scores(detected_answers, correct_answers):
    """Calculate subject-wise scores (20 questions per subject)"""
    subject_scores = {}
    total_score = 0

    for subject_num in range(1, 6):  # 5 subjects
        start_q = (subject_num - 1) * QUESTIONS_PER_SUBJECT
        end_q = start_q + QUESTIONS_PER_SUBJECT

        subject_correct = 0
        for q_num in range(start_q, end_q):
            q_key = str(q_num + 1)  # Questions numbered from 1
            if q_key in detected_answers and q_key in correct_answers:
                if detected_answers[q_key] == correct_answers[q_key]:
                    subject_correct += 1

        subject_scores[SUBJECTS[subject_num]] = subject_correct
        total_score += subject_correct

    return subject_scores, total_score


# --------------------------
# Main App
# --------------------------
st.title("🎯 OMR Sheet Evaluator")
st.markdown("**Automated OMR Evaluation System for Innomatics Research Labs**")

# Sidebar for configuration
st.sidebar.header("Configuration")
sheet_set = st.sidebar.selectbox("Select Sheet Set", ["A", "B"])
selected_key = answer_keys.get(sheet_set, {})

if not selected_key:
    st.sidebar.error(f"No answer key found for Set {sheet_set}")

# Display answer key info
st.sidebar.info(f"Answer Key Set {sheet_set} loaded with {len(selected_key)} questions")

# --------------------------
# File Upload Section
# --------------------------
st.header("📤 Upload OMR Sheets")
uploaded_files = st.file_uploader(
    "Upload OMR Images (JPG, PNG formats supported)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="You can upload multiple OMR sheets at once"
)

# --------------------------
# Processing Section
# --------------------------
if uploaded_files:
    st.header("🔄 Processing Results")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    processed_images = {}

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")

            # Convert file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not decode image")

            # Evaluate OMR
            detected_answers, total_score = evaluate_omr(img, selected_key, template)

            # Calculate subject-wise scores
            subject_scores, calculated_total = calculate_subject_scores(detected_answers, selected_key)

            # Store results
            result = {
                "Student_ID": uploaded_file.name.split('.')[0],
                "Image_Name": uploaded_file.name,
                "Set": sheet_set,
                "Total_Score": calculated_total,
                **{f"{subject}_Score": score for subject, score in subject_scores.items()},
                "Percentage": round((calculated_total / 100) * 100, 2),
                "Processing_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            results.append(result)
            processed_images[uploaded_file.name] = img

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            results.append({
                "Student_ID": uploaded_file.name.split('.')[0],
                "Image_Name": uploaded_file.name,
                "Set": sheet_set,
                "Error": str(e),
                "Processing_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("Processing completed!")

    # --------------------------
    # Results Display
    # --------------------------
    if results:
        st.header("📊 Evaluation Results")

        # Create results DataFrame
        df_results = pd.DataFrame(results)

        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Sheets", len(results))
        with col2:
            successful = len([r for r in results if 'Error' not in r])
            st.metric("Successfully Processed", successful)
        with col3:
            if successful > 0:
                avg_score = df_results['Total_Score'].mean()
                st.metric("Average Score", f"{avg_score:.1f}/100")
        with col4:
            if successful > 0:
                max_score = df_results['Total_Score'].max()
                st.metric("Highest Score", f"{max_score}/100")

        # Display detailed results table
        st.subheader("Detailed Results")
        st.dataframe(df_results, use_container_width=True)

        # Subject-wise performance chart
        if successful > 0:
            st.subheader("📈 Subject-wise Performance")
            subject_cols = [col for col in df_results.columns if col.endswith('_Score') and 'Total' not in col]

            if subject_cols:
                subject_avg = df_results[subject_cols].mean()
                st.bar_chart(subject_avg)

        # --------------------------
        # Export Options
        # --------------------------
        st.header("💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # CSV Export
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📄 Download Results as CSV",
                data=csv_data,
                file_name=f"omr_results_{sheet_set}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # Excel Export
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_results.to_excel(writer, sheet_name='Results', index=False)

            st.download_button(
                label="📊 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"omr_results_{sheet_set}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # --------------------------
        # Image Review Section
        # --------------------------
        st.header("🔍 Image Review")
        if processed_images:
            selected_image = st.selectbox(
                "Select image to review:",
                options=list(processed_images.keys())
            )

            if selected_image:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        processed_images[selected_image],
                        channels="BGR",
                        caption=f"Processed: {selected_image}",
                        use_column_width=True
                    )
                with col2:
                    # Show results for this image
                    image_result = next((r for r in results if r['Image_Name'] == selected_image), None)
                    if image_result and 'Error' not in image_result:
                        st.subheader("Scores:")
                        st.write(f"**Total Score:** {image_result['Total_Score']}/100")
                        st.write(f"**Percentage:** {image_result['Percentage']}%")

                        st.subheader("Subject-wise Breakdown:")
                        for subject in SUBJECTS.values():
                            score_key = f"{subject}_Score"
                            if score_key in image_result:
                                st.write(f"**{subject}:** {image_result[score_key]}/20")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("**Automated OMR Evaluation System** | Built for Innomatics Research Labs Hackathon")
st.markdown("*Error Tolerance: <0.5% | Processing Time: Minutes instead of Days*")