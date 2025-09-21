# 🎯 OMR Sheet Evaluator

An automated OMR (Optical Mark Recognition) sheet evaluation system built with **Python** and **Streamlit**.

## Features
- Upload multiple OMR sheets (JPG/PNG) for automatic evaluation
- Supports answer key **Set A & Set B**
- Subject-wise scoring (Data Analytics, Machine Learning, Statistics, Python, SQL)
- Export results to **CSV/Excel**
- Debug tool for mapping verification
- Parameter tuning tool for detection accuracy

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/Night-E-Eye/Adv_OMR-Detection-System.git
   cd omr-evaluator
2. Install dependencies:
   ```bash

   pip install -r requirements.txt

3. Run the main app:
   ```bash  

   streamlit run app.py

4. Debug or tune parameters:
    ```bash
   streamlit run debug_app.py
   streamlit run parameter_tuning.py

Answer Keys

answer_key.json → Used by app
  
    Key (Set A and B).xlsx → Reference version

Author

Durgesh Sharma
