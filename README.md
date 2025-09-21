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
Screenshots
<img width="1906" height="882" alt="image" src="https://github.com/user-attachments/assets/a97e0bc6-b16d-4d01-af08-fbfc4049c257" />
<img width="1883" height="886" alt="image" src="https://github.com/user-attachments/assets/fdd335b6-8ccf-4c90-ad7e-332fd1264af3" />
<img width="1912" height="898" alt="image" src="https://github.com/user-attachments/assets/6f831298-75b6-46b9-9791-4f6d05a0699e" />

Author

Durgesh Sharma
