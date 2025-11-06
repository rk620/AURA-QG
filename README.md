# AURA-QG: Automated Unsupervised Replicable Assessment for Question Generation

## Paper and Conference Information

This research work has been accepted in the [International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP–AACL 2025)](https://2025.aaclnet.org/) and is presented at the conference scheduled from **20 December 2025 to 24 December 2025** in **Mumbai, India**.

This repository provides the codebase corresponding to our paper:

> **Title:** AURA-QG: Automated Unsupervised Replicable Assessment for Question Generation  
> **Authors:** Rajshekar K, Harshad Khadilkar, Pushpak P Bhattacharyya  
> **Conference:** International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP–AACL 2025) Main Conference
>  
> The repository includes a fully replicable implementation of our deterministic, LLM-free, and reference-free evaluation pipeline for assessing question answerability and coverage.

---

## 🔖 License

This project is licensed under the **Apache License 2.0** (recommended for research reproducibility).  
You are free to use, modify, and distribute this work, provided that:
- Proper credit is given to the authors and the original publication.
- Any derivative work or publication citing this code references the paper:  
  > Rajshekar, K., Harshad Khadilkar, and Pushpak P. Bhattacharyya. **“AURA-QG: Automated Unsupervised Replicable Assessment for Question Generation.”** In Proceedings of the International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL), 2025

For citation convenience, please use the BibTeX entry below:
@inproceedings{rajashekar2025auraqg,
  title     = {AURA-QG: Automated Unsupervised Replicable Assessment for Question Generation},
  author    = {Rajshekar, K. and Khadilkar, Harshad and Bhattacharyya, Pushpak P.},
  booktitle = {Proceedings of the International Joint Conference on Natural Language Processing and the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL 2025)},
  year      = {2025},
  address   = {Mumbai, India},
  publisher = {Association for Computational Linguistics},
}

---

## ⚠️ Disclaimer

- This repository contains only the **code implementation** and does **not include** the camera-ready or accepted version of the paper.  
- The code is released for **academic and research use only**.  

---





**Abstract**  
We present a reference-free, interpretable evaluation pipeline for assessing question sets generated from document-level text. Our method scores question sets along four key dimensions — Answerability, Redundancy, Coverage, and Structural Entropy — and demonstrates strong agreement with human preferences. This repository provides the complete implementation to reproduce our automatic evaluation pipeline.

## Setup Instructions

1. **Create a new virtual environment** (recommended):
   ```bash
      conda create -n aura-qg
      conda activate aura-qg
   ```

2. **Install required packages**:
   Navigate to the code_submission directory and run the following command:
   ```bash
      pip install --no-cache-dir -r requirements.txt
   ```

## Run the Evaluation

After setup, execute the main script:
```bash
   python driver_code.py
```

This will:
- Read input from `sample_input.json`
- Evaluate two question sets grounded in a document (provided in markdown format)
- Compute all four metric scores for each set (Answerability, Coverage, Non-Redundancy, Structural Entropy)
- Generate a **spider web chart** to visualize metric-wise differences

## Input Format
If you wanna provide a customized input:

         Input must be provided as a `.json` file with the following fields:
         ```json
         {
         "passage": "Markdown-formatted document text...",
         "questions1": ["Q1", "Q2", "..."],
         "questions2": ["Q1", "Q2", "..."]
         }
         ```

         **Important** : The input passage must be in markdown format, as the pipeline's extraction and indexing backbone relies on markdown structure.

         To use your own input, simply follow the format in `sample_input.json`.

## Repository Structure

```
aura-qg/
├── driver_code.py         # Main entry point
├── modules.py             # Metric and utility functions
├── sample_input.json      # Demo input with markdown-formatted document + two question sets
├── requirements.txt       # All required Python packages
├── spider_plot.png        # The spider plot output for current input file
└── readme.md              # This file
```

## Example Output

The plot for the sample_input.json comparing both question sets across the four evaluation metrics is saved as `spider_plot.png`.
This visualization provides a clear comparison of the relative performance of two question sets across Answerability, Coverage, Non-Redundancy, and Structural Entropy dimensions.

## 🧠 Acknowledgements

This work was carried out under the guidance of **Prof. Pushpak Bhattacharyya**, whose vision and mentorship shaped this research.
Heartfelt gratitude to **Prof. Harshad Khadilkar** for his structured technical guidance and invaluable support throughout the work.


## 📫 Contact

For questions, issues, or collaboration requests, please reach out at:
Rajshekar K – [rajshekark@cse.iitb.ac.in] | [www.linkedin.com/in/rajshekar-iit-bombay]
