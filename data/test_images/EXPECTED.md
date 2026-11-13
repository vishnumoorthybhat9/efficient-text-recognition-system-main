# Test images

29 images for trying the system out. Every one has known text,
so you can judge the output rather than guess at it.

Try them with:

```bash
.venv/Scripts/python.exe scripts/extract_text.py data/test_images/03_page_clean_01.png --model trocr_base
```

or drag one onto the **Recognise** page at http://localhost:5173

| File | Note | Expected text |
|---|---|---|
| `01_line_printed_01.png` | Single printed line. | Efficient Text Recognition |
| `01_line_printed_02.png` | Single printed line. | Deep Learning |
| `02_line_script_01.png` | Single line in a handwriting typeface. | Self Supervised Learning |
| `01_line_printed_03.png` | Single printed line. | Invoice Total 1,250.00 |
| `01_line_printed_04.png` | Single printed line. | INV-44718 |
| `02_line_script_02.png` | Single line in a handwriting typeface. | handwritten sample text |
| `02_line_script_03.png` | Single line in a handwriting typeface. | 22 Mar 2026 |
| `01_line_printed_05.png` | Single printed line. | Department of ISE |
| `02_line_handwritten_01.png` | Real handwriting (Kaggle). See the caveats below the table. | ENZO |
| `02_line_handwritten_02.png` | Real handwriting (Kaggle). See the caveats below the table. | LEANNE |
| `02_line_handwritten_03.png` | Real handwriting (Kaggle). See the caveats below the table. | MEGUEDDEM |
| `02_line_handwritten_04.png` | Real handwriting (Kaggle). See the caveats below the table. | EYRAUD |
| `02_line_handwritten_05.png` | Real handwriting (Kaggle). See the caveats below the table. | SAN |
| `02_line_handwritten_06.png` | Real handwriting (Kaggle). See the caveats below the table. | LINA |
| `02_line_handwritten_07.png` | Real handwriting (Kaggle). See the caveats below the table. | LEANE |
| `02_line_handwritten_08.png` | Real handwriting (Kaggle). See the caveats below the table. | YOLLODO |
| `03_page_clean_01.png` | Cleanly rendered page. | INVOICE ⏎ Acme Corporation Limited ⏎ 42 Industrial Estate, Mangalore 575001 ⏎ Invoice No: INV-44718 ⏎ Date:... |
| `04_page_scan_01.png` | Lightly skewed and noisy, like a flatbed scan. | INVOICE ⏎ Acme Corporation Limited ⏎ 42 Industrial Estate, Mangalore 575001 ⏎ Invoice No: INV-44718 ⏎ Date:... |
| `05_page_photo_01.png` | Rotated with uneven lighting, like a phone photo. | INVOICE ⏎ Acme Corporation Limited ⏎ 42 Industrial Estate, Mangalore 575001 ⏎ Invoice No: INV-44718 ⏎ Date:... |
| `03_page_clean_02.png` | Cleanly rendered page. | Department of Information Science and Engineering ⏎ Major Project Synopsis ⏎ Efficient Text Recognition fro... |
| `04_page_scan_02.png` | Lightly skewed and noisy, like a flatbed scan. | Department of Information Science and Engineering ⏎ Major Project Synopsis ⏎ Efficient Text Recognition fro... |
| `05_page_photo_02.png` | Rotated with uneven lighting, like a phone photo. | Department of Information Science and Engineering ⏎ Major Project Synopsis ⏎ Efficient Text Recognition fro... |
| `03_page_clean_03.png` | Cleanly rendered page. | Text recognition from images is a fundamental problem ⏎ in computer vision and pattern recognition, playing... |
| `04_page_scan_03.png` | Lightly skewed and noisy, like a flatbed scan. | Text recognition from images is a fundamental problem ⏎ in computer vision and pattern recognition, playing... |
| `05_page_photo_03.png` | Rotated with uneven lighting, like a phone photo. | Text recognition from images is a fundamental problem ⏎ in computer vision and pattern recognition, playing... |
| `06_hard_low_contrast.png` | Very faint. Try Enhance contrast (CLAHE). | INVOICE ⏎ Acme Corporation Limited ⏎ 42 Industrial Estate, Mangalore 575001 ⏎ Invoice No: INV-44718 ⏎ Date:... |
| `06_hard_skewed.png` | Rotated 8 degrees. Deskew must be on. | INVOICE ⏎ Acme Corporation Limited ⏎ 42 Industrial Estate, Mangalore 575001 ⏎ Invoice No: INV-44718 ⏎ Date:... |
| `06_hard_inverted.png` | White text on black. Polarity is normalised automatically. | INVOICE ⏎ Acme Corporation Limited ⏎ 42 Industrial Estate, Mangalore 575001 ⏎ Invoice No: INV-44718 ⏎ Date:... |
| `06_hard_blurred.png` | Out of focus and noisy. Expect errors here. | INVOICE ⏎ Acme Corporation Limited ⏎ 42 Industrial Estate, Mangalore 575001 ⏎ Invoice No: INV-44718 ⏎ Date:... |

## Reading the results

- `01_*` and `02_line_script_*` are single lines and should be near perfect.
- `02_line_handwritten_*` are real photographed form fields, and they come
  with two annotation quirks that make a *correct* reading look wrong:
    - The crop often includes a printed label such as `NOM :`, which is **not**
      in the expected text. A model that transcribes it is not misreading;
      it has no way to know the convention.
    - The annotators dropped accents. `02_line_handwritten_02.png` clearly
      reads **LÉANNE** on the page but its label is `LEANNE`. Reading the
      accent is right; the label is the simplification.
- `03_*` to `05_*` are the same pages captured three ways, so you can see
  how much of any error is the capture rather than the model.
- `06_hard_*` are meant to fail in specific ways; each row says which
  control to reach for.

`trocr_base` is the most accurate model on these. `crnn_full` is much
faster. Compare them on the **Compare models** page.