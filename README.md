# RainS2S: đánh giá có kiểm soát độ lệch tần suất cho dự báo mưa ngắn hạn từ GPM IMERG

*Frequency-bias-aware evaluation of deep spatiotemporal rainfall nowcasting from GPM IMERG over Vietnam.*

Mã và bảng kết quả đi kèm bài báo: persistence, dòng quang học, CNN–RNN dạng vector, ConvLSTM, ConvLSTM–U-Net, **RainS2S**, SimVPv2, PredRNN++. RainS2S là tên đặt cho ConvLSTM–U-Net theo kiến trúc Seq2Seq (dự báo thẳng +30/+60/+90 phút). Điểm chính: so sánh với persistence ở ngưỡng cố định 0,5 mm/h và sau khi hiệu chuẩn ngưỡng trên tập kiểm định để độ lệch tần suất (bias) xấp xỉ 1.

## Dữ liệu
GPM IMERG Final V07 (biến `precipitation`), 30 phút, 0,1°, vùng 8–23°N, 102–110°E (lưới 150 × 80), 01/01/2022 00:00 → 04/08/2023 14:00 UTC, 27.869 khung, không có khoảng trống. Tải từ NASA GES DISC (cần tài khoản Earthdata). Chia theo thời gian 70/15/15 (huấn luyện 19.508, kiểm định 4.180, kiểm tra 4.181 khung); chuẩn hóa log(1+x)/4,6846 (hằng số tính trên tập huấn luyện). Kho dữ liệu đã xử lý (chuỗi mưa cắt theo vùng và dự báo của mô hình) sẽ được bổ sung sau.

## Cách tái lập (Colab; mỗi bước tự resume; dùng chung thư mục Drive `rainfall_p0`)
1. `notebooks/00_preprocess_gpm.ipynb`: đọc các file `.nc4`, cắt vùng, tạo `rain_raw.npy` (mm/h, kích thước T × 150 × 80). CPU là đủ.
2. `notebooks/01_benchmark_train_eval.ipynb`: huấn luyện 10 cấu hình × 3 seed, đánh giá ngưỡng cố định và hiệu chuẩn bias, bootstrap theo khối ngày (GPU; RainS2S, SimVPv2, PredRNN++ chạy lâu nhất).
3. `notebooks/02_bias_matched_multithreshold.ipynb`: hiệu chuẩn bias ở 1/2/5 mm/h từ checkpoint của bước 2.
4. `notebooks/03_bce_threshold_ablation.ipynb`: RainS2S với ngưỡng nhãn BCE 0,5 và 1 mm/h (bản `..._with_outputs` kèm kết quả đã chạy).

Cài thư viện: `pip install -r requirements.txt` (Colab đã có sẵn hầu hết).

## Kết quả (`results/`)
| File | Bảng trong bài |
|---|---|
| `fixed_threshold_0p5mmh.csv` | 1, 2 |
| `bias_matched_0p5mmh.csv` | 3, 5 |
| `bootstrap_delta_vs_persistence.csv` | 4 (ΔCSI, khoảng tin cậy 95%, bootstrap theo khối ngày) |
| `fixed_multithreshold.csv` | 6 |
| `bias_matched_multithreshold.csv` | 7 |
| `bce_threshold_ablation.csv` | 8 |
| `bestcsi_on_val_0p5mmh.csv` | bổ sung (ngưỡng chọn theo CSI tốt nhất trên kiểm định) |

Trung bình ± độ lệch chuẩn qua 3 seed. Độ lệch chuẩn trong bài tính theo n−1; cột `*_sd` của `bias_matched_multithreshold.csv` và `bce_threshold_ablation.csv` do notebook tính theo n. Tên file notebook tạo ra (`T1…T7`) khác tên ở đây: T1→fixed_threshold_0p5mmh, T2→bias_matched_0p5mmh, T3→bestcsi_on_val_0p5mmh, T4→fixed_multithreshold, T5→bootstrap_delta_vs_persistence, T6→bias_matched_multithreshold, T7→bce_threshold_ablation.

## Hình (`figures_code/`)
`plot_paper_figures.py` (Hình 3–5) và `plot_qualitative_example.py` (Hình 6). Cần `rain_raw.npy` và dự báo seed 0 (`preds/seq2seq_unet_s0_test.npy`, tạo bởi notebook 01); sửa các đường dẫn đầu file nếu cần.

## Lưu ý
- Ngưỡng nhãn BCE mặc định là 0,05 (chuẩn hóa) ≈ 0,26 mm/h; thực nghiệm ngưỡng 0,5 và 1 mm/h ở notebook 03.
- Kết quả huấn luyện chính chạy trên GPU A100; notebook 03 chạy trên GPU L4.
- Checkpoint và dự báo (`.pt`, `.npy`) không đưa vào kho vì kích thước.
- Tập kiểm tra chỉ gồm một mùa mưa (09/05–04/08/2023); xem phần Hạn chế trong bài.

## Giấy phép và trích dẫn
Mã nguồn theo giấy phép MIT (`LICENSE`). Khi dùng mã hoặc kết quả, vui lòng trích dẫn bài báo (xem `CITATION.cff`). Dữ liệu IMERG thuộc NASA và tuân theo điều khoản của GES DISC.

## Calibration thresholds and calibrated persistence
- `results/calibration_thresholds_tau.csv`: validation-selected thresholds tau (mm/h, mean over 3 seeds) for 1, 2 and 5 mm/h.
- `results/persistence_calibrated.csv`: persistence calibrated with the same procedure; tau equals the evaluation threshold, so CSI is unchanged.
- `figures_code/persistence_calibration.py`: script that produced the persistence table (needs the processed rainfall arrays).

## Test-matched bias diagnostic
`results/test_matched_bias_diagnostic.csv` and `figures_code/test_matched_diagnostic.py`: RainS2S (seed 0) recalibrated on the test set so that bias = 1.000. CSI still exceeds persistence by +0.016, +0.079, +0.086 at +30, +60, +90 min. Uses test labels, so diagnostic only.
