# gt_extract.py — ground truth từ MRZ và QR

Bước 1 của quy trình gán nhãn mờ GTTT: lấy "đáp án" của thẻ mà không dùng OCR của chính trường cần kiểm tra.

- **MRZ** (mặt sau): đọc từ `mrz1/mrz2/mrz3` trong JSON OCR, kiểm tra 4 check digit (ICAO 9303 TD1), bóc tách số ID, họ tên, ngày sinh, giới tính, ngày hết hạn, quốc tịch.
- **QR** (`CCCD_FRONT_CHIP` và `CC_BACK_CHIP`): giải mã từ ảnh gốc, tách các trường.
- `CC_BACK_CHIP` có cả QR và MRZ trên một ảnh nên script so luôn id, ngày sinh, giới tính. CCCD thì QR ở mặt trước, MRZ ở mặt sau: cần ghép theo phiên sau.

## Cài đặt

Chỉ chạy MRZ thì không cần cài gì. Quét QR cần:

```
pip install opencv-python-headless zxing-cpp tqdm
```

`tqdm` (thanh tiến độ) và `zxing-cpp` tùy chọn (không có `zxing-cpp` thì dùng bộ đọc QR của OpenCV). `pyzbar` là phương án thứ ba, cần `apt install libzbar0`.

## Cấu trúc thư mục

Ảnh và JSON ghép với nhau theo đường dẫn tương đối, một jpg ứng một json:

```
external/idcard-quality/26-09-01/abc.jpg
ocr_results/idcard-quality/26-09-01/abc.jpg.json
```

## Lệnh chạy

Chạy thử nhanh (mỗi loại `document` lấy 3 file, hoặc `--test 2` lấy 2 file):

```
python tools/gt_extract.py \
    --ocr-root /home/ducnq7/workspace/card_quality_ekyc/id_quality/ocr_results/idcard-quality \
    --img-root /home/ducnq7/workspace/card_quality_ekyc/id_quality/external/idcard-quality \
    --csv gt_test.csv --test
```

Chạy toàn bộ:

```
python tools/gt_extract.py \
    --ocr-root /home/ducnq7/workspace/card_quality_ekyc/id_quality/ocr_results/idcard-quality \
    --img-root /home/ducnq7/workspace/card_quality_ekyc/id_quality/external/idcard-quality \
    --csv gt.csv
```

Chỉ MRZ, không quét QR (bỏ `--img-root`):

```
python tools/gt_extract.py --ocr-root ocr_output --csv gt_mrz.csv
```

## Output

Terminal in tóm tắt theo từng loại `document`:

```
3 json -> gt.csv
  CCCD_FRONT_CHIP    n=1
  CC_BACK_CHIP       n=1  MRZ valid 1/1
  CC_FRONT_CHIP      n=1
```

File CSV có một dòng cho mỗi JSON, ô không áp dụng để trống:

| Nhóm cột | Cột |
|---|---|
| Nhận diện | `ocr_json`, `image`, `document` |
| MRZ | `mrz_valid` (1/0), `mrz_error`, `mrz_id`, `mrz_name`, `mrz_birthday`, `mrz_sex`, `mrz_expiry`, `mrz_nationality` |
| QR | `qr_ok`, `qr_error`, `qr_id`, `qr_cmnd_id`, `qr_name`, `qr_birthday`, `qr_sex`, `qr_address`, `qr_issue_date` |
| So sánh | `qr_vs_mrz`: `match` hoặc `mismatch:id,sex,...` (chỉ `CC_BACK_CHIP`) |

Giá trị hay gặp của `mrz_error`: `length a/b/c`, `check digit: birth,composite`, `id12 ... does not match doc number`.
Của `qr_error`: `no QR decoded`, `N fields`, hoặc lỗi đọc ảnh.

## Giả định cần kiểm tra trên thẻ thật

Đánh dấu `ASSUMPTION` trong code:

1. Chuỗi QR có dạng `id|cmnd_id|name|ddmmyyyy|sex|address|ddmmyyyy(issue)`.
2. Số ID 12 số nằm trong phần tùy chọn của dòng 1 MRZ, và 9 số cuối là số giấy tờ. Mẫu Căn cước đã khớp, CCCD cũ có MRZ chưa kiểm tra.

Nên chạy `--test` trước, xem `QR ok` và `qr_error` rồi mới chạy toàn bộ.

---

# gt_pair.py — gắn ground truth cho ảnh mặt trước

Ghép mặt trước với "sổ đáp án" lấy từ các mặt sau trong `gt.csv`, theo danh tính (loại thẻ + số ID, kèm ngày sinh / tên khi ID bị OCR sai), không cần mã phiên.

```
python tools/gt_pair.py --gt gt.csv \
    --ocr-root /home/ducnq7/workspace/card_quality_ekyc/id_quality/ocr_results/idcard-quality \
    --csv pairs.csv
```

Cột `match`:

| Giá trị | Ý nghĩa | Xử lý |
|---|---|---|
| `exact_id` | ID mặt trước trùng ID trong sổ | chấm tự động |
| `fuzzy` | ID lệch ≤ 2 ký tự và trùng ngày sinh hoặc tên, chỉ có một ứng viên | chấm tự động |
| `ambiguous` | có nhiều ứng viên | không có ground truth |
| `no_back` | không có ứng viên; thực tế gần như luôn là thẻ không có ảnh mặt sau trong tập, không phải dấu hiệu mờ | không có ground truth |

Các cột còn lại: `gt_group` (A/B/C, để trống nếu không ghép được), các cột `gt_*` (đáp án), `ocr_*` (giá trị OCR của chính ảnh mặt trước), `back_ocr_json` và `n_backs`.

Cột `id_check` (cho mọi ảnh mặt trước, không cần mặt sau): số ID 12 số có chữ số thứ 4 mã hóa giới tính và thế kỷ sinh, chữ số 5–6 là hai số cuối năm sinh, nên phải khớp với ngày sinh và giới tính OCR trên cùng ảnh.

| Giá trị | Ý nghĩa |
|---|---|
| `ok` | nhất quán (bằng chứng khá mạnh là cả ba trường đúng, chưa phải ground truth) |
| `mismatch:year,sex` | ít nhất một trong ID / ngày sinh / giới tính bị OCR sai |
| `bad_id` | ID rỗng hoặc không đủ 12 số |
| `incomplete` | thiếu cả ngày sinh lẫn giới tính nên không kiểm tra được |

---

# gt_compare.py — gộp GT theo cặp và so OCR với GT (bước 1 cuối + bước 2)

Dùng cho dữ liệu theo thư mục cặp: `<ocr-root>/pair_<id>/<ảnh>.jpg.json`. Chỉ gán GT và so sánh; lọc theo module quality / spoof là bước sau.

```
python tools/gt_compare.py --gt gt.csv --ocr-root <...>/vds_api_sample/ocr --out-dir step2/
```

Tùy chọn: `--cer 0.05` (ngưỡng CER cho địa chỉ).

Nơi cấp (`issue_by`) có GT cố định: `CC_BACK_CHIP` là "BỘ CÔNG AN / MINISTRY OF PUBLIC SECURITY", `CCCD_BACK_CHIP` là "CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI".

Output:
- `step2/fields.csv`: mỗi dòng một ảnh × trường: `ocr`, `gt`, `gt_source` (MRZ/QR/checksum/fixed), `status`, `cer`, `conf`, `min_char_conf`.
  `status`: `correct`, `wrong`, `missing` (trường bắt buộc nhưng OCR trả rỗng), `no_gt`, `correct_noaccent` (tên chỉ kiểm được không dấu, phần dấu vẫn cần bước 3).
- `step2/images.csv`: mỗi dòng một ảnh, cột `status`:

| status | Ý nghĩa | Tiếp theo |
|---|---|---|
| `gt_conflict` | nhóm D, MRZ và QR lệch nhau | bỏ cặp |
| `need_human` | nhóm E, không có GT nào | người xem toàn bộ |
| `need_b4` | có trường sai/thiếu | người xác định mờ hay lỗi khác |
| `need_b3` | mọi trường có GT đều đúng, còn trường thiếu GT | bước 3 (conf + VLM) |
| `ok` | mọi trường đúng theo GT | OK |

---

# calibrate_t.py — bước 3: xác nhận trường không có GT bằng độ tin cậy + VLM

Trường được máy xác nhận "OCR đúng" khi **cả hai**: độ tin cậy ≥ T **và** VLM đọc giống OCR (sau chuẩn hóa). Không đạt thì giao người; máy không bao giờ kết luận "sai".

T được hiệu chỉnh trên các trường **có GT** (`fields.csv`), riêng cho từng nhóm: T nhỏ nhất mà độ chính xác ≥ `--target` với ít nhất `--min-accepted` ô được chấp nhận.

| Nhóm | Trường |
|---|---|
| `short` | id, birthday, expiry, issue_date, sex, nationality |
| `name` | name (chỉ ô so với tên có dấu từ QR) |
| `address` | address, birthplace, hometown, feature |

```
python tools/calibrate_t.py --fields step2/fields.csv --images step2/images.csv \
    --vlm vlm.jsonl --out-dir step3/
```

Tùy chọn: `--conf conf|min_char_conf` (cột độ tin cậy, mặc định `conf`), `--target 0.995`, `--min-accepted 1000`.

`vlm.jsonl`: mỗi dòng một ảnh `{"image": "pair_x/<ảnh>.jpg", "fields": {...}, "template_ok": true}`; dòng `template_ok: false` bị bỏ qua.

Output (`step3/`):
- `calibration.csv`: đường cong T → số ô chấp nhận, độ chính xác, độ phủ, theo nhóm.
- `thresholds.csv`: T đã chọn cho mỗi nhóm (trống nếu không T nào đạt, khi đó cả nhóm giao người).
- `b3_fields.csv`: mỗi trường cần bước 3 với `decision` (`auto_correct` / `human`) và `reason` (`vlm_differs`, `low_conf`, `no_vlm`, `no_T`).
- `b3_images.csv`: ảnh `need_b3` chuyển thành `ok` hoặc `need_human_b3` kèm danh sách trường cần người xem.
