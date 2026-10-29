# gt_extract.py — ground truth từ MRZ và QR

Bước 1 của quy trình gán nhãn mờ GTTT: lấy "đáp án" của thẻ mà không dùng OCR của chính trường cần kiểm tra.

- **MRZ** (mặt sau): đọc từ `mrz1/mrz2/mrz3` trong JSON OCR, kiểm tra 4 check digit (ICAO 9303 TD1), bóc tách số ID, họ tên, ngày sinh, giới tính, ngày hết hạn, quốc tịch.
- **QR** (`CCCD_FRONT_CHIP` và `CC_BACK_CHIP`): giải mã từ ảnh gốc, tách các trường.
- `CC_BACK_CHIP` có cả QR và MRZ trên một ảnh nên script so luôn id, ngày sinh, giới tính. CCCD thì QR ở mặt trước, MRZ ở mặt sau: cần ghép theo phiên sau.

## Cài đặt

Chỉ chạy MRZ thì không cần cài gì. Quét QR cần:

```
pip install opencv-python-headless zxing-cpp
```

`zxing-cpp` tùy chọn (không có thì dùng bộ đọc QR của OpenCV). `pyzbar` là phương án thứ ba, cần `apt install libzbar0`.

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
