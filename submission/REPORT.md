# Lab 21 — Evaluation Report

**Họ tên**: Tống Duy An  **MSSV**: 2A202601995  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 16GB (fp16 — T4 không có bf16)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (train_frac=0.9, seed 42) |
| `max_length` | 1024 (tier T4) — p95 đo được là **98**, suggested 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / **30** (cả 4 run cùng 30 step — xem `runs.csv`) |

Ghi chú `max_length`: p95 = 98 token, nên 256 là đủ; run này giữ 1024 theo mặc định tier.
Với `per_device_batch=1` và không packing, phần dư chỉ là padding không được supervise —
không mẫu nào bị cắt (max = 101 < 1024), không ảnh hưởng kết quả, chỉ tốn compute hơn
mức cần thiết. Nếu chạy lại tôi sẽ đặt 256 theo đúng số đo.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`:
`"verdict": "reasoning preserved — safe to train on traces"`, cả `open_tag_present` và
`body_present` đều `true`. Không cần xử lý gì thêm.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (từ `mask_proof.json`, `supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị mask kết thúc bằng `<|im_start|>assistant\n<think>\n\n` — tức toàn bộ system +
user turn và thẻ mở `<think>` nằm ngoài loss; loss bắt đầu từ `</think>` và JSON trả
lời. Đúng như thiết kế `assistant-only` với khối think rỗng trong dữ liệu train.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3216 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1007 |
| (c) LoRA fine-tune | **0.970** | **0.611** | 1.000 | 1356 |

**(b) có thật sự mạnh hơn (a) không?** **Có, áp đảo**: (a) không xuất nổi một JSON hợp lệ
nào (format 0.000 → target 0.000, và chậm gấp 3 vì sinh văn bản tự do dài); (b) đạt
format 1.000 và target 0.765 chỉ nhờ prompt có schema + 1 ví dụ. Riêng khoảng cách
(a)→(b) đã cho thấy phần lớn "độ khó" của bài này là ép định dạng, và prompt tốt giải
quyết được miễn phí.

**Có sửa `OPTIMIZED_PROMPT` không?** Không — SHA đóng băng `719e74d3b6232053` khớp với
`baselines_frozen.json`, baseline đo trước khi train và không đụng tới sau đó.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 modules) | 16 | 32 464 896 | 1e-4 | 0.6262 | **0.970** | 919.8 | 12.01 |
| `attn_only` | q,v (2 modules) | **283** *(matched)* | 32 456 704 | 1e-4 | **0.5381** | **0.970** | 779.6 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32 464 896 | **1e-5** | 1.5704 | 0.000 | 915.4 | 12.01 |
| `qlora` | text-linear | 16 | 32 464 896 | 1e-4 | 0.7058 | 0.940 | 977.3 | **7.09** |

Ngân sách tham số `attn_only` khớp `correct` với sai lệch 0.025% (32.457M vs 32.465M),
cả bốn run cùng `max_steps=30` — `make verify` xác nhận phép so sánh công bằng.

**4.1 — attn_only vs correct: rank hay vị trí?**
Trên tập target hai run **hoà tuyệt đối: 0.970 = 0.970**. Nhưng theo train loss thì thứ
tự khác: `attn_only` có loss **thấp hơn** (0.5381 vs 0.6262) — nếu chấm bằng loss, tôi
đã kết luận attention-only *thắng* và ship Mistake #1. Hai cột cho hai câu chuyện khác
nhau, và đó là kết quả đáng giá nhất tôi đo được: train loss đo độ khớp trên phân phối
train, không đo được năng lực tổng quát hoá sang eval. Việc hoà nhau cũng nói một điều
thật thà: với task 4-trường tương đối mẫu mực này và 30 step, cả hai cấu hình đều chạm
trần (0.97, phần sai còn lại là các ca `urgency` mơ hồ — xem §6), nên thí nghiệm này
**không đủ độ phân giải** để tách vị trí khỏi rank; muốn thấy khác biệt cần task khó
hơn hoặc ít step hơn. Lưu ý kiến trúc: Qwen3.5-4B là hybrid attention — `q_proj/v_proj`
chỉ tồn tại ở các lớp full-attention, nên `matched_rank()` phải đẩy r lên 283 (gấp ~18
lần) chỉ để bằng ngân sách — rank khổng lồ dồn vào ít lớp không mua được điểm target
nào so với r=16 trải đều: bằng chứng rằng **rank không phải đòn bẩy**.

**4.2 — wrong_lr: một con số, khác biệt gì?**
Chỉ đổi LR 1e-4 → 1e-5 (thang full-FT). Loss cuối 1.5704 — gần như không giảm so với
đầu train, trong khi `correct` cùng mọi thứ còn lại xuống 0.6262. Hậu quả ở eval là
tuyệt đối: target 0.000, format 0.000 — 30 step không đủ cho adapter học nổi việc đóng
JSON, model sinh văn bản tự do (latency 4957 ms, dài gấp ~4 lần vì không biết dừng).
Nếu chỉ nhìn đường loss mà không biết LR, tôi sẽ kết luận sai rằng "LoRA không học được
task này" hoặc "cần model to hơn / nhiều data hơn" — chính là cách danh tiếng *"LoRA
kém hơn full fine-tune"* ra đời: một hyperparameter sai thang bị đọc nhầm thành giới
hạn của phương pháp.

**4.3 — qlora: tiết kiệm gì, trả giá gì?**
VRAM đỉnh 12.01 → 7.09 GB (**−41%**), đổi lại target 0.970 → 0.940 (−3 điểm), loss cuối
cao hơn (0.7058 vs 0.6262), train chậm hơn ~6% (977 vs 920 s) và latency eval chậm hơn
~27% (1725 vs 1356 ms — dequantize khi sinh). Số đo của tôi **ủng hộ có mức độ** khuyến
nghị của vendor ("không dùng QLoRA cho Qwen3.5"): chi phí chất lượng là thật và đo được,
nhưng không thảm hoạ. Kết luận thực dụng: trên T4 16GB, fp16 LoRA vừa VRAM thoải mái
(12 GB) nên QLoRA **không cần thiết** — chỉ đáng cân nhắc khi VRAM < 10 GB, và khi đó
phải chấp nhận trả −3 điểm target cộng latency cao hơn.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.147` (tolerance 0.020) · `valid_trace_rate = 0.00`

Diễn giải: bản fine-tune **thắng thuyết phục đúng một trận và thua trận còn lại**. Trên
task đích nó vượt baseline (b) tới +0.205 (0.970 vs 0.765) với format tuyệt đối — nếu
chỉ nhìn nhóm target, đây là một thành công. Nhưng cổng hồi quy phát hiện cái giá:
điểm kiến thức phổ thông tụt 0.758 → 0.611, gấp 7 lần ngưỡng cho phép — **quên thảm
hoạ (catastrophic forgetting)** kinh điển. Cơ chế nhất quán với `valid_trace_rate = 0.00`:
dữ liệu train có khối `<think>` rỗng và 100% output là JSON 4 khoá, nên 30 step đủ để
model over-specialize — nó bỏ suy luận và nghiêng mọi câu trả lời về khuôn JSON, kể cả
câu hỏi thường. Đây chính là điều thiết kế lab muốn dạy: nếu tôi chỉ đo perplexity hoặc
chỉ đo target, tôi đã ship một model hỏng. Phán quyết FAILED là *đúng* và có giá trị
hơn một PASSED không kiểm chứng — nó nói rằng bài toán này **chưa nên deploy adapter
hiện tại**, và chỉ ra đúng thuốc chữa: trộn 1–5% dữ liệu phổ thông (replay) vào tập
train (deck §14.3) rồi train lại, thay vì chỉnh prompt hay đổi cấu hình LoRA.

---

## 6. Định tính — bắt buộc có cả ca THUA

Nguồn: `results/qualitative.json` (6 ca thua score 0.75, 44 ca đúng 4/4). NB5 không lưu
dự đoán từng-mẫu của baseline (b), chỉ lưu điểm nhóm — cột (b) để "—".

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "chuột không dây… Cho tôi trả lại. **Gấp**." (i=0) | doi_tra · **cao** | — | doi_tra · cao ✓ 4/4 | ✅ FT thắng — bắt đúng cue "Gấp" |
| 2 | "đèn bàn LED… Hoàn tiền. **Quá hạn rồi**." (i=2) | hoan_tien · **cao** | — | hoan_tien · cao ✓ 4/4 | ✅ FT thắng — suy ra khẩn cấp từ "quá hạn" |
| 3 | "bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện**." (i=3) | hoan_tien · **thap** | — | urgency = `trung_binh` ✗ | ❌ **FT thua** (3/4) |
| 4 | "nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện**." (i=5) | san_pham_loi · **thap** | — | urgency = `trung_binh` ✗ | ❌ **FT thua** (3/4) |
| 5 | "áo khoác gió… Bị lỗi. **Khi nào tiện**." (i=12) | san_pham_loi · **thap** | — | urgency = `trung_binh` ✗ | ❌ **FT thua** (3/4) |
| 6 | "đèn bàn LED… Sai màu. **Khi nào tiện**." (i=46) | san_pham_loi · **thap** | — | urgency = `trung_binh` ✗ | ❌ **FT thua** (3/4) |

**Mẫu chung ở các ca thua: rõ và duy nhất.** Cả 6/6 ca sai đều là ticket chứa cụm
"**Khi nào tiện**" (nhãn `urgency=thap`) bị model đoán thành `trung_binh`. Các trường
còn lại (intent, product, sentiment) đúng hết. Model học tốt cue khẩn cấp tường minh
("Gấp", "Ngay lập tức", "Quá hạn") nhưng rơi về giá trị giữa `trung_binh` khi cue là
phép lịch sự gián tiếp — gợi ý tập train thiếu/mỏng biến thể "Khi nào tiện" cho lớp
`thap`. Đây là lỗi *dữ liệu*, sửa bằng thêm mẫu, không phải lỗi cấu hình.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Không nên deploy adapter này — và chính pipeline của lab cho tôi bằng
chứng để nói vậy một cách định lượng. Fine-tune thắng baseline (b) +0.205 điểm target
với format 1.000, nhưng cổng hồi quy cho thấy nó trả giá bằng −0.147 điểm năng lực
phổ thông (gấp 7 lần ngưỡng 0.02) và `valid_trace_rate = 0`: model đã đánh đổi khả
năng suy luận tổng quát lấy một khuôn JSON. Với use-case triage nội bộ chỉ-một-việc,
có thể biện luận chấp nhận; nhưng phán quyết đúng là train lại với 1–5% replay data
trước đã, vì chi phí sửa thấp mà rủi ro giữ nguyên thì cao. Về đòn bẩy: thí nghiệm NB4
cho thấy **mask và dữ liệu mới là đòn bẩy thật**, không phải cấu hình LoRA — mask sai
thì mọi thứ sau vô nghĩa (NB1), prompt tốt đã lấy được 0.765 "miễn phí" (NB2), LR sai
thang phá sạch mọi thứ (wrong_lr 0.000), trong khi vị trí-vs-rank ở cùng ngân sách
tham số chỉ tạo khác biệt 0.000 điểm trên task này, và 6 ca sai còn lại đều truy về
một lỗ hổng dữ liệu ("Khi nào tiện"). Thứ tự ưu tiên khi làm tiếp: dữ liệu > LR đúng
thang > mọi biến thể LoRA khác.

**Ba điều tôi học được:**
1. Train loss xếp hạng ngược với chất lượng thật: `attn_only` có loss thấp hơn `correct`
   (0.5381 < 0.6262) nhưng hoà trên target, còn `wrong_lr` loss 1.57 nhìn "vẫn đang
   giảm" mà eval bằng 0 tuyệt đối. Không bao giờ chấm run bằng loss thay cho eval đích.
2. Quên thảm hoạ xảy ra nhanh hơn tôi nghĩ rất nhiều: chỉ 30 step trên 225 mẫu JSON
   thuần đã đủ kéo regression tụt 0.147 và xoá sạch thói quen suy luận — và nếu bộ eval
   không có nhóm regression thì tôi hoàn toàn không có cách nào biết.
3. Cùng "attention-only" nhưng trên kiến trúc hybrid (6/24 lớp full-attention) nghĩa là
   matched_rank phải lên 283 — bài học là phải đọc `layer_types` của chính model mình
   train, không được giả định kiến trúc từ tên gọi chung "transformer".

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu phổ thông (replay) vào tập
train và chạy lại NB3+NB5 để xem regression gate chuyển PASS mà target giữ ≥0.95 không;
thêm ~20 mẫu train chứa biến thể "Khi nào tiện"/"lúc nào cũng được" cho lớp `urgency=thap`;
và chạy B3 (`MASK_MODE=response-only` vs `assistant-only`) để đo trực tiếp reasoning-trace
collapse qua `valid_trace_rate` của hai bản.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/ipgalone321/lab21-2A202601995-qwen35-triage-vi
      (adapter + `results/` + REPORT.md + model card ghi rõ base `unsloth/Qwen3.5-4B` và verdict FAILED)
