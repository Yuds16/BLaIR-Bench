# Data Processing Scripts

Run commands from the project root.

```bash
# 1) Amazon-C4 titles-only
python data_processing/amazon_c4_titles_only.py \
  --output outputs/amazon_c4_titles_only_na2period.jsonl

# 2) ESCI step 1: process and sample
python data_processing/process_esci.py \
  --output-dir outputs/esci

# 3) ESCI step 2: titles-only (must run after step 1)
python data_processing/esci_titles_only.py \
  --input-jsonl outputs/esci/sampled_item_metadata_esci.jsonl \
  --output-jsonl outputs/esci/esci_titles_only_na2period.jsonl

# 4) Reddit-Movie titles-only CSVs
python data_processing/reddit_movie_titles_only.py \
  --output-dir outputs/reddit_movie
```
